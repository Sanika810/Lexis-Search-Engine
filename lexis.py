"""Lexis - a search engine for the Reuters-21578 archive (CSL 436 programming assignment).

Layout of this file
  1. text analysis        one Analyzer used for documents AND queries
  2. byte codecs          variable-byte coding, posting-list encode/decode (with skip table)
  3. index construction   SPIMI and BSBI block writers, k-way merge, 4-shard parallel build
  4. on-disk index        front-coded dictionary + gap/vbyte postings with positions and skips
  5. engine               Boolean / phrase / proximity / wildcard (permuterm) / spelling /
                          incremental add + delete (delta segment + tombstones)
  6. ranker               zoned tf-idf cosine, champion lists, index elimination
"""
import bisect, heapq, json, math, os, re, sys, time
from array import array
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np

# ----------------------------------------------------------------------------- 1. analysis
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = frozenset("""a about after all also an and any are as at be been but by can for from had has have he her his
i if in into is it its more no not of on one or said say says she so such than that the their there they this to up was
we were what which who will with would you""".split())


class Analyzer:
    """Tokenise (lower-case, runs of [a-z0-9]), optional stop-word removal, optional Porter stemming.
    Stop words are dropped but token POSITIONS are preserved, so phrase/proximity distances stay true."""

    def __init__(self, stem=True, stop=True):
        self.stem, self.stop, self._cache = stem, stop, {}
        if stem:
            from nltk.stem import PorterStemmer
            self._st = PorterStemmer()

    def norm(self, tok):
        if self.stop and tok in STOP_WORDS:
            return None
        if self.stem:
            s = self._cache.get(tok)
            if s is None:
                s = self._cache[tok] = self._st.stem(tok)
            return s
        return tok

    def doc_terms(self, text):
        """-> ([(term, position)], n_raw_tokens, n_title_tokens). Title = first line."""
        text = text.lower()
        toks = TOKEN_RE.findall(text)
        ntitle = len(TOKEN_RE.findall(text.split("\n", 1)[0]))
        norm = self.norm
        out = []
        for p, t in enumerate(toks):
            n = norm(t)
            if n is not None:
                out.append((n, p))
        return out, len(toks), ntitle

    def query_terms(self, text):
        out = []
        for t in TOKEN_RE.findall(text.lower()):
            n = self.norm(t)
            if n is not None:
                out.append(n)
        return out


# ----------------------------------------------------------------------------- 2. codecs
def venc(n, out):
    """variable-byte: 7 payload bits per byte, high bit set on the LAST byte of a number."""
    while n >= 128:
        out.append(n & 127)
        n >>= 7
    out.append(n | 128)


def vdec(buf, i):
    x = s = 0
    while True:
        b = buf[i]
        i += 1
        if b >= 128:
            return x | ((b & 127) << s), i
        x |= b << s
        s += 7


def encode_list(docs, tfs, poss, skips=True):
    """Posting list -> bytes.  Layout: [nskips][(doc gap, offset delta)*nskips] body
    body = per posting: doc gap, tf, tf position gaps.  A skip entry every S=int(sqrt(df)) postings
    stores the absolute docID of that posting and the body offset of its tf field.
    Returns (blob, bytes_without_positions) - the second is the size of a plain (doc,tf) index."""
    df = len(docs)
    S = int(math.sqrt(df)) if (skips and df >= 16) else 0
    body, sk, prev, plain = bytearray(), [], 0, 0
    for k in range(df):
        d, ps = docs[k], poss[k]
        a = len(body)
        venc(d - prev, body)
        prev = d
        if S and k and k % S == 0:
            sk.append((d, len(body)))
        venc(tfs[k], body)
        plain += len(body) - a
        pp = 0
        for p in ps:
            venc(p - pp, body)
            pp = p
    head = bytearray()
    venc(len(sk), head)
    pd = po = 0
    for d, o in sk:
        venc(d - pd, head)
        venc(o - po, head)
        pd, po = d, o
    return bytes(head) + bytes(body), plain + len(head)


def decode_postings(buf, i, df, want_pos=True):
    """Decode a posting BODY (no skip header). -> docs, tfs, poss|None, next index."""
    docs, tfs = [0] * df, [0] * df
    poss = [None] * df if want_pos else None
    d = 0
    for k in range(df):
        x = s = 0
        while True:
            b = buf[i]
            i += 1
            if b >= 128:
                x |= (b & 127) << s
                break
            x |= b << s
            s += 7
        d += x
        docs[k] = d
        x = s = 0
        while True:
            b = buf[i]
            i += 1
            if b >= 128:
                x |= (b & 127) << s
                break
            x |= b << s
            s += 7
        tfs[k] = x
        if want_pos:
            p, lst = 0, []
            for _ in range(x):
                g = s = 0
                while True:
                    b = buf[i]
                    i += 1
                    if b >= 128:
                        g |= (b & 127) << s
                        break
                    g |= b << s
                    s += 7
                p += g
                lst.append(p)
            poss[k] = lst
        else:
            for _ in range(x):
                while buf[i] < 128:
                    i += 1
                i += 1
    return docs, tfs, poss, i


def parse_header(buf, i):
    n, i = vdec(buf, i)
    sk, d, o = [], 0, 0
    for _ in range(n):
        g, i = vdec(buf, i)
        og, i = vdec(buf, i)
        d += g
        o += og
        sk.append((d, o))
    return sk, i


# ----------------------------------------------------------------------------- 3. construction
TERM_OVERHEAD = 64          # accounted bytes for a new dictionary entry in the SPIMI buffer
SPIMI_INT = 4               # accounted bytes per buffered int (docID, tf or position)
BSBI_REC = 8                # accounted bytes per buffered (termID, docID, pos) record


def _rec(term, df, data, out):
    tb = term.encode("utf-8")
    venc(len(tb), out)
    out += tb
    venc(df, out)
    venc(len(data), out)
    out += data


def _read_block(path):
    buf = open(path, "rb").read()
    i, n = 0, len(buf)
    while i < n:
        ln, i = vdec(buf, i)
        term = buf[i:i + ln].decode("utf-8")
        i += ln
        df, i = vdec(buf, i)
        nb, i = vdec(buf, i)
        yield term, df, buf[i:i + nb]
        i += nb


def _encode_arr(arr):
    out, i, n, prev, df = bytearray(), 0, len(arr), 0, 0
    while i < n:
        d, tf = arr[i], arr[i + 1]
        venc(d - prev, out)
        prev = d
        venc(tf, out)
        pp = 0
        for p in arr[i + 2:i + 2 + tf]:
            venc(p - pp, out)
            pp = p
        i += 2 + tf
        df += 1
    return df, out


def corpus_files(corpus_dir):
    return sorted(f for f in os.listdir(corpus_dir) if f.endswith(".txt"))


def _read_doc(corpus_dir, fn):
    with open(os.path.join(corpus_dir, fn), encoding="utf-8", errors="replace") as f:
        return f.read()


def _front_code(entries, B=16):
    """entries: [(term, df, nbytes)] sorted. -> (dict bytes, [(first_term, block offset)])"""
    out, idx, off = bytearray(), [], 0
    for s in range(0, len(entries), B):
        chunk = entries[s:s + B]
        idx.append((chunk[0][0], len(out)))
        venc(len(chunk), out)
        venc(off, out)
        prev = ""
        for j, (t, df, nb) in enumerate(chunk):
            tb = t.encode("utf-8")
            if j == 0:
                venc(len(tb), out)
                out += tb
            else:
                c = 0
                pb = prev.encode("utf-8")
                while c < len(pb) and c < len(tb) and pb[c] == tb[c]:
                    c += 1
                venc(c, out)
                venc(len(tb) - c, out)
                out += tb[c:]
            venc(df, out)
            venc(nb, out)
            off += nb
            prev = t
    return bytes(out), idx


def merge_to_index(sources, out_dir, skips=True):
    """K-way merge of sorted (term, df, body) streams whose docID ranges are increasing in source order
    -> one finished index (front-coded dictionary + postings with skip tables)."""
    os.makedirs(out_dir, exist_ok=True)
    pf = open(os.path.join(out_dir, "postings.bin"), "wb")
    entries, plain_tot, post_tot, n_post, n_occ = [], 0, 0, 0, 0
    def tag(i, s):
        for t, df, d in s:
            yield t, i, df, d
    tagged = [tag(i, s) for i, s in enumerate(sources)]
    cur, parts = None, []

    def flush():
        nonlocal plain_tot, post_tot, n_post, n_occ
        docs, tfs, poss = [], [], []
        for df, data in parts:
            d, t, p, _ = decode_postings(data, 0, df)
            docs += d
            tfs += t
            poss += p
        blob, plain = encode_list(docs, tfs, poss, skips)
        pf.write(blob)
        entries.append((cur, len(docs), len(blob)))
        plain_tot += plain
        post_tot += len(blob)
        n_post += len(docs)
        n_occ += sum(tfs)

    for term, i, df, data in heapq.merge(*tagged):
        if term != cur:
            if cur is not None:
                flush()
            cur, parts = term, []
        parts.append((df, data))
    if cur is not None:
        flush()
    pf.close()
    dbytes, idx = _front_code(entries)
    open(os.path.join(out_dir, "dict.bin"), "wb").write(dbytes)
    json.dump(idx, open(os.path.join(out_dir, "dict.idx"), "w"))
    raw_dict = sum(len(t.encode()) + 1 + 8 for t, _, _ in entries)   # terms + NUL + 4B df + 4B pointer
    return dict(nterms=len(entries), postings_bytes=post_tot, plain_bytes=plain_tot, dict_bytes=len(dbytes),
                dict_raw_bytes=raw_dict, n_postings=n_post, n_occurrences=n_occ)


def build_index(corpus_dir, out_dir, algo="spimi", mem_limit=2 << 20, stem=False, stop=False,
                doc_range=None, skips=True, keep_blocks=False):
    """Stream articles, flush sorted blocks whenever the (accounted) buffer reaches mem_limit bytes,
    then merge the blocks. algo = 'spimi' (term -> postings dictionary per block)
    or 'bsbi' (vocabulary pre-pass, buffer of (termID,docID,pos) records sorted per block)."""
    t0 = time.perf_counter()
    files = corpus_files(corpus_dir)
    lo, hi = doc_range or (0, len(files))
    an = Analyzer(stem, stop)
    os.makedirs(out_dir, exist_ok=True)
    bdir = os.path.join(out_dir, "blocks")
    os.makedirs(bdir, exist_ok=True)
    blocks, block_bytes = [], 0
    names, tlen, dlen = [], [], []

    def write_block(records):
        nonlocal block_bytes
        path = os.path.join(bdir, "b%04d.blk" % len(blocks))
        out = bytearray()
        for term, df, data in records:
            _rec(term, df, data, out)
        open(path, "wb").write(out)
        blocks.append(path)
        block_bytes += len(out)

    def docs_stream():
        for d in range(lo, hi):
            terms, n, nt = an.doc_terms(_read_doc(corpus_dir, files[d]))
            names.append(files[d][:-4])
            tlen.append(nt)
            dlen.append(n)
            yield d, terms

    if algo == "spimi":
        mem, used = {}, 0
        for d, terms in docs_stream():
            loc = defaultdict(list)
            for t, p in terms:
                loc[t].append(p)
            for t, ps in loc.items():
                a = mem.get(t)
                if a is None:
                    a = mem[t] = array("I")
                    used += TERM_OVERHEAD
                a.append(d)
                a.append(len(ps))
                a.extend(ps)
                used += SPIMI_INT * (2 + len(ps))
            if used >= mem_limit:
                write_block((t, *_encode_arr(mem[t])) for t in sorted(mem))
                mem, used = {}, 0
        if mem:
            write_block((t, *_encode_arr(mem[t])) for t in sorted(mem))
    else:  # bsbi
        vocab = set()
        for d in range(lo, hi):
            vocab.update(t for t, _ in an.doc_terms(_read_doc(corpus_dir, files[d]))[0])
        id2t = sorted(vocab)
        t2id = {t: i for i, t in enumerate(id2t)}
        del vocab
        buf = array("Q")

        def flush_bsbi():
            keys = sorted(buf)
            recs, i, n = [], 0, len(keys)
            while i < n:
                tid = keys[i] >> 40
                arr = array("I")
                while i < n and keys[i] >> 40 == tid:
                    d = (keys[i] >> 20) & 0xFFFFF
                    ps = []
                    while i < n and keys[i] >> 40 == tid and (keys[i] >> 20) & 0xFFFFF == d:
                        ps.append(keys[i] & 0xFFFFF)
                        i += 1
                    arr.append(d)
                    arr.append(len(ps))
                    arr.extend(ps)
                recs.append((id2t[tid], *_encode_arr(arr)))
            write_block(recs)

        for d, terms in docs_stream():
            for t, p in terms:
                buf.append((t2id[t] << 40) | (d << 20) | p)
            if len(buf) * BSBI_REC >= mem_limit:
                flush_bsbi()
                buf = array("Q")
        if buf:
            flush_bsbi()

    t_blocks = time.perf_counter() - t0
    import tracemalloc
    peak_blocks = None
    if tracemalloc.is_tracing():
        peak_blocks = tracemalloc.get_traced_memory()[1]
        tracemalloc.reset_peak()
    st = merge_to_index((_read_block(p) for p in blocks), out_dir, skips)
    for p in blocks:
        if not keep_blocks:
            os.remove(p)
    if not keep_blocks:
        os.rmdir(bdir)
    if peak_blocks is not None:
        st.update(peak_blocks=peak_blocks, peak_merge=tracemalloc.get_traced_memory()[1])
    st.update(blocks=len(blocks), block_bytes=block_bytes, seconds=time.perf_counter() - t0,
              seconds_blocks=t_blocks, algo=algo, mem_limit=mem_limit, stem=stem, stop=stop)
    return st, dict(names=names, title_len=tlen, doc_len=dlen)


def _shard_job(args):
    return build_index(*args[0], **args[1])


def build_sharded(corpus_dir, out_dir, shards=4, algo="spimi", mem_limit=2 << 20, stem=False, stop=False):
    """Contiguous docID ranges -> `shards` worker processes each build their own index (own memory budget),
    then the shard indexes are merged into the single final index."""
    import multiprocessing as mp
    t0 = time.perf_counter()
    n = len(corpus_files(corpus_dir))
    cut = [round(i * n / shards) for i in range(shards + 1)]
    jobs = [((corpus_dir, os.path.join(out_dir, "shard%d" % s)),
             dict(algo=algo, mem_limit=mem_limit, stem=stem, stop=stop, doc_range=(cut[s], cut[s + 1]),
                  skips=False)) for s in range(shards)]
    if shards == 1:
        res = [_shard_job(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(shards) as pool:
            res = pool.map(_shard_job, jobs)
    t_shards = time.perf_counter() - t0
    srcs = [Index(os.path.join(out_dir, "shard%d" % s), docs=False).iter_raw() for s in range(shards)]
    st = merge_to_index(srcs, out_dir)
    docs = {k: sum((r[1][k] for r in res), []) for k in ("names", "title_len", "doc_len")}
    json.dump(docs, open(os.path.join(out_dir, "docs.json"), "w"))
    st.update(shards=shards, algo=algo, mem_limit=mem_limit, stem=stem, stop=stop,
              blocks=sum(r[0]["blocks"] for r in res), block_bytes=sum(r[0]["block_bytes"] for r in res),
              seconds=time.perf_counter() - t0, seconds_shards=t_shards,
              ndocs=len(docs["names"]), ntokens=sum(docs["doc_len"]))
    json.dump(st, open(os.path.join(out_dir, "meta.json"), "w"))
    import shutil
    for s in range(shards):
        shutil.rmtree(os.path.join(out_dir, "shard%d" % s))
    return st


# ----------------------------------------------------------------------------- 4. on-disk index
class Index:
    def __init__(self, path, docs=True):
        self.path = path
        self.dict = open(os.path.join(path, "dict.bin"), "rb").read()
        self.post = open(os.path.join(path, "postings.bin"), "rb").read()
        idx = json.load(open(os.path.join(path, "dict.idx")))
        self.first = [t for t, _ in idx]
        self.boff = [o for _, o in idx]
        mp = os.path.join(path, "meta.json")
        self.meta = json.load(open(mp)) if os.path.exists(mp) else {}
        self.analyzer = Analyzer(self.meta.get("stem", False), self.meta.get("stop", False))
        if docs:
            d = json.load(open(os.path.join(path, "docs.json")))
            self.names, self.title_len, self.doc_len = d["names"], d["title_len"], d["doc_len"]
            self.ndocs = len(self.names)
        self._vocab = None

    def _block(self, b):
        buf, i = self.dict, self.boff[b]
        n, i = vdec(buf, i)
        base, i = vdec(buf, i)
        out, prev, off = [], b"", base
        for j in range(n):
            if j == 0:
                ln, i = vdec(buf, i)
                tb = buf[i:i + ln]
                i += ln
            else:
                c, i = vdec(buf, i)
                ln, i = vdec(buf, i)
                tb = prev[:c] + buf[i:i + ln]
                i += ln
            df, i = vdec(buf, i)
            nb, i = vdec(buf, i)
            out.append((tb.decode("utf-8"), df, off, nb))
            off += nb
            prev = tb
        return out

    def lookup(self, term):
        b = bisect.bisect_right(self.first, term) - 1
        if b < 0:
            return None
        for t, df, off, nb in self._block(b):
            if t == term:
                return df, off, nb
        return None

    def items(self):
        for b in range(len(self.first)):
            yield from self._block(b)

    def vocab(self):
        if self._vocab is None:
            self._vocab = {t: (df, off, nb) for t, df, off, nb in self.items()}
        return self._vocab

    def read(self, term, want_pos=True):
        e = self.lookup(term)
        if e is None:
            return [], [], []
        df, off, nb = e
        _, i = parse_header(self.post, off)
        d, t, p, _ = decode_postings(self.post, i, df, want_pos)
        return d, t, p

    def skip_table(self, term):
        df, off, nb = self.lookup(term)
        return parse_header(self.post, off)[0]

    def iter_raw(self):
        for t, df, off, nb in self.items():
            _, i = parse_header(self.post, off)
            yield t, df, self.post[i:off + nb]


class MemSegment:
    """Small in-memory index for documents added after the main index was built."""

    def __init__(self, analyzer):
        self.an = analyzer
        self.p = {}

    def add(self, docid, text):
        terms, n, nt = self.an.doc_terms(text)
        loc = defaultdict(list)
        for t, pos in terms:
            loc[t].append(pos)
        for t, ps in loc.items():
            d, tf, po = self.p.setdefault(t, ([], [], []))
            d.append(docid)
            tf.append(len(ps))
            po.append(ps)
        return n, nt

    def read(self, term, want_pos=True):
        return self.p.get(term, ([], [], []))

    def terms(self):
        return self.p.keys()


# ----------------------------------------------------------------------------- 5. engine
def intersect_skip(a, b, spacing=None, stats=None):
    """AND-merge of two sorted docID lists with skip pointers every `spacing` entries
    (None -> sqrt(len) per list, 0 -> no skips). stats['cmp'] counts docID comparisons."""
    sa = int(math.sqrt(len(a))) if spacing is None else spacing
    sb = int(math.sqrt(len(b))) if spacing is None else spacing
    i = j = c = 0
    la, lb, res = len(a), len(b), []
    while i < la and j < lb:
        c += 1
        x, y = a[i], b[j]
        if x == y:
            res.append(x)
            i += 1
            j += 1
        elif x < y:
            if sa and i % sa == 0 and i + sa < la:
                c += 1
                if a[i + sa] <= y:
                    while sa and i % sa == 0 and i + sa < la and a[i + sa] <= y:
                        i += sa
                        c += 1
                    continue
            i += 1
        else:
            if sb and j % sb == 0 and j + sb < lb:
                c += 1
                if b[j + sb] <= x:
                    while sb and j % sb == 0 and j + sb < lb and b[j + sb] <= x:
                        j += sb
                        c += 1
                    continue
            j += 1
    if stats is not None:
        stats["cmp"] = stats.get("cmp", 0) + c
    return res


def _union(a, b):
    return sorted(set(a) | set(b))


def _diff(a, b):
    s = set(b)
    return [x for x in a if x not in s]


QTOK = re.compile(r'"([^"]*)"|(\()|(\))|/(\d+)|([^\s()"]+)')


def parse_query(q):
    """Boolean grammar: OR < AND (also implicit) < NOT < proximity (a /k b) < atom; atoms are words,
    "phrases", wildcard words and parenthesised sub-expressions."""
    toks = []
    for m in QTOK.finditer(q):
        if m.group(1) is not None:
            toks.append(("PHRASE", m.group(1)))
        elif m.group(2):
            toks.append(("(", None))
        elif m.group(3):
            toks.append((")", None))
        elif m.group(4):
            toks.append(("PROX", int(m.group(4))))
        else:
            w = m.group(5)
            toks.append((w, None) if w in ("AND", "OR", "NOT") else ("WORD", w))
    pos = 0

    def peek():
        return toks[pos][0] if pos < len(toks) else None

    def orx():
        nonlocal pos
        n = andx()
        while peek() == "OR":
            pos += 1
            n = ("or", n, andx())
        return n

    def andx():
        nonlocal pos
        n = notx()
        while peek() in ("AND", "NOT", "WORD", "PHRASE", "("):
            if peek() == "AND":
                pos += 1
            n = ("and", n, notx())
        return n

    def notx():
        nonlocal pos
        if peek() == "NOT":
            pos += 1
            return ("not", notx())
        return proxx()

    def proxx():
        nonlocal pos
        a = atom()
        if peek() == "PROX":
            k = toks[pos][1]
            pos += 1
            return ("prox", a, atom(), k)
        return a

    def atom():
        nonlocal pos
        kind, val = toks[pos]
        pos += 1
        if kind == "(":
            n = orx()
            pos += 1  # ')'
            return n
        if kind == "PHRASE":
            return ("phrase", TOKEN_RE.findall(val.lower()))
        if "*" in val:
            return ("wild", val.lower())
        return ("term", val.lower())

    return orx()


class Lexis:
    """Query engine over a finished index + delta segment (added docs) + tombstones (deleted docs)."""

    def __init__(self, index_dir):
        self.dir = index_dir
        self.main = Index(index_dir)
        self.an = self.main.analyzer
        self.delta = MemSegment(self.an)
        self.names = list(self.main.names)
        self.deleted = set()
        self.cache = {}
        self._perm = None
        self._spell = None
        dp = os.path.join(index_dir, "delta.jsonl")
        if os.path.exists(dp):
            for line in open(dp):
                r = json.loads(line)
                self._add(r["name"], r["text"])
        tp = os.path.join(index_dir, "deleted.json")
        if os.path.exists(tp):
            self.deleted = set(json.load(open(tp)))

    # ---- updates (no rebuild)
    def _add(self, name, text):
        did = len(self.names)
        self.names.append(name)
        self.delta.add(did, text)
        self.cache.clear()
        self._perm = self._spell = None
        return did

    def add_document(self, name, text, persist=True):
        did = self._add(name, text)
        if persist:
            with open(os.path.join(self.dir, "delta.jsonl"), "a") as f:
                f.write(json.dumps(dict(name=name, text=text)) + "\n")
        return did

    def delete_document(self, name, persist=True):
        did = self.names.index(name)
        self.deleted.add(did)
        self.cache.clear()
        if persist:
            json.dump(sorted(self.deleted), open(os.path.join(self.dir, "deleted.json"), "w"))
        return did

    # ---- posting access
    def _read(self, term, want_pos):
        key = (term, want_pos)
        r = self.cache.get(key)
        if r is None:
            d, t, p = self.main.read(term, want_pos)
            d2, t2, p2 = self.delta.read(term, want_pos)
            d, t, p = list(d) + d2, list(t) + t2, (list(p) + p2) if want_pos else None
            if self.deleted:
                keep = [k for k, x in enumerate(d) if x not in self.deleted]
                d, t = [d[k] for k in keep], [t[k] for k in keep]
                p = [p[k] for k in keep] if want_pos else None
            r = self.cache[key] = (d, t, p)
        return r

    def docs_of(self, term):
        return self._read(term, False)[0]

    def pos_of(self, term):
        d, _, p = self._read(term, True)
        return dict(zip(d, p))

    def live_docs(self):
        return [d for d in range(len(self.names)) if d not in self.deleted]

    def vocabulary(self):
        v = {t: df for t, (df, _, _) in self.main.vocab().items()}
        for t in self.delta.terms():
            v[t] = v.get(t, 0) + len(self.delta.p[t][0])
        return v

    # ---- wildcard: permuterm index
    def _permuterm(self):
        if self._perm is None:
            rows = []
            for t in self.vocabulary():
                s = t + "$"
                rows += [(s[i:] + s[:i], t) for i in range(len(s))]
            rows.sort()
            self._perm = ([r[0] for r in rows], [r[1] for r in rows])
        return self._perm

    def expand_wildcard(self, pat):
        """terms matching pat (one or more *). The last * is rotated to the end so that a prefix range scan on
        the permuterm index narrows the candidates; a regex then enforces the remaining * pieces."""
        p = pat + "$"
        i = p.rindex("*")
        key = (p[i + 1:] + p[:i]).split("*")[0]
        keys, terms = self._permuterm()
        lo = bisect.bisect_left(keys, key)
        cand = set()
        while lo < len(keys) and keys[lo].startswith(key):
            cand.add(terms[lo])
            lo += 1
        rx = re.compile("^" + ".*".join(re.escape(x) for x in pat.split("*")) + "$")
        return sorted(t for t in cand if rx.match(t))

    # ---- spelling
    def suggest(self, word, n=3):
        if self._spell is None:
            self._spell = SpellChecker(self.vocabulary())
        return self._spell.suggest(word, n)

    # ---- Boolean evaluation
    def search(self, q):
        """-> (sorted docIDs, suggestions {oov term: [corrections]})"""
        tree = parse_query(q)
        sug = {}
        res = self._eval(tree, sug)
        return res, sug

    def _term_docs(self, w, sug):
        ts = self.an.query_terms(w)
        if not ts:
            return []
        d = self.docs_of(ts[0])
        if not d and ts[0] not in self.vocabulary():
            sug[w] = self.suggest(ts[0])
        return d

    def _eval(self, n, sug, stats=None):
        k = n[0]
        if k == "term":
            return self._term_docs(n[1], sug)
        if k == "wild":
            out = set()
            for t in self.expand_wildcard(n[1]):
                out.update(self.docs_of(t))
            return sorted(out)
        if k == "phrase":
            return self._phrase(n[1])
        if k == "prox":
            return self._prox(n[1], n[2], n[3])
        if k == "or":
            return _union(self._eval(n[1], sug), self._eval(n[2], sug))
        if k == "not":
            return _diff(self.live_docs(), self._eval(n[1], sug))
        if k == "and":
            l, r = n[1], n[2]
            if r[0] == "not":
                return _diff(self._eval(l, sug), self._eval(r[1], sug))
            if l[0] == "not":
                return _diff(self._eval(r, sug), self._eval(l[1], sug))
            return intersect_skip(self._eval(l, sug), self._eval(r, sug), stats=stats)
        raise ValueError(n)

    def _phrase(self, words):
        ts = [self.an.query_terms(w) for w in words]
        ts = [t[0] for t in ts if t]
        if not ts:
            return []
        if len(ts) == 1:
            return self.docs_of(ts[0])
        pos = [self.pos_of(t) for t in ts]
        cand = pos[0].keys()
        for p in pos[1:]:
            cand = cand & p.keys()
        out = []
        for d in sorted(cand):
            sets = [set(p[d]) for p in pos[1:]]
            if any(all((x + k + 1) in sets[k] for k in range(len(sets))) for x in pos[0][d]):
                out.append(d)
        return out

    def _prox(self, a, b, k):
        ta, tb = self.an.query_terms(a[1]), self.an.query_terms(b[1])
        if not ta or not tb:
            return []
        pa, pb = self.pos_of(ta[0]), self.pos_of(tb[0])
        out = []
        for d in sorted(pa.keys() & pb.keys()):
            ys = pb[d]
            for x in pa[d]:
                j = bisect.bisect_left(ys, x - k)
                while j < len(ys) and ys[j] <= x + k:
                    if ys[j] != x:
                        break
                    j += 1
                if j < len(ys) and ys[j] <= x + k:
                    out.append(d)
                    break
        return out


def dl_distance(s, t):
    """Damerau-Levenshtein (optimal string alignment) distance."""
    n, m = len(s), len(t)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if s[i - 1] == t[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and s[i - 1] == t[j - 2] and s[i - 2] == t[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[n][m]


class SpellChecker:
    """Bigram (k-gram) index to shortlist candidates by Jaccard overlap, then edit distance, ties by frequency."""

    def __init__(self, vocab):
        self.df = vocab
        self.g = defaultdict(list)
        self.ng = {}
        for t in vocab:
            gs = self.grams(t)
            self.ng[t] = len(gs)
            for x in gs:
                self.g[x].append(t)

    @staticmethod
    def grams(w):
        s = "$" + w + "$"
        return {s[i:i + 2] for i in range(len(s) - 1)}

    def suggest(self, word, n=3):
        gs = self.grams(word)
        cnt = Counter()
        for x in gs:
            for t in self.g.get(x, ()):
                cnt[t] += 1
        scored = sorted(((c / (len(gs) + self.ng[t] - c), t) for t, c in cnt.items()), reverse=True)[:150]
        cand = [(dl_distance(word, t), -self.df[t], t) for j, t in scored if j >= 0.15]
        cand.sort()
        return [t for dist, _, t in cand[:n] if dist <= 3]


# ----------------------------------------------------------------------------- 6. ranking
class Config:
    def __init__(self, sub=True, idf=True, norm=True, zones=True, gt=0.3):
        self.sub, self.idf, self.norm, self.zones, self.gt = sub, idf, norm, zones, gt

    def replace(self, **kw):
        c = Config(self.sub, self.idf, self.norm, self.zones, self.gt)
        c.__dict__.update(kw)
        return c

    def __repr__(self):
        return "Config(%s)" % ", ".join("%s=%s" % kv for kv in self.__dict__.items())


class Ranker:
    """Cosine similarity over tf-idf vectors with title and body as separate zones.
       doc weight  w(t,d,z) = (1+log tf) * idf        score = gt*cos(q, d_title) + (1-gt)*cos(q, d_body)
       query weight (1+log tf_q)  [ltc.ltn]. Accumulators are filled only from the postings of the query terms,
       so documents containing no query term are never scored."""

    def __init__(self, index):
        self.ix = index
        self.N = index.ndocs
        tl = index.title_len
        self.T = {}
        for t, (df, off, nb) in index.vocab().items():
            d, tf, ps = index.read(t)
            ft = [sum(1 for p in l if p < tl[x]) for x, l in zip(d, ps)]
            tft = np.array(ft, dtype=np.float32)
            tfa = np.array(tf, dtype=np.float32)
            self.T[t] = (np.array(d, dtype=np.int32), tft, tfa - tft, math.log10(self.N / df))
        self._norm, self._champ = {}, {}

    def _w(self, tf, sub):
        if sub:
            return np.where(tf > 0, 1.0 + np.log(np.maximum(tf, 1)), 0.0)
        return tf

    def norms(self, cfg):
        key = (cfg.sub, cfg.idf, cfg.zones)
        if key not in self._norm:
            a, b = np.zeros(self.N), np.zeros(self.N)
            for d, tft, tfb, idf in self.T.values():
                k = idf if cfg.idf else 1.0
                if cfg.zones:
                    a += np.bincount(d, (self._w(tft, cfg.sub) * k) ** 2, self.N)
                    b += np.bincount(d, (self._w(tfb, cfg.sub) * k) ** 2, self.N)
                else:
                    a += np.bincount(d, (self._w(tft + tfb, cfg.sub) * k) ** 2, self.N)
            a, b = np.sqrt(a), np.sqrt(b)
            a[a == 0] = 1
            b[b == 0] = 1
            self._norm[key] = (a, b)
        return self._norm[key]

    def champions(self, term, r):
        key = (term, r)
        if key not in self._champ:
            d, tft, tfb, _ = self.T[term]
            if len(d) <= r:
                self._champ[key] = d
            else:
                tf = tft + tfb
                o = np.lexsort((d, -tf))[:r]
                self._champ[key] = np.sort(d[o])
        return self._champ[key]

    def scores(self, qterms, cfg, prune=None):
        """-> (score array over all docs, n docs scored). prune = None | ('champ', r) | ('elim', theta)"""
        qtf = Counter(t for t in qterms if t in self.T)
        if not qtf:
            return np.zeros(self.N), 0
        if prune and prune[0] == "elim":
            mx = max(self.T[t][3] for t in qtf)
            qtf = Counter({t: c for t, c in qtf.items() if self.T[t][3] >= prune[1] * mx}) or qtf
        cand = None
        if prune and prune[0] == "champ":
            cand = np.unique(np.concatenate([self.champions(t, prune[1]) for t in qtf]))
        A, B = np.zeros(self.N), np.zeros(self.N)
        touched = np.zeros(self.N, bool)
        for t, c in qtf.items():
            d, tft, tfb, idf = self.T[t]
            qw = (1.0 + math.log(c)) if cfg.sub else float(c)
            k = qw * (idf if cfg.idf else 1.0)
            if cand is not None:
                pos = np.searchsorted(d, cand)
                pos[pos >= len(d)] = 0
                ok = d[pos] == cand
                dd, a_t, a_b = cand[ok], tft[pos[ok]], tfb[pos[ok]]
            else:
                dd, a_t, a_b = d, tft, tfb
            if cfg.zones:
                A[dd] += k * self._w(a_t, cfg.sub)
                B[dd] += k * self._w(a_b, cfg.sub)
            else:
                A[dd] += k * self._w(a_t + a_b, cfg.sub)
            touched[dd] = True
        if cfg.norm:
            na, nb = self.norms(cfg)
            A, B = A / na, B / nb
        s = cfg.gt * A + (1 - cfg.gt) * B if cfg.zones else A
        return s, int(touched.sum()) if cand is None else len(cand)

    def rank(self, text, cfg, k=10, prune=None, return_all=False):
        s, n = self.scores(self.ix.analyzer.query_terms(text), cfg, prune)
        nz = np.nonzero(s)[0]
        order = nz[np.lexsort((nz, -s[nz]))]
        out = order if return_all else order[:k]
        return [int(x) for x in out], n


# ----------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    usage = "usage: lexis.py build CORPUS OUT [stem|raw] | bool INDEX 'query' | rank INDEX 'free text' | spell INDEX word"
    if len(sys.argv) < 4:
        sys.exit(usage)
    cmd = sys.argv[1]
    if cmd == "build":
        stem = len(sys.argv) > 4 and sys.argv[4] == "stem"
        print(build_sharded(sys.argv[2], sys.argv[3], 4, "spimi", 2 << 20, stem, stem))
    elif cmd == "bool":
        e = Lexis(sys.argv[2])
        r, s = e.search(sys.argv[3])
        print(len(r), [e.names[d] for d in r[:20]], s)
    elif cmd == "rank":
        ix = Index(sys.argv[2])
        rk = Ranker(ix)
        for d in rk.rank(sys.argv[3], Config())[0]:
            print(ix.names[d])
    elif cmd == "spell":
        print(Lexis(sys.argv[2]).suggest(sys.argv[3]))
    else:
        sys.exit(usage)
