"""Runs every experiment of the Lexis report. Usage: python3 run_all.py [stage ...]  (default: all stages)
Stages are cached in results.json so a stage can be re-run alone."""
import csv, fnmatch, json, math, os, random, re, shutil, sys, time, tracemalloc
from collections import Counter
import numpy as np
from scipy import stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lexis
from lexis import Analyzer, Config, Index, Lexis, Ranker, TOKEN_RE, build_index, build_sharded

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE_DIR, "lexis_dataset", "corpus")
UP = os.path.join(BASE_DIR, "lexis_dataset") + os.sep
IDX = os.path.join(BASE_DIR, "idx")
OUT = os.path.join(BASE_DIR, "report")
os.makedirs(OUT, exist_ok=True)
RES = os.path.join(OUT, "results.json")
R = json.load(open(RES)) if os.path.exists(RES) else {}
VARIANTS = {"raw": (False, False), "stem_stop": (True, True), "stem": (True, False), "stop": (False, True)}


def save():
    json.dump(R, open(RES, "w"), indent=1)


def tsv(name):
    return list(csv.DictReader(open(UP + name, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))


# ------------------------------------------------------------------ A. build the 4 analysis variants (4 shards)
def stage_build():
    R["build"] = {}
    for name, (st, sp) in VARIANTS.items():
        d = os.path.join(IDX, name)
        shutil.rmtree(d, ignore_errors=True)
        R["build"][name] = build_sharded(CORPUS, d, 4, "spimi", 2 << 20, st, sp)
        print("built", name, R["build"][name]["seconds"])


# ------------------------------------------------------------------ B. exact retrieval
def brute_force(q, texts):
    """Evaluate query syntax by scanning lowercased corpus tokens, without using the index."""
    token_re = re.compile(r'"([^"]*)"|(\()|(\))|/(\d+)|([^\s()"]+)')
    tokens = []
    for match in token_re.finditer(q):
        if match.group(1) is not None:
            tokens.append(("PHRASE", match.group(1).lower()))
        elif match.group(2):
            tokens.append(("(", None))
        elif match.group(3):
            tokens.append((")", None))
        elif match.group(4):
            tokens.append(("PROX", int(match.group(4))))
        else:
            word = match.group(5)
            tokens.append((word if word in ("AND", "OR", "NOT") else "WORD", word.lower()))

    pos = 0

    def peek():
        return tokens[pos][0] if pos < len(tokens) else None

    def parse_or():
        nonlocal pos
        node = parse_and()
        while peek() == "OR":
            pos += 1
            node = ("or", node, parse_and())
        return node

    def parse_and():
        nonlocal pos
        node = parse_not()
        while peek() in ("AND", "NOT", "WORD", "PHRASE", "("):
            if peek() == "AND":
                pos += 1
            node = ("and", node, parse_not())
        return node

    def parse_not():
        nonlocal pos
        if peek() == "NOT":
            pos += 1
            return ("not", parse_not())
        return parse_proximity()

    def parse_proximity():
        nonlocal pos
        node = parse_atom()
        if peek() == "PROX":
            distance = tokens[pos][1]
            pos += 1
            node = ("prox", node, parse_atom(), distance)
        return node

    def parse_atom():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Expected a query term")
        kind, value = tokens[pos]
        pos += 1
        if kind == "(":
            node = parse_or()
            if peek() != ")":
                raise ValueError("Unmatched opening parenthesis")
            pos += 1
            return node
        if kind == "PHRASE":
            return ("phrase", re.findall(r"[a-z0-9]+", value))
        if kind != "WORD":
            raise ValueError("Expected a query term")
        if "*" in value:
            return ("wild", value)
        return ("term", value)

    tree = parse_or()
    if pos != len(tokens):
        raise ValueError("Unexpected query token: %s" % tokens[pos][0])
    if not texts:
        return set()
    universe = set(texts)

    def matches(node, words):
        kind = node[0]
        if kind == "term":
            terms = re.findall(r"[a-z0-9]+", node[1])
            return bool(terms) and terms[0] in words
        if kind == "wild":
            return any(fnmatch.fnmatchcase(word, node[1]) for word in words)
        if kind == "phrase":
            phrase = node[1]
            return bool(phrase) and any(
                words[i:i + len(phrase)] == phrase
                for i in range(len(words) - len(phrase) + 1)
            )
        if kind == "prox":
            left, right = node[1], node[2]
            if left[0] != "term" or right[0] != "term":
                raise ValueError("Proximity operands must be terms")
            left_terms = re.findall(r"[a-z0-9]+", left[1])
            right_terms = re.findall(r"[a-z0-9]+", right[1])
            if not left_terms or not right_terms:
                return False
            a, b, distance = left_terms[0], right_terms[0], node[3]
            return any(
                wa == a and wb == b and i != j and abs(i - j) <= distance
                for i, wa in enumerate(words)
                for j, wb in enumerate(words)
            )
        if kind == "not":
            return not matches(node[1], words)
        if kind == "and":
            return matches(node[1], words) and matches(node[2], words)
        if kind == "or":
            return matches(node[1], words) or matches(node[2], words)
        raise ValueError("Unknown query node: %s" % kind)

    return {name for name, words in texts.items() if matches(tree, words) and name in universe}


def stage_exact():
    e = Lexis(os.path.join(IDX, "raw"))
    rows = tsv("boolean_tests.tsv")
    texts = {
        fn[:-4]: re.findall(r"[a-z0-9]+", lexis._read_doc(CORPUS, fn).lower())
        for fn in lexis.corpus_files(CORPUS)
    }
    res, ok, brute_ok = [], 0, 0
    for r in rows:
        got_ids, _ = e.search(r["query"])
        got = {e.names[d] for d in got_ids}
        scanned = brute_force(r["query"], texts)
        exp = set(r["expected_doc_ids"].split(","))
        matches_brute = got == scanned
        res.append(dict(id=r["query_id"], q=r["query"], type=r["type"], n=len(exp), got=len(got),
                        brute=len(scanned), ok=got == exp, matches_brute=matches_brute))
        ok += got == exp
        brute_ok += matches_brute
    R["exact"] = dict(rows=res, passed=ok, total=len(rows),
                      brute_force_passed=brute_ok, brute_force_total=len(rows),
                      brute_force_failures=[r for r in res if not r["matches_brute"]])
    # skip pointers
    ix = e.main
    pairs = [("said", "opec"), ("the", "tanker"), ("oil", "opec"), ("trade", "deficit"), ("bank", "yen"),
             ("wheat", "corn"), ("gold", "silver"), ("interest", "rates"), ("company", "merger"), ("price", "coffee")]
    sk = []
    for a, b in pairs:
        A, B = ix.read(a, False)[0], ix.read(b, False)[0]
        truth = sorted(set(A) & set(B))
        row = dict(a=a, b=b, dfa=len(A), dfb=len(B), common=len(truth))
        st = {}
        assert lexis.intersect_skip(A, B, 0, st) == truth
        row["none"] = st["cmp"]
        for sp in (4, 16, 64, None):
            st = {}
            assert lexis.intersect_skip(A, B, sp, st) == truth
            row["sqrt" if sp is None else "s%d" % sp] = st["cmp"]
        sk.append(row)
    R["skips"] = sk
    d, _, _ = ix.read("the", False)
    tab = ix.skip_table("the")
    S = int(math.sqrt(len(d)))
    R["skip_table_check"] = all(tab[j - 1][0] == d[j * S] for j in range(1, len(tab) + 1)) and len(tab) == (len(d) - 1) // S
    m = ix.meta
    R["positional"] = dict(positional=m["postings_bytes"], plain=m["plain_bytes"], dict=m["dict_bytes"],
                           dict_raw=m["dict_raw_bytes"], nterms=m["nterms"], npost=m["n_postings"],
                           nocc=m["n_occurrences"])
    # incremental updates on a copy of the index
    tmp = os.path.join(IDX, "upd")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(os.path.join(IDX, "raw"), tmp)
    u = Lexis(tmp)
    before = u.search("merger")[0]
    t0 = time.perf_counter()
    u.add_document("doc_new_1", "ZORBLAX MERGER APPROVED\n\nThe zorblax merger was approved on Monday.")
    u.add_document("doc_new_2", "Another zorblax story\n\nNothing about the other word here.")
    dt = time.perf_counter() - t0
    victim = u.names[u.search("oil")[0][0]]
    in_before = u.names.index(victim) in u.search("oil")[0]
    u.delete_document(victim)
    u2 = Lexis(tmp)  # reopen: delta + tombstones persisted
    R["updates"] = dict(
        added_found=[u2.names[d] for d in u2.search("zorblax")[0]],
        new_in_merger=("doc_new_1" in [u2.names[d] for d in u2.search("merger")[0]]),
        phrase_new=[u2.names[d] for d in u2.search('"zorblax merger"')[0]],
        deleted_gone=(u2.names.index(victim) not in u2.search("oil")[0]), deleted_was_present=in_before,
        merger_count_before=len(before), merger_count_after=len(u2.search("merger")[0]),
        add_seconds=dt, rebuild_seconds=R.get("build", {}).get("raw", {}).get("seconds"))
    # spelling
    sp = tsv("misspellings.tsv")
    ok1 = ok3 = 0
    byk = {1: [0, 0, 0], 2: [0, 0, 0]}
    fails = []
    for i, r in enumerate(sp):
        k = 1 if i < 50 else 2
        s = e.suggest(r["misspelled"], 3)
        acc = set(r["accepted_corrections"].split("|") if "|" in r["accepted_corrections"] else r["accepted_corrections"].split(","))
        h1 = bool(s) and s[0] in acc
        h3 = any(x in acc for x in s)
        byk[k][0] += 1
        byk[k][1] += h1
        byk[k][2] += h3
        if not h1:
            fails.append((r["misspelled"], r["accepted_corrections"], s))
    R["spell"] = dict(by_typos=byk, fails=fails[:15], oov_all=all(w["misspelled"] not in e.vocabulary() for w in sp))
    save()


# ------------------------------------------------------------------ C. construction
def stage_construct():
    out = []
    limits = [256 << 10, 512 << 10, 1 << 20, 2 << 20]
    for algo in ("spimi", "bsbi"):
        for lim in limits:
            d = os.path.join(IDX, "c_tmp")
            shutil.rmtree(d, ignore_errors=True)
            st, _ = build_index(CORPUS, d, algo, lim)
            shutil.rmtree(d, ignore_errors=True)
            tracemalloc.start()
            st2, _ = build_index(CORPUS, d, algo, lim)
            tracemalloc.stop()
            peak, peak_merge = st2["peak_blocks"], st2["peak_merge"]
            shutil.rmtree(d, ignore_errors=True)
            out.append(dict(algo=algo, limit=lim, blocks=st["blocks"], block_bytes=st["block_bytes"],
                            seconds=st["seconds"], seconds_blocks=st["seconds_blocks"], peak=peak, peak_merge=peak_merge,
                            index_bytes=st["postings_bytes"] + st["dict_bytes"]))
            print(out[-1])
    R["construct"] = out
    sh = []
    for algo in ("spimi", "bsbi"):
        for n in (1, 2, 4):
            d = os.path.join(IDX, "sh_tmp")
            shutil.rmtree(d, ignore_errors=True)
            st = build_sharded(CORPUS, d, n, algo, 2 << 20)
            shutil.rmtree(d, ignore_errors=True)
            sh.append(dict(algo=algo, shards=n, seconds=st["seconds"], seconds_shards=st["seconds_shards"],
                           blocks=st["blocks"]))
            print(sh[-1])
    R["shards"] = sh
    R["nproc"] = os.cpu_count()
    save()


# ------------------------------------------------------------------ D. compression
def gamma_bits(g):
    return 2 * (g.bit_length() - 1) + 1


def gamma_encode(vals):
    s = ""
    for v in vals:
        b = bin(v)[2:]
        s += "0" * (len(b) - 1) + b
    return s


def gamma_decode(s):
    out, i = [], 0
    while i < len(s):
        z = 0
        while s[i] == "0":
            z += 1
            i += 1
        out.append(int(s[i:i + z + 1], 2))
        i += z + 1
    return out


def stage_compress():
    ix = Index(os.path.join(IDX, "raw"))
    rows = []
    for t, df, off, nb in ix.items():
        d = ix.read(t, False)[0]
        g = [d[0] + 1] + [d[i] - d[i - 1] for i in range(1, len(d))]   # first gap = docID+1 so every value >= 1
        v = sum(1 if x < 128 else 2 if x < 16384 else 3 if x < 2097152 else 4 for x in g)
        gm = (sum(gamma_bits(x) for x in g) + 7) // 8
        fw = (max(1, max(g).bit_length()) * len(g) + 7) // 8
        rows.append((t, df, 4 * len(d), fw, v, gm))
    t0 = rows[0]
    sample = ix.read("opec", False)[0]
    gs = [sample[0] + 1] + [sample[i] - sample[i - 1] for i in range(1, len(sample))]
    assert gamma_decode(gamma_encode(gs)) == gs
    rows.sort(key=lambda r: (-r[1], r[0]))
    top = rows[:100]
    rare = sorted(rows, key=lambda r: (r[1], r[0]))[:10000]

    def agg(rs):
        a = np.array([[r[2], r[3], r[4], r[5]] for r in rs]).sum(0)
        return dict(unc=int(a[0]), gaps=int(a[1]), vbyte=int(a[2]), gamma=int(a[3]), n=len(rs),
                    postings=int(sum(r[1] for r in rs)))
    R["compress"] = dict(all=agg(rows), top100=agg(top), rare10000=agg(rare), rare_maxdf=rare[-1][1],
                         top_terms=[r[0] for r in top[:8]])
    save()


# ------------------------------------------------------------------ E. vocabulary: Zipf and Heaps
def stage_vocab():
    an = Analyzer(False, False)
    files = lexis.corpus_files(CORPUS)
    cf, seen, T, pts = Counter(), set(), 0, []
    for i, f in enumerate(files):
        toks = TOKEN_RE.findall(lexis._read_doc(CORPUS, f).lower())
        cf.update(toks)
        seen.update(toks)
        T += len(toks)
        pts.append((T, len(seen)))
    P = np.array(pts, float)
    m = P[:, 0] >= 1000
    b, lk = np.polyfit(np.log10(P[m, 0]), np.log10(P[m, 1]), 1)
    pred = lk + b * np.log10(P[m, 0])
    r2h = 1 - ((np.log10(P[m, 1]) - pred) ** 2).sum() / ((np.log10(P[m, 1]) - np.log10(P[m, 1]).mean()) ** 2).sum()
    freqs = np.array(sorted(cf.values(), reverse=True), float)
    rk = np.arange(1, len(freqs) + 1)

    def zfit(hi):
        s, a = np.polyfit(np.log10(rk[:hi]), np.log10(freqs[:hi]), 1)
        pr = a + s * np.log10(rk[:hi])
        y = np.log10(freqs[:hi])
        return dict(s=-s, c=10 ** a, r2=1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    R["vocab"] = dict(tokens=int(T), types=len(seen), heaps=dict(k=10 ** lk, b=b, r2=r2h),
                      zipf_all=zfit(len(freqs)), zipf_1000=zfit(1000), cf_top=[(w, c) for w, c in cf.most_common(5)])
    plt.figure(figsize=(5, 3.6))
    plt.loglog(rk, freqs, ".", ms=2, label="Reuters-21578")
    z = R["vocab"]["zipf_1000"]
    plt.loglog(rk[:20000], z["c"] * rk[:20000] ** -z["s"], "r-", lw=1, label="fit ranks 1-1000: s=%.2f" % z["s"])
    plt.loglog(rk, freqs[0] / rk, "g--", lw=1, label="ideal Zipf cf=cf1/i")
    plt.xlabel("rank"); plt.ylabel("collection frequency"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "zipf.png"), dpi=130); plt.close()
    plt.figure(figsize=(5, 3.6))
    plt.loglog(P[:, 0], P[:, 1], "b-", label="Reuters-21578")
    plt.loglog(P[:, 0], 10 ** lk * P[:, 0] ** b, "r--", label="fit k=%.1f b=%.2f" % (10 ** lk, b))
    plt.loglog(P[:, 0], 44 * P[:, 0] ** 0.49, "g:", label="RCV1 k=44 b=0.49")
    plt.xlabel("tokens T"); plt.ylabel("vocabulary M"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "heaps.png"), dpi=130); plt.close()
    save()


# ------------------------------------------------------------------ F/G/H. ranking
def load_q(name):
    return [(r["query_id"], r["query"]) for r in tsv(name)]


def load_qrels(names_ix):
    q = {}
    for r in tsv("qrels.tsv"):
        q.setdefault(r["query_id"], {})[names_ix[r["doc_id"]]] = int(r["grade"])
    return q


def ap(rank, rel):
    nrel = sum(1 for g in rel.values() if g > 0)
    if not nrel:
        return 0.0
    h = s = 0
    for i, d in enumerate(rank, 1):
        if rel.get(d, 0) > 0:
            h += 1
            s += h / i
    return s / nrel


def p_at(rank, rel, k=10):
    return sum(1 for d in rank[:k] if rel.get(d, 0) > 0) / k


def ndcg(rank, rel, k=10):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(rank[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def interp11(rank, rel):
    nrel = sum(1 for g in rel.values() if g > 0)
    prec, rec, h = [], [], 0
    for i, d in enumerate(rank, 1):
        if rel.get(d, 0) > 0:
            h += 1
        prec.append(h / i)
        rec.append(h / nrel)
    out = []
    for lv in np.linspace(0, 1, 11):
        c = [p for p, r in zip(prec, rec) if r >= lv - 1e-9]
        out.append(max(c) if c else 0.0)
    return out


def evaluate(system, queries, qrels):
    """system(text) -> ranked docIDs. returns per-query metric lists"""
    m = dict(ap=[], p10=[], ndcg=[], pr=[])
    for qid, text in queries:
        rel = qrels.get(qid, {})
        r = system(text)
        m["ap"].append(ap(r, rel)); m["p10"].append(p_at(r, rel)); m["ndcg"].append(ndcg(r, rel))
        m["pr"].append(interp11(r, rel))
    return m


def summary(m):
    return dict(MAP=float(np.mean(m["ap"])), P10=float(np.mean(m["p10"])), NDCG10=float(np.mean(m["ndcg"])),
                PR=[float(x) for x in np.mean(m["pr"], 0)])


_RK = {}


def ranker(name):
    if name not in _RK:
        _RK[name] = Ranker(Index(os.path.join(IDX, name)))
    return _RK[name]


def boolean_system(rk, mode):
    def f(text):
        ts = [t for t in set(rk.ix.analyzer.query_terms(text))]
        arrs = [rk.T[t][0] if t in rk.T else np.array([], np.int32) for t in ts]
        if not arrs:
            return []
        out = arrs[0]
        for a in arrs[1:]:
            out = np.intersect1d(out, a) if mode == "and" else np.union1d(out, a)
        return [int(x) for x in out]
    return f


def stage_rank():
    t0 = time.perf_counter()
    names = Index(os.path.join(IDX, "raw")).names
    nix = {n: i for i, n in enumerate(names)}
    qrels = load_qrels(nix)
    train, test = load_q("queries_train.tsv"), load_q("queries_test.tsv")
    # ---- tune on TRAIN only
    grid = []
    for var in ("stem_stop", "stem"):
        rk = ranker(var)
        for sub in (True, False):
            for gt in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                cfg = Config(sub=sub, gt=gt)
                m = evaluate(lambda t: rk.rank(t, cfg, return_all=True)[0], train, qrels)
                grid.append(dict(index=var, sub=sub, gt=gt, MAP=float(np.mean(m["ap"])), P10=float(np.mean(m["p10"]))))
    grid.sort(key=lambda r: -r["MAP"])
    best = grid[0]
    bcfg = Config(sub=best["sub"], gt=best["gt"])
    R["tuning"] = dict(grid=grid[:8], n=len(grid), best=best)
    print("best", best, "tuning s", time.perf_counter() - t0)
    # ---- final systems, run ONCE on TEST (plus train numbers for reference)
    brk = ranker(best["index"])
    base_cfg = Config(sub=False, idf=False, norm=True, zones=False)
    systems = {
        "Boolean AND": (brk, boolean_system(brk, "and")),
        "Boolean OR (docID order)": (brk, boolean_system(brk, "or")),
        "Raw-tf cosine baseline": (brk, lambda t: brk.rank(t, base_cfg, return_all=True)[0]),
        "Best (zoned tf-idf)": (brk, lambda t: brk.rank(t, bcfg, return_all=True)[0]),
    }
    R["rank"] = {"train": {}, "test": {}}
    per_q = {}
    for split, qs in (("train", train), ("test", test)):
        for n, (_, f) in systems.items():
            m = evaluate(f, qs, qrels)
            R["rank"][split][n] = summary(m)
            if split == "test":
                per_q[n] = m
    R["best_cfg"] = dict(index=best["index"], **bcfg.__dict__)
    plt.figure(figsize=(5.2, 3.8))
    for n in systems:
        plt.plot(np.linspace(0, 1, 11), R["rank"]["test"][n]["PR"], marker="o", ms=3, label=n)
    plt.xlabel("recall"); plt.ylabel("interpolated precision"); plt.legend(fontsize=7); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "pr_curve.png"), dpi=130); plt.close()
    # significance of best vs baseline on test
    a, b = per_q["Best (zoned tf-idf)"]["ap"], per_q["Raw-tf cosine baseline"]["ap"]
    R["rank"]["p_best_vs_base"] = float(sps.ttest_rel(a, b).pvalue)
    # ---- ablations on TEST, one component at a time
    bestap = per_q["Best (zoned tf-idf)"]["ap"]
    abl = []
    nostem = ("stop" if brk.ix.meta["stop"] else "raw")
    nostop = ("stem" if brk.ix.meta["stem"] else "raw")
    variants = [("stemming off", ranker(nostem), bcfg), ("stop words off", ranker(nostop), bcfg),
                ("idf off", brk, bcfg.replace(idf=False)), ("length normalisation off", brk, bcfg.replace(norm=False)),
                ("zone weighting off", brk, bcfg.replace(zones=False))]
    if not brk.ix.meta["stop"]:   # tuning left stop words off, so also show the opposite switch
        variants.append(("stop words ON (not chosen by tuning)", ranker("stem_stop"), bcfg))
    zt = evaluate(lambda t: brk.rank(t, bcfg.replace(zones=False), return_all=True)[0], train, qrels)
    R["tuning"]["zones_off_train_MAP"] = float(np.mean(zt["ap"]))
    for n, rk, cfg in variants:
        m = evaluate(lambda t: rk.rank(t, cfg, return_all=True)[0], test, qrels)
        x = m["ap"]
        same = np.allclose(x, bestap)
        abl.append(dict(name=n, MAP=float(np.mean(x)), dMAP=float(np.mean(x) - np.mean(bestap)),
                        p_t=1.0 if same else float(sps.ttest_rel(x, bestap).pvalue),
                        p_w=1.0 if same else float(sps.wilcoxon(x, bestap).pvalue),
                        P10=float(np.mean(m["p10"])), NDCG10=float(np.mean(m["ndcg"]))))
    R["ablation"] = dict(rows=abl, best_MAP=float(np.mean(bestap)))
    # ---- efficiency (all 80 queries, best config)
    allq = [t for _, t in train + test]
    exh = [brk.rank(t, bcfg, 10) for t in allq]
    def lat(prune):
        best_t = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            for t in allq:
                brk.rank(t, bcfg, 10, prune)
            best_t = min(best_t, time.perf_counter() - t0)
        return 1000 * best_t / len(allq)
    eff = [dict(name="exhaustive (all postings of query terms)", overlap=1.0, scored=float(np.mean([e[1] for e in exh])),
                ms=lat(None))]
    for name, pr in [("champion lists r=10", ("champ", 10)), ("champion lists r=25", ("champ", 25)),
                     ("champion lists r=50", ("champ", 50)), ("champion lists r=100", ("champ", 100)),
                     ("index elimination idf>=0.3*max", ("elim", 0.3)), ("index elimination idf>=0.5*max", ("elim", 0.5)),
                     ("index elimination idf>=0.7*max", ("elim", 0.7))]:
        res = [brk.rank(t, bcfg, 10, pr) for t in allq]
        ov = float(np.mean([len(set(a[0]) & set(b[0])) / 10 for a, b in zip(res, exh)]))
        eff.append(dict(name=name, overlap=ov, scored=float(np.mean([r[1] for r in res])), ms=lat(pr)))
    R["efficiency"] = dict(rows=eff, ndocs=brk.N)
    R["rank_seconds"] = time.perf_counter() - t0
    save()


STAGES = dict(build=stage_build, exact=stage_exact, construct=stage_construct, compress=stage_compress,
              vocab=stage_vocab, rank=stage_rank)
if __name__ == "__main__":
    for s in (sys.argv[1:] or STAGES):
        t = time.perf_counter()
        STAGES[s]()
        save()
        print("stage", s, "done in %.1fs" % (time.perf_counter() - t))
