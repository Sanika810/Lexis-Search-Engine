# Lexis Search Engine

An information retrieval assignment implementation for Reuters-21578 with Boolean, phrase, proximity, wildcard, spelling, and ranked search.

## Requirements

- Python 3.13 (tested with Python 3.13.5)
- The assignment dataset, placed locally as:

```text
lexis_dataset\
  corpus\
    doc_00001.txt
    ...
  boolean_tests.tsv
  misspellings.tsv
  qrels.tsv
  queries_train.tsv
  queries_test.tsv
```

The dataset is not included in this repository. `run_all.py` expects this directory layout next to the scripts.

## Setup (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

## Build and evaluate

Run stages sequentially from the repository root:

```powershell
.\.venv\Scripts\python.exe .\run_all.py build
.\.venv\Scripts\python.exe .\run_all.py exact
.\.venv\Scripts\python.exe .\run_all.py construct
.\.venv\Scripts\python.exe .\run_all.py compress
.\.venv\Scripts\python.exe .\run_all.py vocab
.\.venv\Scripts\python.exe .\run_all.py rank
```

Stages share the `idx` and `report` directories; do not run them concurrently. The exact stage compares indexed search with both the provided expected results and a corpus-wide brute-force scan. Outputs are written to `report\results.json`; plots and the written report are in `report\`.

## Try the search CLI

Build a four-shard raw index first, or run the `build` stage above:

```powershell
.\.venv\Scripts\python.exe .\lexis.py bool .\idx\raw 'oil AND opec'
.\.venv\Scripts\python.exe .\lexis.py bool .\idx\raw '"crude oil"'
.\.venv\Scripts\python.exe .\lexis.py bool .\idx\raw 'economy /5 growth'
.\.venv\Scripts\python.exe .\lexis.py bool .\idx\raw 'econom*'
.\.venv\Scripts\python.exe .\lexis.py rank .\idx\stem 'wheat and corn price support'
.\.venv\Scripts\python.exe .\lexis.py spell .\idx\raw 'prokkects'
```

## Report

See [report/REPORT.md](report/REPORT.md) for experiment results, evaluation tables, limitations, and plots.
