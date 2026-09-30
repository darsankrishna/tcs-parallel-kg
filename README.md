# BRD knowledge-gap prototype

Builds a knowledge graph from a regulation excerpt and a BRD, then finds the knowledge the BRD
needed that neither document supplies, and ranks where inside the bank it probably lives.

Example: PML (Maintenance of Records) Rules 2005 (real text) × an illustrative CTR BRD (fictional bank).

## Run

```
pip install networkx scikit-learn
python pipeline.py        # graph, gaps, routing, evaluation  -> out/
python build_viewer.py    # interactive explorer               -> explorer/ctr-gap-map.html
```

## Files

| Path | What it is |
|---|---|
| `data/regulation.txt` | Rule excerpts with clause ids in brackets |
| `data/brd.md` | Illustrative BRD (12 requirements) |
| `prompts/prompts.md` | The three LLM prompts: decompose into atoms, judge against clauses, find open terms and delegations |
| `data/extraction.json` | Output of those prompts. Written by Claude by hand in this run (no model API available); every quote is verified against the BRD by the pipeline |
| `data/gold_provenance.json` | Answer key: where each atom's knowledge really came from. Used only for scoring |
| `data/source_catalog.json` | Mock catalog of 16 places knowledge lives (3 real regulator sources, 13 fictional internal ones) |
| `pipeline.py` | Parse → verify → regulation analysis → graph → gap queries → routing → evaluation |
| `out/graph.json` | The graph (networkx node-link format; load into Neo4j or Gephi) |
| `out/gaps.csv`, `out/gaps.json` | Every gap with kind, severity and top justify / fulfil sources |
| `out/summary.json` | Counts and evaluation |

## What is deterministic and what needs an LLM

Deterministic: clause and requirement parsing, evidence verification, defined-term extraction,
delegation detection, every gap query, routing scores, evaluation.

LLM: splitting requirements into atoms, judging stated / refined / none, proposing open terms.
To run with a real model, implement those three prompts and write their JSON to `data/extraction.json`
in the same schema; nothing else changes.

## Trying it on a real pair

1. Put the regulation's clauses in `data/regulation.txt` with `[ID]` markers.
2. Put the BRD requirements in `data/brd.md` as lines starting `BR-nn Title.`
3. Run the prompts per requirement and per clause group.
4. Replace `source_catalog.json` with a real inventory (document titles and one-line descriptions).
5. For evaluation, have a business analyst label where each atom came from (`gold_provenance.json`).
