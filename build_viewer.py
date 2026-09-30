"""Builds explorer/ctr-gap-map.html from the pipeline outputs (run pipeline.py first)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "out"


def cite(cid):
    m = re.match(r"R(\d+)(?:\.(\d+))?(?:\.(\w+))?$", cid)
    rule, sub, cl = m.groups()
    return f"Rule {rule}" + (f"({sub})" if sub else "") + (f"({cl})" if cl else "")


def main():
    import pipeline as p
    clauses = p.parse_regulation(p.DATA / "regulation.txt")
    reqs = p.parse_brd(p.DATA / "brd.md")
    ext = json.loads((p.DATA / "extraction.json").read_text())
    gold = json.loads((p.DATA / "gold_provenance.json").read_text())["atoms"]
    catalog = json.loads((p.DATA / "source_catalog.json").read_text())["sources"]
    gaps = json.loads((OUT / "gaps.json").read_text())
    summ = json.loads((OUT / "summary.json").read_text())
    by_target = {g["target"]: g for g in gaps}

    requirements = []
    for br, r in reqs.items():
        atoms = []
        for a in ext["requirements"][br]:
            g = by_target.get(a["id"])
            status = "stated" if a["judge"]["label"] == "stated" else {"INTERPRETATION": "interp", "POINTER": "pointer", "HIDDEN": "hidden"}[g["kind"]]
            atoms.append({"id": a["id"], "type": a["type"], "text": a["text"], "evidence": a["evidence"],
                          "judge": a["judge"]["label"], "clause": a["judge"].get("clause"),
                          "related": a["judge"].get("related_clause"), "open_term": a["judge"].get("open_term"),
                          "rationale": a["judge"]["rationale"], "named": a["named_sources"], "cites": a["reg_citations"],
                          "requires": a["requires"], "provides": a["provides"], "status": status,
                          "severity": g["severity"] if g else None, "overreach": g.get("citation_overreach") if g else [],
                          "justify": [c["source"] for c in g["justify"]] if g else [],
                          "fulfil": [c["source"] for c in g["fulfil"]] if g else [],
                          "authority": g.get("authority") if g else None,
                          "gold": gold[a["id"]]["origin"], "gold_sources": gold[a["id"]]["gold_sources"],
                          "gold_note": gold[a["id"]].get("note")})
        requirements.append({"id": br, "title": r["title"], "text": r["text"], "atoms": atoms})

    filled = {}
    for a in (x for alist in ext["requirements"].values() for x in alist):
        for c in a["concepts"]:
            filled.setdefault(c, []).append(a["id"])
    data = {
        "clauses": {c: {"cite": cite(c), "text": t, "obligation": c in summ["obligations"]} for c, t in clauses.items()},
        "requirements": requirements,
        "sources": {s["id"]: s for s in catalog},
        "open_terms": [{**t, "filled_by": filled.get(t["concept"], [])} for t in ext["regulation_terms"]
                       if t["term"] not in [d["term"] for d in summ["dropped_terms"]]],
        "dropped": summ["dropped_terms"],
        "delegations": summ["delegations"],
        "defined": summ["defined_terms"],
        "structural": [g for g in gaps if g["kind"] in ("DATA_UNMAPPED", "OPEN_TERM_UNFILLED", "OBLIGATION_UNCOVERED")],
        "gap_kinds": summ["gap_kinds"],
        "graph": summ["graph"],
        "eval": summ["evaluation"],
    }
    tpl = (ROOT / "explorer_template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    (ROOT / "explorer").mkdir(exist_ok=True)
    (ROOT / "explorer" / "ctr-gap-map.html").write_text(html, encoding="utf-8")
    print("wrote explorer/ctr-gap-map.html", len(html), "bytes")


if __name__ == "__main__":
    main()
