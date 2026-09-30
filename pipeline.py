"""
BRD knowledge-gap prototype.

Given a regulation excerpt and a BRD derived from it, build a knowledge graph with three layers
(normative / specification / provenance), find the knowledge the BRD needed that neither document
supplies, and rank the internal sources most likely to hold it.

    python pipeline.py            # uses data/extraction.json (LLM stage output)

Stages
  1. Parse regulation into clauses, BRD into requirements (deterministic).
  2. LLM extraction (prompts/prompts.md): atoms, judge labels, candidate open terms.
     Loaded from data/extraction.json; every evidence span is verified against the BRD.
  3. Regulation analysis (deterministic): defined terms, obligations, delegations.
  4. Graph build (networkx).
  5. Gap detection = graph queries.
  6. Gap -> source routing (type prior + TF-IDF + delegation authority + named-source match).
  7. Evaluation against the answer key, including a lexical baseline.
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "out"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- 1. parsing

def parse_regulation(path):
    text = path.read_text(encoding="utf-8")
    clauses = {}
    for m in re.finditer(r"^\[(R[\w.]+)\]\s*(.+?)(?=^\[R|\Z)", text, re.M | re.S):
        clauses[m.group(1)] = " ".join(m.group(2).split())
    return clauses


def parse_brd(path):
    reqs = {}
    for m in re.finditer(r"^(BR-\d+)\s+([^.]+)\.\s*(.+?)(?=^BR-\d+|\Z)", path.read_text(encoding="utf-8"), re.M | re.S):
        reqs[m.group(1)] = {"title": m.group(2).strip(), "text": " ".join(m.group(3).split())}
    return reqs

# ----------------------------------------------------------------------------- 3. regulation analysis

DELEGATION_PATTERNS = [
    (r"specified by the Regulator", "Regulator"),
    (r"guidelines issued by the Director", "Director"),
    (r"as may be directed by the Director", "Director"),
    (r"as may be specified by (?:its|the) regulator", "Regulator"),
    (r"referred to the Central Government", "Central Government"),
]
# Who the authorities are is itself knowledge that is NOT in the excerpt (Rule 2 points to the Act).
AUTHORITY_ALIASES = {"Director": "FIU-IND", "Regulator": "RBI", "Central Government": "Government of India"}


def analyse_regulation(clauses, proposed_terms):
    defined = {}
    for cid, txt in clauses.items():
        m = re.match(r'"([^"]+)"\s+means', txt)
        if m:
            defined[m.group(1).lower()] = cid
    obligations = [c for c, t in clauses.items() if " shall " in f" {t} " and not c.startswith("R2")]
    delegations = []
    for cid, txt in clauses.items():
        for pat, who in DELEGATION_PATTERNS:
            for m in re.finditer(pat, txt, re.I):
                delegations.append({"clause": cid, "phrase": m.group(0), "authority": who,
                                    "resolved_to": AUTHORITY_ALIASES[who]})
    open_terms, dropped = [], []
    for t in proposed_terms:
        norm = re.sub(r"^(the|its)\s+", "", t["term"].lower())
        if norm in defined:
            dropped.append({**t, "reason": f"defined in {defined[norm]}"})
        else:
            open_terms.append(t)
    return defined, obligations, delegations, open_terms, dropped

# ----------------------------------------------------------------------------- 2. verify LLM output

def load_and_verify(extraction, reqs, clauses):
    atoms, problems = [], []
    for br, alist in extraction["requirements"].items():
        for a in alist:
            a = {**a, "br": br}
            if a["evidence"] not in reqs[br]["text"]:
                problems.append(f"{a['id']}: evidence not found verbatim in {br}")
            for c in a["reg_citations"] + [a["judge"].get("clause")]:
                if c and c not in clauses:
                    problems.append(f"{a['id']}: cites unknown clause {c}")
            atoms.append(a)
    return atoms, problems

# ----------------------------------------------------------------------------- 4. graph

def build_graph(clauses, reqs, atoms, defined, obligations, delegations, open_terms):
    G = nx.MultiDiGraph()
    for cid, txt in clauses.items():
        G.add_node(cid, kind="Provision", label=cid, text=txt, obligation=cid in obligations)
    for term, cid in defined.items():
        G.add_node(f"term:{term}", kind="DefinedTerm", label=term)
        G.add_edge(cid, f"term:{term}", rel="DEFINES")
    for t in open_terms:
        cn = f"concept:{t['concept']}"
        G.add_node(cn, kind="Concept", label=t["concept"], open=True, term=t["term"], why_open=t["why_open"])
        G.add_edge(t["clause"], cn, rel="LEAVES_OPEN")
    for i, d in enumerate(delegations):
        dn = f"deleg:{i}"
        G.add_node(dn, kind="Delegation", label=f"{d['authority']} ({d['resolved_to']})", **d)
        G.add_edge(d["clause"], dn, rel="DELEGATES_TO")
    for br, r in reqs.items():
        G.add_node(br, kind="Requirement", label=br, title=r["title"], text=r["text"])
    for a in atoms:
        aid = a["id"]
        G.add_node(aid, kind="Atom", label=aid, **{k: a[k] for k in ("type", "text", "evidence", "br")},
                   judge=a["judge"]["label"], named_sources=a["named_sources"])
        G.add_edge(a["br"], aid, rel="HAS_ATOM")
        for c in a["concepts"]:
            cn = f"concept:{c}"
            if cn not in G:
                G.add_node(cn, kind="Concept", label=c, open=False)
            G.add_edge(aid, cn, rel="ABOUT")
        j = a["judge"]
        if j["label"] == "stated":
            G.add_edge(aid, j["clause"], rel="SUPPORTED_BY")
        elif j["label"] == "refined":
            G.add_edge(aid, j["clause"], rel="REFINES", open_term=j["open_term"])
        elif j.get("related_clause"):
            G.add_edge(aid, j["related_clause"], rel="RELATED_TO")
        for c in a["reg_citations"]:
            G.add_edge(aid, c, rel="CITES_CLAUSE")
        for s in a["named_sources"]:
            sn = f"named:{s}"
            G.add_node(sn, kind="NamedSource", label=s, in_corpus=False)
            G.add_edge(aid, sn, rel="CITES_SOURCE")
        for d in a["requires"]:
            G.add_node(f"data:{d}", kind="DataElement", label=d)
            G.add_edge(aid, f"data:{d}", rel="REQUIRES")
        for d in a["provides"]:
            G.add_node(f"data:{d}", kind="DataElement", label=d)
            G.add_edge(aid, f"data:{d}", rel="PROVIDES")
    return G

# ----------------------------------------------------------------------------- 5. gap detection (graph queries)

SCOPE_TYPES = {"BusinessRule", "ProductDefinition", "Decision", "Calculation", "Threshold"}


def out_rel(G, n, rel):
    return [(v, d) for _, v, d in G.out_edges(n, data=True) if d["rel"] == rel]


def in_rel(G, n, rel):
    return [(u, d) for u, _, d in G.in_edges(n, data=True) if d["rel"] == rel]


def detect_gaps(G, delegations):
    gaps = []
    # G1-G3: atoms the regulation does not state
    for aid, nd in G.nodes(data=True):
        if nd["kind"] != "Atom" or nd["judge"] == "stated":
            continue
        named = out_rel(G, aid, "CITES_SOURCE")
        refines = out_rel(G, aid, "REFINES")
        if named:
            kind = "POINTER"          # BRD names a source, but it is not in the corpus
        elif refines:
            kind = "INTERPRETATION"   # fills an open/delegated term with no recorded authority
        else:
            kind = "HIDDEN"           # knowledge from nowhere in the corpus, and unacknowledged
        sev = "high" if nd["type"] in SCOPE_TYPES else ("low" if nd["type"] in {"Mapping", "ProcessStep"} else "medium")
        if kind == "POINTER" and sev == "high":
            sev = "medium"
        overreach = [c for c, _ in out_rel(G, aid, "CITES_CLAUSE")]
        gaps.append({"gap_id": f"G-{aid}", "kind": kind, "target": aid, "severity": sev,
                     "open_term": refines[0][1]["open_term"] if refines else None,
                     "refined_clause": refines[0][0] if refines else None,
                     "named_sources": [G.nodes[v]["label"] for v, _ in named],
                     "citation_overreach": overreach,
                     "summary": nd["text"]})
    # G4: open terms nobody fills
    for n, nd in G.nodes(data=True):
        if nd["kind"] == "Concept" and nd.get("open") and not in_rel(G, n, "ABOUT"):
            clause = in_rel(G, n, "LEAVES_OPEN")[0][0]
            gaps.append({"gap_id": f"G-open-{nd['label'].replace(' ', '_')}", "kind": "OPEN_TERM_UNFILLED",
                         "target": n, "severity": "high", "clause": clause,
                         "summary": f"'{nd['term']}' ({clause}) is not defined and the BRD never pins it down: {nd['why_open']}"})
    # G5: data the BRD needs but never maps to a source
    for n, nd in G.nodes(data=True):
        if nd["kind"] != "DataElement":
            continue
        if in_rel(G, n, "REQUIRES") and not in_rel(G, n, "PROVIDES"):
            needers = sorted(u for u, _ in in_rel(G, n, "REQUIRES"))
            gaps.append({"gap_id": f"G-data-{nd['label'].replace(' ', '_').replace('/', '-')}", "kind": "DATA_UNMAPPED",
                         "target": n, "severity": "medium", "needed_by": needers,
                         "summary": f"'{nd['label']}' is needed by {', '.join(needers)} but no mapping provides it."})
    # G6: obligations with no requirement behind them
    for n, nd in G.nodes(data=True):
        if nd["kind"] == "Provision" and nd["obligation"]:
            if not (in_rel(G, n, "SUPPORTED_BY") or in_rel(G, n, "REFINES")):
                is_meta = any(d["clause"] == n for d in delegations)
                gaps.append({"gap_id": f"G-uncovered-{n}", "kind": "OBLIGATION_UNCOVERED", "target": n,
                             "severity": "low" if is_meta else "medium",
                             "summary": ("Meta-provision: tells you who holds interpretation authority. "
                                         if is_meta else "No BRD requirement implements this obligation: ") + nd["text"][:160]})
    for g in gaps:
        G.add_node(g["gap_id"], kind="Gap", label=g["kind"], gap_kind=g["kind"],
                   **{k: v for k, v in g.items() if k not in ("gap_id", "kind")})
        G.add_edge(g["gap_id"], g["target"], rel="ON")
    return gaps

# ----------------------------------------------------------------------------- 6. routing

TYPE_PRIOR = {
    "BusinessRule":      {"REG_GUIDANCE": 1.0, "INT_DECISION": 0.9, "INT_POLICY": 0.6, "INT_SYSTEM": 0.4, "TACIT": 0.4, "REG_FORMAT": 0.3},
    "Calculation":       {"INT_POLICY": 0.9, "REG_GUIDANCE": 0.8, "INT_DECISION": 0.7, "TACIT": 0.4},
    "ProductDefinition": {"INT_SYSTEM": 0.9, "INT_DECISION": 0.8, "REG_GUIDANCE": 0.6, "TACIT": 0.4},
    "Decision":          {"INT_DECISION": 1.0, "TACIT": 0.5, "REG_GUIDANCE": 0.4},
    "DataElement":       {"REG_FORMAT": 1.0, "INT_SYSTEM": 0.8, "REG_GUIDANCE": 0.4},
    "Mapping":           {"INT_SYSTEM": 1.0, "TACIT": 0.4},
    "ReportFormat":      {"REG_FORMAT": 1.0, "REG_GUIDANCE": 0.5},
    "SystemCapability":  {"INT_SYSTEM": 0.9, "REG_FORMAT": 0.7},
    "ProcessStep":       {"INT_PROCESS": 1.0, "REG_FORMAT": 0.6, "INT_POLICY": 0.5},
    "Threshold":         {"REG_GUIDANCE": 0.8, "REG_FORMAT": 0.5},
    "Deadline":          {"REG_GUIDANCE": 0.8, "REG_FORMAT": 0.5},
    "_DATA_UNMAPPED":    {"INT_SYSTEM": 1.0, "REG_FORMAT": 0.3},
    "_OPEN_TERM_UNFILLED": {"REG_GUIDANCE": 1.0, "INT_DECISION": 0.8, "REG_FORMAT": 0.5, "TACIT": 0.4},
    "_OBLIGATION_UNCOVERED": {"INT_PROCESS": 0.8, "INT_POLICY": 0.8, "REG_GUIDANCE": 0.6},
}
W_PRIOR, W_SIM, W_AUTH, W_NAMED = 0.55, 0.45, 0.30, 0.50
JUSTIFY = {"REG_GUIDANCE", "REG_FORMAT", "INT_POLICY", "INT_DECISION", "TACIT"}
FULFIL = {"INT_SYSTEM", "INT_PROCESS", "TACIT"}


def tokens(s):
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in ENGLISH_STOP_WORDS and len(t) > 1}


def gap_query(g, G):
    t = G.nodes[g["target"]]
    if t["kind"] == "Atom":
        concepts = [G.nodes[v]["label"] for v, _ in out_rel(G, g["target"], "ABOUT")]
        return " ".join([t["text"], t["evidence"], " ".join(concepts), g.get("open_term") or ""])
    if g["kind"] == "DATA_UNMAPPED":
        return f"{t['label']} field table source system " + " ".join(G.nodes[a]["text"] for a in g["needed_by"])
    if g["kind"] == "OPEN_TERM_UNFILLED":
        return f"{t['term']} {t['why_open']}"
    return t["text"]


def route(gaps, G, catalog, delegations):
    docs = [f"{s['title']}. {s['description']}" for s in catalog]
    queries = [gap_query(g, G) for g in gaps]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2)).fit(docs + queries)
    sims = cosine_similarity(vec.transform(queries), vec.transform(docs))
    deleg_auth = {(d["clause"], d["phrase"].lower()): d["resolved_to"] for d in delegations}
    for gi, g in enumerate(gaps):
        t = G.nodes[g["target"]]
        prior = TYPE_PRIOR[t["type"]] if t["kind"] == "Atom" else TYPE_PRIOR["_" + g["kind"]]
        # if the atom fills a delegated term, the delegate's publications get a boost
        auth = None
        if g.get("open_term"):
            for (cl, phrase), who in deleg_auth.items():
                if cl == g.get("refined_clause") and (phrase in g["open_term"].lower() or g["open_term"].lower() in phrase
                                                      or ("regulator" in g["open_term"].lower() and "regulator" in phrase)
                                                      or ("director" in g["open_term"].lower() and "director" in phrase)):
                    auth = who
        mx = sims[gi].max() or 1.0
        scored = []
        for si, s in enumerate(catalog):
            named = max((len(tokens(n) & tokens(s["title"])) / max(1, len(tokens(n))) for n in g.get("named_sources", [])), default=0)
            score = (W_PRIOR * prior.get(s["category"], 0.0) + W_SIM * sims[gi, si] / mx
                     + (W_AUTH if auth and s["publisher"] == auth else 0) + (W_NAMED if named >= 0.5 else 0))
            scored.append((round(score, 3), s["id"], s["category"]))
        scored.sort(reverse=True)
        g["authority"] = auth
        g["candidates"] = [{"source": sid, "score": sc} for sc, sid, _ in scored[:3]]
        # Two questions hide in every gap: WHY is this required (justification) and
        # WHERE do we get it / HOW is it done (fulfilment). Rank each separately.
        needs_justify = not (t["kind"] == "DataElement" or t.get("type") == "Mapping")
        g["justify"] = [{"source": sid, "score": sc} for sc, sid, c in scored if c in JUSTIFY][:3] if needs_justify else []
        g["fulfil"] = [{"source": sid, "score": sc} for sc, sid, c in scored if c in FULFIL][:3]
        for rank, c in enumerate(g["candidates"], 1):
            G.add_edge(g["gap_id"], f"src:{c['source']}", rel="RESOLVABLE_BY", score=c["score"], rank=rank)
    for s in catalog:
        if f"src:{s['id']}" in G:
            G.nodes[f"src:{s['id']}"].update(kind="SourceCandidate", label=s["id"], title=s["title"], category=s["category"])

# ----------------------------------------------------------------------------- 7. evaluation

def stem(w):
    for suf in ("ies", "es", "s", "ed", "ing"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def lexical_novelty(evidence, reg_vocab):
    toks = [stem(t) for t in tokens(evidence)]
    return (sum(t not in reg_vocab for t in toks) / len(toks)) if toks else 0.0


def prf(pred, gold):
    tp = sum(p and g for p, g in zip(pred, gold)); fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(P, 2), "recall": round(R, 2),
            "f1": round(2 * P * R / (P + R), 2) if P + R else 0.0}


def evaluate(atoms, gaps, gold, clauses, catalog):
    reg_vocab = {stem(t) for t in tokens(" ".join(clauses.values()))}
    cat = {s["id"]: s["category"] for s in catalog}
    rows, gold_ext, pred_judge, pred_lex = [], [], [], []
    for a in atoms:
        g = gold["atoms"][a["id"]]
        nov = lexical_novelty(a["evidence"], reg_vocab)
        rows.append({"atom": a["id"], "gold_origin": g["origin"], "judge": a["judge"]["label"], "novelty": round(nov, 2)})
        gold_ext.append(g["origin"] != "REG")
        pred_judge.append(a["judge"]["label"] != "stated")
        pred_lex.append(nov >= 0.5)
    hits1 = hits3 = cat_ok = n = 0
    s_hits1 = s_hits3 = 0
    per_gap = []
    for gp in gaps:
        if gp["target"] not in gold["atoms"]:
            continue
        gs = gold["atoms"][gp["target"]]["gold_sources"]
        if not gs:
            continue
        n += 1
        top = [c["source"] for c in gp["candidates"]]
        hits1 += top[0] in gs; hits3 += bool(set(top) & set(gs))
        cat_ok += cat[top[0]] == gold["atoms"][gp["target"]]["origin"] or cat[top[0]] in {cat[s] for s in gs}
        # split view: look in the list that matches the kind of knowledge actually missing
        view = "justify" if gold["atoms"][gp["target"]]["origin"] in JUSTIFY | {"REG_GUIDANCE", "REG_FORMAT"} else "fulfil"
        vtop = [c["source"] for c in gp[view]] or top
        s_hits1 += vtop[0] in gs; s_hits3 += bool(set(vtop) & set(gs))
        per_gap.append({"gap": gp["gap_id"], "top3": top, "view": view, "view_top3": vtop, "gold": gs,
                        "hit@1": top[0] in gs, "view_hit@1": vtop[0] in gs})
    return {"atoms": rows,
            "gap_detection": {"lexical_baseline": prf(pred_lex, gold_ext), "graph_plus_judge": prf(pred_judge, gold_ext)},
            "routing": {"n": n, "hit@1": round(hits1 / n, 2), "hit@3": round(hits3 / n, 2),
                        "top1_category_match": round(cat_ok / n, 2),
                        "split_view_hit@1": round(s_hits1 / n, 2), "split_view_hit@3": round(s_hits3 / n, 2),
                        "per_gap": per_gap}}

# ----------------------------------------------------------------------------- main

def main():
    clauses = parse_regulation(DATA / "regulation.txt")
    reqs = parse_brd(DATA / "brd.md")
    extraction = json.loads((DATA / "extraction.json").read_text())
    gold = json.loads((DATA / "gold_provenance.json").read_text())
    catalog = json.loads((DATA / "source_catalog.json").read_text())["sources"]

    atoms, problems = load_and_verify(extraction, reqs, clauses)
    defined, obligations, delegations, open_terms, dropped = analyse_regulation(clauses, extraction["regulation_terms"])
    G = build_graph(clauses, reqs, atoms, defined, obligations, delegations, open_terms)
    gaps = detect_gaps(G, delegations)
    route(gaps, G, catalog, delegations)
    ev = evaluate(atoms, gaps, gold, clauses, catalog)

    profile = Counter(a["judge"]["label"] for a in atoms)
    gold_profile = Counter(gold["atoms"][a["id"]]["origin"] for a in atoms)
    gap_kinds = Counter(g["kind"] for g in gaps)

    summary = {"clauses": len(clauses), "requirements": len(reqs), "atoms": len(atoms),
               "evidence_problems": problems, "defined_terms": defined, "obligations": obligations,
               "delegations": delegations, "open_terms": [t["term"] for t in open_terms], "dropped_terms": dropped,
               "judge_profile": dict(profile), "gold_profile": dict(gold_profile), "gap_kinds": dict(gap_kinds),
               "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                         "node_kinds": dict(Counter(d["kind"] for _, d in G.nodes(data=True)))},
               "evaluation": ev}

    (OUT / "graph.json").write_text(json.dumps(nx.node_link_data(G, edges="links"), indent=1, default=str))
    (OUT / "gaps.json").write_text(json.dumps(gaps, indent=1))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(OUT / "gaps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gap_id", "kind", "severity", "target", "summary", "justify_sources", "fulfil_sources"])
        for g in gaps:
            w.writerow([g["gap_id"], g["kind"], g["severity"], g["target"], g["summary"],
                        " ".join(c["source"] for c in g["justify"]), " ".join(c["source"] for c in g["fulfil"])])

    print(f"clauses={len(clauses)} requirements={len(reqs)} atoms={len(atoms)} evidence_problems={len(problems)}")
    for p in problems:
        print("  !", p)
    print("defined terms:", defined)
    print("dropped proposed open terms:", [(d['term'], d['reason']) for d in dropped])
    print("delegations:", [(d['clause'], d['phrase'], d['resolved_to']) for d in delegations])
    print("judge profile:", dict(profile), "| gold profile:", dict(gold_profile))
    print("gap kinds:", dict(gap_kinds))
    print("graph:", summary["graph"])
    print("gap detection:", json.dumps(ev["gap_detection"]))
    print("routing:", {k: v for k, v in ev["routing"].items() if k != "per_gap"})
    for pg in ev["routing"]["per_gap"]:
        print("   ", pg)


if __name__ == "__main__":
    main()
