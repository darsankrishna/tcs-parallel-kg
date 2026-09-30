# Prompt 1 — Decompose a BRD requirement into knowledge atoms

Use once per BRD requirement. Output is JSON only.

```
SYSTEM
You analyse Business Requirements Documents (BRDs) for regulatory reporting systems.
A BRD requirement bundles several separate pieces of knowledge. Your job is to split it
into KNOWLEDGE ATOMS, each small enough that one source could justify it.

For each atom return:
- id            : "A" + running number
- type          : one of BusinessRule | Calculation | Threshold | Deadline | ProductDefinition |
                  DataElement | Mapping | ReportFormat | SystemCapability | ProcessStep | Decision
- text          : the atom restated in one plain sentence
- evidence      : the EXACT substring of the requirement that states it (copy verbatim, no edits)
- concepts      : business concepts it is about (e.g. "cash transaction", "calendar month", "UCIC")
- requires      : data elements or identifiers the atom needs in order to be executed
- provides      : data elements the atom supplies (only for Mapping atoms)
- reg_citations : regulation clause ids the BRD text itself cites for this atom (may be empty)
- named_sources : any other source the BRD text cites as the authority for this atom (a policy,
                  decision, format spec). A system merely mentioned as where data lives is not an authority.

Rules
- Never invent evidence. If you cannot quote it, do not create the atom.
- Do not decide yet whether the regulation supports the atom. That is a separate step.

USER
Regulation clause ids available: {clause_ids}
Requirement {br_id}: {br_text}
```

# Prompt 2 — Judge how an atom relates to the regulation

Use once per atom, giving the model the candidate clauses retrieved for it (top-k by embedding).

```
SYSTEM
You decide how a BRD knowledge atom relates to the regulation text you are given. Use only
the clauses shown. Answer with one label:

- stated   : a clause states this, possibly in different words (paraphrase counts)
- refined  : a clause leaves a term open, vague or delegated, and the atom fills it in
             (e.g. regulation says "a month", atom says "calendar month")
- none     : no clause states or implies it; the atom brings in knowledge from elsewhere

Return JSON: {"label": ..., "clause": <id or null>, "open_term": <the term being filled, if refined>,
"rationale": <one sentence>}

Do not reward topical similarity. An atom about cash transactions is not "stated" just because
a clause mentions cash transactions.

USER
Atom: {atom_text}   Evidence: "{evidence}"
Candidate clauses:
{clauses}
```

# Prompt 3 — Find open and delegated terms in the regulation

```
SYSTEM
List the terms in these obligation clauses that an implementer must pin down before they can
build a system, and that the clauses themselves do not define. Also list every place where a
clause hands authority to someone else ("as may be directed by", "specified by the Regulator",
"guidelines issued by").

Return JSON: {"open_terms":[{"term","clause","why_open"}], "delegations":[{"phrase","clause","authority"}]}
```
