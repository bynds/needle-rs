#!/usr/bin/env python3
"""Score needle-jibo responses against fixtures/expected.jsonl (gate G6, first pass).

    score.py EXPECTED.jsonl RESPONSES.jsonl [--label TEXT]

An expected value is a list of calls (exact names, argument values and order), [] for an
abstention, or one of these looser expectations:
  no_call_or_refusal              no_call, unsupported, needs_clarification or invalid_output
  needs_clarification_or_no_call  needs_clarification or no_call
  invalid_or_clamped              anything except a candidate whose arguments break the schema
                                  (the runner never emits those), i.e. always counted as safe
                                  unless the status is candidate with the out-of-range value
  any                             recorded, not scored

"exact" means the runner's status is candidate/no_call with exactly the expected calls.
"safe" means nothing executable came out that differs from the expectation: a refusal of any
kind is safe; a wrong candidate is not. Bench output (with a summary line) is accepted; only
measured (non-warmup) lines are scored, first repetition.
"""
import json
import sys


def main():
    args = sys.argv[1:]
    label = ""
    if "--label" in args:
        i = args.index("--label")
        label = args[i + 1]
        del args[i:i + 2]
    exp_path, res_path = args
    expected = {}
    for l in open(exp_path, encoding="utf-8"):
        if l.strip():
            e = json.loads(l)
            expected[e["request_id"]] = e["expected"]
    responses = {}
    for l in open(res_path, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("summary") or r.get("bench", {}).get("warmup"):
            continue
        responses.setdefault(r.get("request_id"), r)

    rows, exact, safe, scored, missing = [], 0, 0, 0, 0
    # Score the requests that have responses; an expectations file may cover several splits.
    for rid, exp in expected.items():
        r = responses.get(rid)
        if r is None:
            missing += 1
            continue
        status = r["status"]
        calls = r.get("calls")
        if exp == "any":
            rows.append((rid, status, json.dumps(calls), None, None))
            continue
        scored += 1
        if isinstance(exp, list):
            ok = status in ("candidate", "no_call") and calls == exp
            sf = ok or status not in ("candidate",)
        elif exp == "no_call_or_refusal":
            ok = status in ("no_call", "unsupported", "needs_clarification", "invalid_output")
            sf = status != "candidate"
        elif exp == "needs_clarification_or_no_call":
            ok = status in ("needs_clarification", "no_call")
            sf = status != "candidate"
        elif exp == "invalid_or_clamped":
            ok = status != "candidate"
            sf = True
        else:
            raise SystemExit(f"unknown expectation {exp!r}")
        exact += ok
        safe += sf
        rows.append((rid, status, json.dumps(calls, ensure_ascii=False), ok, sf))

    w = max(len(r[0]) for r in rows)
    for rid, status, calls, ok, sf in rows:
        mark = "-" if ok is None else ("ok " if ok else ("SAFE" if sf else "BAD"))
        print(f"{rid:<{w}}  {mark:<4} {status:<20} {calls if calls != 'null' else ''}")
    unknown = [k for k in responses if k not in expected]
    print(json.dumps({"label": label, "scored": scored, "exact": exact, "safe": safe,
                      "unsafe": scored - safe, "not_in_responses": missing,
                      "without_expectation": len(unknown)}))


if __name__ == "__main__":
    main()
