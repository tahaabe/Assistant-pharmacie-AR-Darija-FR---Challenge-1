"""Evaluate the NLU pipeline on eval/scenarios.jsonl.

    python scripts/evaluate.py --tag v1        ->  eval/results_v1.json + eval/errors_v1.jsonl

Metrics: language ID accuracy, intent accuracy + macro-F1, medication extraction P/R/F1
(brand level, micro), quantity / dosage / city accuracy (on the scenarios that specify them),
and latency. Broken down by language group.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.dialog import Assistant  # noqa: E402


def f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def macro_f1(gold, pred):
    labels = sorted(set(gold) | set(pred))
    out = {}
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[l] = {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1(pr, rc), 3), "support": tp + fn}
    gold_labels = [l for l in labels if out[l]["support"] > 0]
    return round(statistics.mean(out[l]["f1"] for l in gold_labels), 3), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "eval" / "scenarios.jsonl"))
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]
    bot = Assistant()

    groups = defaultdict(lambda: Counter())
    gold_i, pred_i, errors, lat = [], [], [], []
    llm_used = 0
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        o = bot.handle(r["text"], session_id=f"eval-{i}")     # fresh session per scenario
        lat.append((time.perf_counter() - t0) * 1000)
        llm_used += o.get("nlu_source") != "rules"
        g = groups[r["lang"]]
        g["n"] += 1
        # language
        lang_ok = o["language"]["code_mixed"] if r["lang"] == "mixed" else o["language"]["primary"] == r["lang"]
        g["lang_ok"] += lang_ok
        # intent
        pi = o["intent"]["name"]
        gold_i.append(r["intent"]); pred_i.append(pi)
        g["intent_ok"] += pi == r["intent"]
        # medications (brand level)
        pm = [m["matched"] for m in o["entities"]["medications"]]
        gm = r["medications"]
        tp = len(set(pm) & set(gm))
        g["med_tp"] += tp; g["med_fp"] += len(set(pm) - set(gm)); g["med_fn"] += len(set(gm) - set(pm))
        meds = o["entities"]["medications"]
        if r.get("quantity") is not None:
            g["qty_n"] += 1
            g["qty_ok"] += bool(meds) and meds[0]["quantity"] == r["quantity"]
        if r.get("dosage_mg") is not None:
            g["dose_n"] += 1
            g["dose_ok"] += bool(meds) and meds[0]["requested_dosage_mg"] == r["dosage_mg"]
        if r.get("city") is not None or o["entities"]["city"]:
            g["city_n"] += 1
            g["city_ok"] += o["entities"]["city"] == r.get("city")
        ok = lang_ok and pi == r["intent"] and set(pm) == set(gm)
        g["exact"] += ok
        if not ok:
            errors.append({"id": r["id"], "text": r["text"], "gold": {k: r[k] for k in ("lang", "intent", "medications")},
                           "pred": {"lang": o["language"]["primary"], "code_mixed": o["language"]["code_mixed"],
                                    "intent": pi, "medications": pm}, "evidence": o["intent"].get("evidence")})

    def summarize(c: Counter) -> dict:
        p = c["med_tp"] / (c["med_tp"] + c["med_fp"]) if c["med_tp"] + c["med_fp"] else 1.0
        rc = c["med_tp"] / (c["med_tp"] + c["med_fn"]) if c["med_tp"] + c["med_fn"] else 1.0
        rate = lambda k, n: round(c[k] / c[n], 3) if c[n] else None
        return {"n": c["n"], "language_acc": rate("lang_ok", "n"), "intent_acc": rate("intent_ok", "n"),
                "med_precision": round(p, 3), "med_recall": round(rc, 3), "med_f1": round(f1(p, rc), 3),
                "quantity_acc": rate("qty_ok", "qty_n"), "dosage_acc": rate("dose_ok", "dose_n"),
                "city_acc": rate("city_ok", "city_n"), "exact_match": rate("exact", "n")}

    total = sum(groups.values(), Counter())
    mf1, per_intent = macro_f1(gold_i, pred_i)
    res = {"tag": a.tag, "data": {"medications": bot.cat.source_file, "pharmacies": bot.ph.source_file},
           "overall": summarize(total) | {"intent_macro_f1": mf1},
           "by_language": {k: summarize(v) for k, v in sorted(groups.items())},
           "per_intent": per_intent,
           "llm_fallback_calls": llm_used,
           "latency_ms": {"median": round(statistics.median(lat), 2), "p95": round(sorted(lat)[int(0.95 * len(lat)) - 1], 2)}}
    (ROOT / "eval" / f"results_{a.tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(ROOT / "eval" / f"errors_{a.tag}.jsonl", "w", encoding="utf-8") as fh:
        for e in errors:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"{'group':14}" + "".join(f"{h:>10}" for h in ("n", "lang", "intent", "med_F1", "qty", "dose", "city", "exact")))
    for k, s in list(res["by_language"].items()) + [("OVERALL", res["overall"])]:
        fmt = lambda v: f"{v:>10.2f}" if isinstance(v, float) else f"{str(v):>10}"
        print(f"{k:14}" + "".join(fmt(s[x]) for x in ("n", "language_acc", "intent_acc", "med_f1", "quantity_acc",
                                                        "dosage_acc", "city_acc", "exact_match")))
    print(f"intent macro-F1 {mf1} | latency median {res['latency_ms']['median']} ms | errors: {len(errors)}"
          f" | LLM fallback used on {llm_used}/{len(rows)} | catalog: {bot.cat.source_file}")


if __name__ == "__main__":
    main()
