#!/usr/bin/env python3
"""One-command verification: recomputes the paper's headline numbers from the frozen
release. Standard library only. No API key, no network, no model calls.

    python scripts/verify.py

Checks:
 1. dataset hash matches FREEZE.txt
 2. 480 unique items; 120 complete quartets; gold-pair structure valid
 3. 6,240 result records; full coverage of every system x run x item
 4. gold labels agree between data and results
 5. recomputes item accuracy, pair metrics, conditional O-E per system (means over runs)
 6. exits non-zero on any failure
"""
import csv, hashlib, json, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []

def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond: fails.append(name)

# 1. hash
data_path = os.path.join(ROOT, "data", "benchmark_480.csv")
h = hashlib.sha256(open(data_path, "rb").read()).hexdigest()
freeze = open(os.path.join(ROOT, "FREEZE.txt")).read()
check("dataset sha256 in FREEZE.txt", h in freeze, h[:16])

# 2. structure
items = list(csv.DictReader(open(data_path)))
check("480 items", len(items) == 480, str(len(items)))
ids = [i["item_id"] for i in items]
check("unique item ids", len(set(ids)) == 480)
quart = defaultdict(dict)
for i in items:
    quart[(i["lang"], i["boundary"], i["form"], i["quartet"])][i["role"]] = i
check("120 complete quartets", sum(1 for q in quart.values() if len(q) == 4) == 120)
gold_ok = all(q["A1"]["gold_intent"] == q["A2"]["gold_intent"] and
              q["B1"]["gold_intent"] == q["B2"]["gold_intent"] and
              q["A1"]["gold_intent"] != q["B1"]["gold_intent"]
              for q in quart.values() if len(q) == 4)
check("gold-pair structure (A1=A2, B1=B2, A!=B)", gold_ok)

# 3-4. results coverage
rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", "results_final_v2.jsonl"))]
check("6240 result records", len(rows) == 6240, str(len(rows)))
imap = {i["item_id"]: i for i in items}
seen = defaultdict(set)
gold_mismatch = 0
for r in rows:
    seen[(r["system"], r.get("rep", 0))].add(r["item_id"])
    if r["item_id"] in imap and r.get("gold") not in (None, imap[r["item_id"]]["gold_intent"]):
        gold_mismatch += 1
expected = {("embed_e5", 0)} | {(s, rep) for s in
            ("gptoss20", "qwen36", "gptoss20_xlate", "qwen36_xlate") for rep in (0, 1, 2)}
idset = set(ids)
cov = all(seen[k] == idset for k in expected) and set(seen) == expected
check("exact benchmark-ID coverage for every system/run", cov)
keys = [(r["system"], r.get("rep", 0), r["item_id"]) for r in rows]
check("no duplicate result keys", len(keys) == len(set(keys)))
unknown = sum(1 for r in rows if r["item_id"] not in imap)
check("no unknown item ids in results", unknown == 0, str(unknown))
check("gold labels agree data<->results", gold_mismatch == 0, str(gold_mismatch))

# 5. recompute headline metrics
def metrics(pred_of):
    ok = {i["item_id"]: (pred_of.get(i["item_id"]) == i["gold_intent"]) for i in items if i["item_id"] in pred_of}
    acc = sum(ok.values()) / len(ok)
    groups = defaultdict(lambda: defaultdict(dict))
    for i in items:
        groups[(i["boundary"], i["lang"], i["form"])][i["quartet"]][i["role"]] = i["item_id"]
    def fam(pairs):
        on = oh = 0; E = 0.0; N = 0
        for qs in groups.values():
            L, R, gp, gh = [], [], 0, 0
            for qq in qs.values():
                if len(qq) != 4: continue
                for l, r_ in pairs:
                    gp += 1; gh += ok[qq[l]] and ok[qq[r_]]
                    L.append(ok[qq[l]]); R.append(ok[qq[r_]])
            if gp:
                E += gp * (sum(L)/len(L)) * (sum(R)/len(R)); N += gp; on += gp; oh += gh
        return oh/on, E/N
    pO, pE = fam([("A1","A2"), ("B1","B2")])
    cO, cE = fam([("A1","B1"), ("A2","B2")])
    return {"acc": acc, "para_OmE": pO-pE, "contr_OmE": cO-cE}

preds = defaultdict(lambda: defaultdict(dict))
for r in rows: preds[r["system"]][r.get("rep",0)][r["item_id"]] = r["pred"]
ref = json.load(open(os.path.join(ROOT, "results", "analysis_v4.json")))
print("\nrecomputed vs stored analysis_v4.json (means over runs):")
for s in ("embed_e5","gptoss20","qwen36","gptoss20_xlate","qwen36_xlate"):
    ms = [metrics(preds[s][rep]) for rep in sorted(preds[s])]
    m = {k: sum(x[k] for x in ms)/len(ms) for k in ms[0]}
    ok_acc = abs(m["acc"] - ref[s]["acc"]["mean"]) < 5e-4
    ok_p = abs(m["para_OmE"] - ref[s]["para_OmE"]["mean"]) < 5e-4
    ok_c = abs(m["contr_OmE"] - ref[s]["contr_OmE"]["mean"]) < 5e-4
    check(f"{s}: acc/paraOE/contrOE match stored analysis", ok_acc and ok_p and ok_c,
          f"acc={m['acc']:.4f}")
    # direct agreement + collapse + per-language + repeat disagreement + malformed
    def extra(pred_of):
        agree, sep = [], []
        for q in [q for q in quart.values() if len(q) == 4]:
            gi = lambda role: q[role]["item_id"]
            if all(gi(rr) in pred_of for rr in ("A1","A2","B1","B2")):
                agree += [pred_of[gi("A1")] == pred_of[gi("A2")], pred_of[gi("B1")] == pred_of[gi("B2")]]
                sep += [pred_of[gi("A1")] != pred_of[gi("B1")], pred_of[gi("A2")] != pred_of[gi("B2")]]
        acc_l = {}
        for lang in ("hi","ta"):
            sub = [pred_of.get(i["item_id"]) == i["gold_intent"] for i in items if i["lang"] == lang and i["item_id"] in pred_of]
            acc_l[lang] = sum(sub)/len(sub)
        return sum(agree)/len(agree), sum(sep)/len(sep), acc_l
    ex = [extra(preds[s][rep]) for rep in sorted(preds[s])]
    ag = sum(x[0] for x in ex)/len(ex); sp = sum(x[1] for x in ex)/len(ex)
    ok2 = (abs(ag - ref[s]["para_agree"]["mean"]) < 5e-4 and
           abs((1-sp) - ref[s]["collapse"]["mean"]) < 5e-4 and
           abs(sum(x[2]["hi"] for x in ex)/len(ex) - ref[s]["acc_hi"]["mean"]) < 5e-4 and
           abs(sum(x[2]["ta"] for x in ex)/len(ex) - ref[s]["acc_ta"]["mean"]) < 5e-4)
    check(f"{s}: agreement/collapse/per-language match stored analysis", ok2)
    mal = sum(1 for r in rows if r["system"] == s and r["pred"] in ("ERROR","PARSE_FAIL"))
    tot = sum(1 for r in rows if r["system"] == s)
    check(f"{s}: malformed count matches stored", f"{mal}/{tot}" == ref["malformed"][s], f"{mal}/{tot}")
    if s != "embed_e5":
        reps_ = sorted(preds[s])
        dis = sum(1 for i in items if len({preds[s][r].get(i["item_id"]) for r in reps_}) > 1)
        check(f"{s}: repeat disagreement matches stored", abs(dis/len(items) - ref[s]["repeat_disagreement"]) < 5e-4)

print()
if fails:
    print(f"VERIFY FAILED: {len(fails)} check(s): {fails}"); sys.exit(1)
print("VERIFY PASSED: all checks OK")
