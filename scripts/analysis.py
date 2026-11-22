#!/usr/bin/env python3
"""v4 analysis — PM round-5 corrections:
- bootstrap preserves cluster multiplicity (unique per-draw IDs)
- clusters: PRIMARY 15 boundaries; sensitivity 30 lang x boundary; sensitivity 120 quartets
- bootstrap statistic = mean over the three frozen LLM runs within each replicate
- direct paraphrase disagreement rates reported
- malformed counts from final results
Single source of truth for the final paper numbers."""
import csv, json, random, sys
from collections import defaultdict

import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = list(csv.DictReader(open(f"{BASE}/data/benchmark_480.csv")))
for it in ITEMS:
    it["qk"] = f'{it["lang"]}-{it["boundary"]}-{it["form"]}-q{it["quartet"]}'
IMAP = {i["item_id"]: i for i in ITEMS}
ROWS = [json.loads(l) for l in open(f"{BASE}/results/results_final_v2.jsonl")]
SYSTEMS = ["embed_e5", "gptoss20", "qwen36", "gptoss20_xlate", "qwen36_xlate"]

preds = defaultdict(lambda: defaultdict(dict))
malformed = defaultdict(int); total_calls = defaultdict(int)
for r in ROWS:
    preds[r["system"]][r.get("rep", 0)][r["item_id"]] = r["pred"]
    total_calls[r["system"]] += 1
    if r["pred"] in ("ERROR", "PARSE_FAIL"): malformed[r["system"]] += 1

def core(pred_of, items):
    """items may contain bootstrap COPIES: each has base_id (for preds) + unique qk."""
    ok, pd = {}, {}
    for i in items:
        bid = i.get("base_id", i["item_id"])
        if bid in pred_of:
            uid = i.get("uid", i["item_id"])
            ok[uid] = (pred_of[bid] == i["gold_intent"]); pd[uid] = pred_of[bid]
    if not ok: return None
    acc = sum(ok.values()) / len(ok)
    groups = defaultdict(lambda: defaultdict(dict))
    for i in items:
        uid = i.get("uid", i["item_id"])
        if uid in ok:
            groups[(i.get("gk", (i["boundary"], i["lang"], i["form"])))][i.get("qk")] = \
                groups[(i.get("gk", (i["boundary"], i["lang"], i["form"])))].get(i.get("qk"), {})
            groups[(i.get("gk", (i["boundary"], i["lang"], i["form"])))][i.get("qk")][i["role"]] = uid
    def fam(pair_roles, want_agree):
        obs_n = obs_h = 0; E_num = 0.0; N = 0; direct = []
        for g, qs in groups.items():
            L, R, gp, gh = [], [], 0, 0
            for qq in qs.values():
                if len(qq) != 4: continue
                for l, r_ in pair_roles:
                    gp += 1; gh += ok[qq[l]] and ok[qq[r_]]
                    L.append(ok[qq[l]]); R.append(ok[qq[r_]])
                    direct.append((pd[qq[l]] == pd[qq[r_]]) if want_agree else (pd[qq[l]] != pd[qq[r_]]))
            if gp:
                E_num += gp * (sum(L)/len(L)) * (sum(R)/len(R)); N += gp
                obs_n += gp; obs_h += gh
        if not obs_n: return None
        return {"O": obs_h/obs_n, "E": E_num/N, "OmE": obs_h/obs_n - E_num/N,
                "direct": sum(direct)/len(direct)}
    p = fam([("A1","A2"),("B1","B2")], True)
    c = fam([("A1","B1"),("A2","B2")], False)
    if not p or not c: return None
    res = {"acc": acc, "para_agree": p["direct"], "para_both": p["O"], "para_E": p["E"], "para_OmE": p["OmE"],
           "contr_sep": c["direct"], "collapse": 1-c["direct"], "contr_both": c["O"], "contr_E": c["E"], "contr_OmE": c["OmE"]}
    # macro over gold-supported labels (renamed in paper)
    golds = sorted({i["gold_intent"] for i in items})
    f1s = []
    base = {i.get("uid", i["item_id"]): i for i in items}
    for g in golds:
        tp = sum(1 for u, o in ok.items() if base[u]["gold_intent"] == g and pd[u] == g)
        fp = sum(1 for u in ok if base[u]["gold_intent"] != g and pd[u] == g)
        fn = sum(1 for u in ok if base[u]["gold_intent"] == g and pd[u] != g)
        f1s.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0)
    res["macro_f1_gold"] = sum(f1s)/len(f1s)
    for lang in ("hi","ta"):
        for form in (None,"native","roman"):
            sub = [ok[i.get("uid", i["item_id"])] for i in items
                   if i.get("uid", i["item_id"]) in ok and i["lang"]==lang and (form is None or i["form"]==form)]
            if sub: res[f"acc_{lang}" + (f"_{form}" if form else "")] = sum(sub)/len(sub)
    return res

def draw(bmap, keys, rnd):
    """resample clusters WITH multiplicity:每 draw gets unique uid/qk/gk suffixes."""
    sample = []
    for d in range(len(keys)):
        k = rnd.choice(keys)
        for i in bmap[k]:
            c = dict(i)
            c["base_id"] = i["item_id"]; c["uid"] = f'{i["item_id"]}#d{d}'
            c["qk"] = f'{i["qk"]}#d{d}'; c["gk"] = (i["boundary"], i["lang"], i["form"], d)
            sample.append(c)
    return sample

def boot(system, statname, clusterfn, n=5000, seed=31):
    bmap = defaultdict(list)
    for i in ITEMS: bmap[clusterfn(i)].append(i)
    keys = list(bmap); rnd = random.Random(seed)
    reps = sorted(preds[system]); vals = []
    for _ in range(n):
        sample = draw(bmap, keys, rnd)
        rv = []
        for rep in reps:
            c = core(preds[system][rep], sample)
            if c and c.get(statname) is not None: rv.append(c[statname])
        if rv: vals.append(sum(rv)/len(rv))
    vals.sort()
    return [round(vals[int(.025*len(vals))], 4), round(vals[int(.975*len(vals))-1], 4)]

def tamil_paired(system):
    reps = sorted(preds[system])
    def rep_diffs(rep):
        p0 = preds[system][rep]; out = defaultdict(list)
        for i in ITEMS:
            if i["lang"]=="ta" and i["form"]=="roman":
                nid = i["item_id"].replace("-rom-","-nat-")
                if i["item_id"] in p0 and nid in p0:
                    out[i["boundary"]].append(int(p0[i["item_id"]]==i["gold_intent"]) - int(p0[nid]==IMAP[nid]["gold_intent"]))
        return out
    per_rep = [rep_diffs(r) for r in reps]
    point = sum(sum(sum(d[b]) for b in d) for d in per_rep) / sum(sum(len(d[b]) for b in d) for d in per_rep)
    keys = list(per_rep[0]); rnd = random.Random(37); vals = []
    for _ in range(5000):
        ks = [rnd.choice(keys) for _ in keys]
        rv = []
        for d in per_rep:
            s = [x for k in ks for x in d[k]]
            rv.append(sum(s)/len(s))
        vals.append(sum(rv)/len(rv))
    vals.sort()
    disc = sum(1 for x in per_rep[0][keys[0]] for _ in [0]) # placeholder
    n_pairs = sum(len(per_rep[0][k]) for k in keys)
    n_disc = sum(1 for k in keys for x in per_rep[0][k] if x != 0)
    return {"point": round(point,4), "ci": [round(vals[int(.025*len(vals))],4), round(vals[int(.975*len(vals))-1],4)],
            "n_pairs": n_pairs, "n_discordant_rep0": n_disc}

report = {"dataset": {"n_items": len(ITEMS), "distinct_gold": len({i["gold_intent"] for i in ITEMS}),
                      "quartets": len({i["qk"] for i in ITEMS})},
          "malformed": {s: f"{malformed[s]}/{total_calls[s]}" for s in SYSTEMS}}
for s in SYSTEMS:
    reps = sorted(preds[s])
    per_rep = [core(preds[s][r], ITEMS) for r in reps]
    per_rep = [p for p in per_rep if p]
    agg = {}
    for k in per_rep[0]:
        vs = [m[k] for m in per_rep if m.get(k) is not None]
        agg[k] = {"mean": round(sum(vs)/len(vs), 4), "min": round(min(vs), 4), "max": round(max(vs), 4)}
    print(f"[{s}] bootstrapping...", flush=True)
    agg["para_OmE_ci15"]  = boot(s, "para_OmE",  lambda i: i["boundary"])
    agg["contr_OmE_ci15"] = boot(s, "contr_OmE", lambda i: i["boundary"])
    agg["para_OmE_ci30"]  = boot(s, "para_OmE",  lambda i: (i["lang"], i["boundary"]), n=2000)
    agg["contr_OmE_ci30"] = boot(s, "contr_OmE", lambda i: (i["lang"], i["boundary"]), n=2000)
    agg["para_OmE_ci120"] = boot(s, "para_OmE",  lambda i: i["qk"], n=2000)
    agg["contr_OmE_ci120"]= boot(s, "contr_OmE", lambda i: i["qk"], n=2000)
    if len(reps) > 1:
        dis = sum(1 for i in ITEMS if len({preds[s][r].get(i["item_id"]) for r in reps}) > 1)
        agg["repeat_disagreement"] = round(dis/len(ITEMS), 4)
    agg["ta_script_paired"] = tamil_paired(s)
    report[s] = agg

json.dump(report, open(f"{BASE}/results/analysis_v4.json", "w"), indent=2)
print("analysis_v4.json written")
for s in SYSTEMS:
    a = report[s]
    print(f'{s:15s} acc={a["acc"]["mean"]:.3f} paraDisagree={(1-a["para_agree"]["mean"])*100:.1f}% '
          f'paraOmE={a["para_OmE"]["mean"]*100:+.2f}pp CI15={a["para_OmE_ci15"]} '
          f'contrOmE={a["contr_OmE"]["mean"]*100:+.2f}pp CI15={a["contr_OmE_ci15"]} '
          f'taScript={a["ta_script_paired"]["point"]*100:+.1f} {a["ta_script_paired"]["ci"]}')
print("malformed:", report["malformed"])
