#!/usr/bin/env python3
"""v2 paper numbers from analysis_v4.json — single source of truth.
Writes: numbers.tex (macros), main_table.tex, lang_table.tex, cis_table.tex, script_table.tex"""
import json

d = json.load(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "results", "analysis_v4.json")))
S = ["embed_e5", "gptoss20", "qwen36", "gptoss20_xlate", "qwen36_xlate"]
NAMES = {"embed_e5": "e5", "gptoss20": "gpt-oss-20b", "qwen36": "qwen3.6-27b",
         "gptoss20_xlate": "gpt-oss$\\rightarrow$EN", "qwen36_xlate": "qwen$\\rightarrow$EN"}

def f3(x): return f"{x:.3f}".lstrip("0") if abs(x) < 1 else f"{x:.3f}"
def mmm(s, k):
    v = d[s][k]
    return f'{f3(v["mean"])}' if v["min"] == v["max"] else f'{f3(v["mean"])} {{\\scriptsize[{f3(v["min"])},{f3(v["max"])}]}}'
def mean(s, k): return d[s][k]["mean"]

# ---- main table (overall, conditional O-E as primary) ----
rows = [
 ("Item accuracy", lambda s: mmm(s, "acc")),
 ("Macro-F1 {\\scriptsize(\\nGold{} gold intents)}", lambda s: f3(mean(s, "macro_f1_gold"))),
 ("\\midrule Paraphrase label agreement", lambda s: f3(mean(s, "para_agree"))),
 ("Paraphrase both-correct", lambda s: f3(mean(s, "para_both"))),
 ("\\quad expected (conditional)", lambda s: f3(mean(s, "para_E"))),
 ("\\quad observed$-$expected", lambda s: f'{mean(s,"para_OmE"):+.3f}'),
 ("\\midrule Contrast label separation", lambda s: f3(mean(s, "contr_sep"))),
 ("Label collapse", lambda s: f3(mean(s, "collapse"))),
 ("Contrast both-correct", lambda s: f3(mean(s, "contr_both"))),
 ("\\quad expected (conditional)", lambda s: f3(mean(s, "contr_E"))),
 ("\\quad observed$-$expected", lambda s: f'{mean(s,"contr_OmE"):+.3f}'),
 ("\\midrule Repeat disagreement",
  lambda s: (f'{d[s]["repeat_disagreement"]*100:.1f}\\%' if "repeat_disagreement" in d[s] else "---")),
]
t = ["\\begin{table*}[t]\\centering\\small",
     "\\begin{tabular}{l" + "c"*len(S) + "}", "\\toprule",
     " & " + " & ".join(NAMES[s] for s in S) + " \\\\", "\\midrule"]
for label, fn in rows:
    if label.startswith("\\midrule"):
        t.append("\\midrule"); label = label.split(" ", 1)[1]
    t.append(label + " & " + " & ".join(fn(s) for s in S) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}",
      "\\caption{Main results over all 480 items. LLM entries are means over three runs at "
      "temperature 0; brackets in the item-accuracy row show the min--max range across runs. "
      "Expected values are conditional independence expectations computed within "
      "boundary$\\times$language$\\times$form groups; all 95\\% CIs are in Table~\\ref{tab:cis}. "
      "All values generated from a single results file.}",
      "\\label{tab:main}", "\\end{table*}"]
open("main_table.tex", "w").write("\n".join(t) + "\n")

# ---- language/script table ----
t = ["\\begin{table}[t]\\centering\\small", "\\begin{tabular}{lcccc}", "\\toprule",
     " & \\multicolumn{2}{c}{Hindi} & \\multicolumn{2}{c}{Tamil} \\\\",
     " & Deva. & Roman & Script & Roman \\\\", "\\midrule"]
for s in S:
    t.append(f'{NAMES[s]} & {f3(mean(s,"acc_hi_native"))} & {f3(mean(s,"acc_hi_roman"))}'
             f' & {f3(mean(s,"acc_ta_native"))} & {f3(mean(s,"acc_ta_roman"))} \\\\')
t += ["\\bottomrule", "\\end{tabular}",
      "\\caption{Accuracy by language and script form (means over runs). Hindi forms are "
      "independently authored (stratified comparison); Tamil forms are parallel author-rendered "
      "pairs (paired comparison; Table~\\ref{tab:script}).}",
      "\\label{tab:lang}", "\\end{table}"]
open("lang_table.tex", "w").write("\n".join(t) + "\n")

# ---- Tamil paired script table ----
t = ["\\begin{table}[t]\\centering\\small", "\\begin{tabular}{lcc}", "\\toprule",
     "System & Roman$-$Script & 95\\% CI \\\\", "\\midrule"]
for s in S:
    p = d[s]["ta_script_paired"]
    t.append(f'{NAMES[s]} & {p["point"]*100:+.1f} & [{p["ci"][0]*100:+.1f}, {p["ci"][1]*100:+.1f}] \\\\')
t += ["\\bottomrule", "\\end{tabular}",
      "\\caption{Tamil paired script comparison (points): per-item accuracy difference between "
      "each romanized item and its parallel Tamil-script rendition; boundary-clustered paired "
      "bootstrap, with statistics averaged over the three frozen runs within each replicate.}",
      "\\label{tab:script}", "\\end{table}"]
open("script_table.tex", "w").write("\n".join(t) + "\n")

# ---- CI table: 15 primary / 30 + 120 sensitivity ----
t = ["\\begin{table}[H]\\centering\\small", "\\begin{tabular}{lcc}", "\\toprule",
     "System & Paraphrase O$-$E CI & Contrast O$-$E CI \\\\", "\\midrule"]
for scheme, label in (("ci15", ""), ("ci30", " {\\scriptsize(30)}"), ("ci120", " {\\scriptsize(120)}")):
    for s in S:
        pc, cc = d[s][f"para_OmE_{scheme}"], d[s][f"contr_OmE_{scheme}"]
        t.append(f'{NAMES[s]}{label} & [{pc[0]:+.3f}, {pc[1]:+.3f}] & [{cc[0]:+.3f}, {cc[1]:+.3f}] \\\\')
    if scheme != "ci120": t.append("\\midrule")
t += ["\\bottomrule", "\\end{tabular}",
      "\\caption{95\\% bootstrap CIs for conditional O$-$E. Primary: 15 boundary clusters "
      "(5{,}000 resamples, multiplicity-preserving, statistics averaged over the three runs "
      "within each replicate); sensitivity: 30 language$\\times$boundary and 120 quartet "
      "clusters (2{,}000 resamples).}",
      "\\label{tab:cis}", "\\end{table}"]
open("cis_table.tex", "w").write("\n".join(t) + "\n")


# ---- per-boundary table (pooled over languages/forms, mean over runs) ----
pbj = json.load(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "results", "per_boundary_pooled.json")))
t = ["\\begin{table}[H]\\centering\\small", "\\begin{tabular}{l" + "c"*len(S) + "}", "\\toprule",
     "Boundary & " + " & ".join(NAMES[s] for s in S) + " \\\\", "\\midrule"]
for b in sorted(pbj):
    t.append(b + " & " + " & ".join(f3(pbj[b][s]) for s in S) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}",
      "\\caption{Accuracy per boundary, pooled over languages and script forms, means over "
      "runs (32 items per boundary; boundary definitions in Appendix~\\ref{app:boundaries}).}",
      "\\label{tab:boundaryacc}", "\\end{table}"]
open("boundary_table.tex", "w").write("\n".join(t) + "\n")

# ---- macros ----
mac = ["% AUTO-GENERATED from analysis_v4.json — do not hand-edit numbers"]
def M(n, v): mac.append(f"\\newcommand{{\\{n}}}{{{v}}}")
eq = [max(abs(d[s]["para_OmE_ci15"][0]), abs(d[s]["para_OmE_ci15"][1]),
          abs(d[s]["contr_OmE_ci15"][0]), abs(d[s]["contr_OmE_ci15"][1])) for s in S]
M("equivMax", f"{max(eq)*100:.1f}"); M("equivMin", f"{min(eq)*100:.1f}")
M("nGold", str(d["dataset"]["distinct_gold"]))
M("accQwen", f3(mean("qwen36", "acc"))); M("accGpt", f3(mean("gptoss20", "acc")))
for s_, mc in (("qwen36","DisQwen"),("qwen36_xlate","DisQwenX"),("gptoss20","DisGpt"),("gptoss20_xlate","DisGptX"),("embed_e5","DisE")):
    M(mc, f"{(1-mean(s_,'para_agree'))*100:.1f}")
M("repDisQwen", f'{d["qwen36"]["repeat_disagreement"]*100:.1f}\\%')
M("repDisGpt", f'{d["gptoss20"]["repeat_disagreement"]*100:.1f}\\%')
for fam, mc in (("gptoss20", "XlDgpt"), ("qwen36", "XlDqwen")):
    M(mc, f'{(mean(fam+"_xlate","acc")-mean(fam,"acc"))*100:+.1f}')
p = d["qwen36"]["ta_script_paired"]
M("taScrQwen", f'{p["point"]*100:+.1f}'); M("taScrQwenCI", f'[{p["ci"][0]*100:+.1f}, {p["ci"][1]*100:+.1f}]')
open("numbers.tex", "w").write("\n".join(mac) + "\n")
print("v3 tables + macros written")
print("headline:", {s: round(mean(s, "acc"), 3) for s in S})
