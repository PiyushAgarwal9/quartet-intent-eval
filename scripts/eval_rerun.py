#!/usr/bin/env python3
"""OPTIONAL model re-running: regenerates raw per-item predictions only.
Requires GROQ_API_KEY. All metrics come from scripts/analysis.py (single authoritative
implementation). Verification (scripts/verify.py) does NOT need this script."""
import argparse, csv, json, os, re, ssl, subprocess, time, urllib.request
import certifi, yaml

BASE = os.path.dirname(os.path.abspath(__file__))
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
UA_HEADERS = {"User-Agent": "curl/8.4.0", "Accept": "application/json"}  # Groq CF blocks python UA
TAXONOMY = os.environ.get("TAXONOMY_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "taxonomy.yaml"))
MODELS = {"gptoss20": "openai/gpt-oss-20b", "qwen36": "qwen/qwen3.6-27b"}

# ---------- taxonomy ----------
tax = yaml.safe_load(open(TAXONOMY))
INTENTS = {}   # name -> {desc, examples}
for dom, dv in tax["domains"].items():
    for name, spec in dv.get("intents", {}).items():
        INTENTS[name] = {"desc": spec.get("description") or name.replace("_", " "),
                         "examples": (spec.get("examples") or [])[:3], "domain": dom}
INTENT_NAMES = sorted(INTENTS)
INTENT_LIST = "\n".join(INTENT_NAMES)

CLS_SYS = ("You are an intent classifier for an Indian banking assistant. "
           "Classify the customer message into EXACTLY ONE intent from this list:\n" + INTENT_LIST +
           "\nRules: choose the single best intent name from the list, verbatim. "
           'Output JSON only: {"intent": "<name>"}')
XLATE_SYS = ("Translate the Indian banking customer message to plain English. "
             "Preserve the exact meaning, including whether it is a request to DO something "
             "or a QUESTION about something. Output only the English translation.")

# ---------- groq ----------
def _find_key():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Set GROQ_API_KEY to re-run models (verification needs no key).")
    return key

def groq_raw(k, model, system, user, max_tokens):
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(GROQ_URL, data=body, headers={
        "Authorization": f"Bearer {k}", "Content-Type": "application/json", **UA_HEADERS})
    with urllib.request.urlopen(req, timeout=90, context=SSL_CTX) as r:
        return json.load(r)["choices"][0]["message"]["content"]

def strip_think(txt):
    """Remove <think>...</think> reasoning blocks (qwen3.6 emits them in content)."""
    if not txt:
        return txt
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    if "<think>" in txt:          # unclosed think block = truncated output; nothing usable
        txt = txt.split("<think>")[0]
    return txt.strip()

KEY = None
def groq(model, system, user, max_tokens=4096):
    global KEY
    if KEY is None:
        KEY = _find_key()
    for attempt in range(4):
        try:
            out = strip_think(groq_raw(KEY, model, system, user, max_tokens))
            if out:
                return out
            # empty after stripping = reasoning ate the budget; retry with more room
            max_tokens = min(max_tokens * 2, 16384)
        except Exception as e:
            if attempt == 3:
                print(f"GROQ FAIL [{model}]: {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    return None

def parse_intent(txt):
    if not txt:
        return "ERROR"
    m = re.search(r'"intent"\s*:\s*"([^"]+)"', txt)
    if m and m.group(1) in INTENTS:
        return m.group(1)
    for name in INTENT_NAMES:
        if name in (txt or ""):
            return name
    return "PARSE_FAIL"

# ---------- systems ----------
_E5 = {"model": None, "mat": None}
def embed_e5(text):
    if _E5["model"] is None:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        _E5["np"] = np
        _E5["model"] = SentenceTransformer(
            "intfloat/multilingual-e5-base",
            revision="835193815a3936a24a0ee7dc9e3d48c1fbb19c55")
        passages = ["passage: " + INTENTS[n]["desc"] + ". " + " ".join(INTENTS[n]["examples"])
                    for n in INTENT_NAMES]
        _E5["mat"] = _E5["model"].encode(passages, normalize_embeddings=True)
        print("e5 index built (163 intents)", flush=True)
    q = _E5["model"].encode(["query: " + text], normalize_embeddings=True)
    sims = (_E5["mat"] @ q.T).ravel()
    return INTENT_NAMES[int(sims.argmax())], float(sims.max())


def run_system(system, text):
    """returns (pred, extra_dict)"""
    if system == "embed_e5":
        p, s = embed_e5(text); return p, {"sim": round(s, 4)}
    if system in ("gptoss20", "qwen36"):
        return parse_intent(groq(MODELS[system], CLS_SYS, text)), {}
    if system.endswith("_xlate"):
        fam = system[:-6]
        en = (groq(MODELS[fam], XLATE_SYS, text) or "").strip()
        return parse_intent(groq(MODELS[fam], CLS_SYS, en)), {"en": en}
    raise ValueError(system)

# ---------- data ----------
def load_items(path):
    items = []
    for row in csv.DictReader(open(path)):
        if not (row.get("text") or "").strip():
            continue
        row["quartet_key"] = f'{row["boundary"]}-{row["lang"]}-{row["form"]}-q{row.get("quartet","1")}'
        items.append(row)
    return items

# ---------- main ----------
LLM_SYSTEMS = {"gptoss20", "qwen36", "gptoss20_xlate", "qwen36_xlate"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--systems", default="embed_e5,gptoss20,qwen36,gptoss20_xlate,qwen36_xlate")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--repeats", type=int, default=3, help="repeats for LLM arms (A1 amendment); deterministic arms run once")
    ap.add_argument("--metrics-only", action="store_true")
    args = ap.parse_args()

    items = load_items(args.data)
    systems = args.systems.split(",")
    res_path = os.path.join(BASE, f"results_{args.tag}.jsonl")
    # A1 error policy: EVERY recorded row is final (ERROR/PARSE_FAIL scores as wrong; no selective reruns)
    done = {}
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line)
            done[(r["item_id"], r["system"], r.get("rep", 0))] = r["pred"]

    if not args.metrics_only:
        out = open(res_path, "a")
        for n, it in enumerate(items, 1):
            for s in systems:
                reps = args.repeats if s in LLM_SYSTEMS else 1
                for rep in range(reps):
                    if (it["item_id"], s, rep) in done:
                        continue
                    pred, extra = run_system(s, it["text"])
                    out.write(json.dumps({"item_id": it["item_id"], "system": s, "rep": rep,
                                          "pred": pred, "gold": it["gold_intent"],
                                          "model": MODELS.get(s.replace("_xlate", ""), s),
                                          "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                          **extra}, ensure_ascii=False) + "\n")
                    out.flush()
                    done[(it["item_id"], s, rep)] = pred
            if n % 20 == 0 or n == len(items):
                print(f"{n}/{len(items)}", flush=True)
        out.close()
    print("raw predictions written; run scripts/analysis.py for all metrics")

if __name__ == "__main__":
    main()
