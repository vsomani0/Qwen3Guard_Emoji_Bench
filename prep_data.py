"""Sample the evaluation items. Writes work/items.jsonl (gitignored -- contains harmful text).

Three arms:
  harmful      -- AdvBench harmful_strings.csv, the paper's own white-box set (574 strings).
  benign_sent  -- first sentence of an EmojiAttack safe_response. PRIMARY benign control.
  benign_full  -- the same responses at full length. SECONDARY benign control.

Why two benign arms: the paper has no benign arm at all, so there is no precedent to
inherit here. `safe_responses.json` was never scored by a judge in the paper -- it is
Mistral-7B-Instruct-v0.1 output capped at max_new_tokens=100, used only as one-shot
in-context demonstrations. That cap is why its median length is 69 words against 11 for
the harmful set. First-sentence extraction closes that 6.3x gap to 1.7x while keeping
the text grammatical; we score both so the length sensitivity is measured, not assumed.
"""

import csv
import json
import random
import re
from pathlib import Path

SEED = 20250813
N_HARMFUL = 150
N_BENIGN = 75

# The paper's white-box set is described as "ranging from 3 to 44 words"; we keep that band.
MIN_WORDS, MAX_WORDS = 3, 44

ROOT = Path(__file__).parent
ADVBENCH = ROOT / "work/llm-attacks/data/advbench/harmful_strings.csv"
SAFE = ROOT / "work/EmojiAttack/in-context-data/safe_responses.json"
OUT = ROOT / "work/items.jsonl"


def norm(s):
    return " ".join(s.replace("</s>", "").replace("\n", " ").split())


def first_sentence(s):
    return re.split(r"(?<=[.!?])\s+", s)[0]


def main():
    rng = random.Random(SEED)

    harmful = [norm(r["target"]) for r in csv.DictReader(ADVBENCH.open())]
    harmful = [h for h in harmful if MIN_WORDS <= len(h.split()) <= MAX_WORDS]

    raw = [norm(v) for v in json.load(SAFE.open()).values()]
    # Drop degenerate generations (bare "</s>", truncation stubs) before sampling.
    benign = [(first_sentence(v), v) for v in raw if len(v.split()) >= 5]
    benign = [(s, f) for s, f in benign if MIN_WORDS <= len(s.split()) <= MAX_WORDS]

    print(f"pool: harmful={len(harmful)} benign={len(benign)}")
    harmful = rng.sample(harmful, N_HARMFUL)
    benign = rng.sample(benign, N_BENIGN)

    rows = []
    for i, t in enumerate(harmful):
        rows.append({"id": f"h{i:03d}", "arm": "harmful", "text": t})
    for i, (sent, full) in enumerate(benign):
        # Same source response in both arms, same id suffix -> pairable across arms too.
        rows.append({"id": f"b{i:03d}", "arm": "benign_sent", "text": sent})
        rows.append({"id": f"b{i:03d}", "arm": "benign_full", "text": full})

    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for arm in ("harmful", "benign_sent", "benign_full"):
        w = [len(r["text"].split()) for r in rows if r["arm"] == arm]
        w.sort()
        print(f"{arm:12} n={len(w):4} words min={w[0]:3} med={w[len(w)//2]:3} max={w[-1]:3}")
    print(f"wrote {len(rows)} rows -> {OUT} (seed={SEED})")


if __name__ == "__main__":
    main()
