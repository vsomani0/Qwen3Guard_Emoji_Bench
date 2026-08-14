# Emoji Attack Replication — Project Brief

## Goal

Replicate and extend the token-segmentation attack from **Emoji Attack: Enhancing Jailbreak
Attacks Against Judge LLM Detection** (Wei, Liu & Erichson, ICML 2025) against a current-generation
guardrail model.

Core question: **is the vulnerability tokenizer-driven or semantics-driven, and does it persist in
2025-generation guardrails?**

- Paper: Stored in Emoji_Attack_Paper.pdf
- Official repo: https://github.com/zhipeng-wei/EmojiAttack

This is a ~total 1-2 hour scoped side-project. Prefer a clean small result over an ambitious
unfinished one. Deliverable: a public repo with a README stating threat model, n, and limitations.

## Hard constraints

- **Hardware:** MacBook, 16GB unified memory, MPS. No GPU rental, no cloud.
- **No fine-tuning.** Guardrail models ship ready to classify.
- Budget ~10GB for model weights; macOS + Python + activations consume the rest.

## Do NOT use the original repo's pipeline

The repo assumes local Llama weights conversion, an unaligned Mistral-7B for in-context generation,
`gtr-t5-xl` as a surrogate, and an HF token. Its `requirements.txt` is a conda spec-file that will
likely fail on Apple Silicon. **Take only the data files; write our own ~60 lines.**

Specifically: skip `InContext_ObtainOutputs.py`, `BlackAttack_*.py`, EasyJailbreak, and the whole
target-LLM generation stage. We are replicating the paper's **Table 4** design (direct string
transform), not Table 2 (in-context insertion via a target LLM). Table 4 is cleaner — no confound
about whether a target model complied with the emoji instruction.

## Data

Shallow-clone rather than guessing raw URLs:

```bash
git clone --depth 1 https://github.com/zhipeng-wei/EmojiAttack.git
git clone --depth 1 https://github.com/llm-attacks/llm-attacks.git
```

- **Harmful set:** `harmful_strings.csv` from AdvBench (574 strings, under `data/advbench/` in
  llm-attacks). Sample with a fixed seed.
- **Benign control:** `in-context-data/safe_responses.json` in the EmojiAttack repo.
- Optional second harmful set: `datasets/en.txt` (402 offensive phrases).

**Do not commit harmful strings to the public repo.** Reference the source and ship the sampling seed.

## Experiment design

### Transform (the entire attack primitive)

```python
import re

def insert_char(word, ch):
    core = re.match(r"^\W*(\w+)", word)
    if not core or len(core.group(1)) < 3:
        return word
    s, e = core.span(1)
    j = s + (e - s) // 2
    return word[:j] + ch + word[j:]

def transform(text, ch):
    if ch is None:
        return text
    return " ".join(insert_char(w, ch) for w in text.split())
```

### Four conditions

| Condition | `ch` | Isolates |
|---|---|---|
| clean | `None` | baseline |
| mid-split | `" "` | tokenization alone |
| positive emoji | `"😊"` | tokenization + positive semantics |
| toxic emoji | `"🖕"` | tokenization + negative semantics |

**The space-split vs 😊 vs 🖕 contrast is the scientific content.** If space-split does nothing but
😊 works, the mechanism is semantic, not tokenizer-level — which would contradict the paper's own
framing. Paper's Table 3 already hints at this: for commercial judges, 😊 lowers detection while 🖕
*raises* it.

### Outcome

Continuous probability from logits, **not** a parsed verdict. This is the whole reason to run
locally. Store raw records to JSONL (text, condition, probabilities, generated string) so analysis
can be redone without re-running inference.

### Analysis

Paired throughout — same string across all conditions.

- Wilcoxon signed-rank on the probability, clean vs each attack condition
- McNemar on thresholded verdicts
- Bootstrap 95% CI on mean per-item delta
- Report all comparisons; note whether multiple-comparison correction was applied
- One figure: distribution of label probabilities by condition, harmful and benign side by side

## Model

**Qwen3Guard-Gen** (Alibaba, Sept 2025) — https://github.com/QwenLM/Qwen3Guard

Available in 0.4B. **Primary model: Qwen3Guard-Gen-4B, fp16 (~8GB). Confirmed — do not
substitute.**

Running fp16 at 4B on 16GB is tight but workable. Close other applications before running. If
loading OOMs, tell Vivek and ask before falling back — do not silently quantize or downsize.

### Three-tier output — important

Qwen3Guard emits `Safe` / `Controversial` / `Unsafe`, not binary. Two consequences:

1. Logit extraction needs three token IDs. Verify each label is a single token in the tokenizer; if
   `Controversial` splits, use its first token consistently and document that.
2. Report both **Strict Mode** (controversial → unsafe) and **Loose Mode** (controversial → safe),
   matching the technical report's convention.

This is an advantage over the original paper: softmax over three labels shows *where* items move.
Unsafe → Controversial is "degraded confidence"; Unsafe → Safe is "successful evasion." The paper's
binary setup could not distinguish these. **Plot the full three-way shift, not just P(unsafe).**

### Logit extraction gotcha

Taking logits at the last prompt position often fails — models emit leading whitespace/newlines
before the label. Generate a few tokens with `output_scores=True` and scan for the step where the
argmax is one of the label tokens. Check the model card for the exact chat template and output
format (`Safety: <label>` plus categories).

## Mandatory sanity checks — run before trusting any number

In order. Do not skip.

1. **Clean condition must separate.** High P(unsafe) on harmful, low on benign. If it doesn't, the
   chat template is wrong and every downstream number is noise. This is the #1 failure mode.
2. **Confirm tokenization actually changed.** Compare `len(tok(clean).input_ids)` vs the emoji
   version. Should jump substantially. If not, the transform isn't doing what the paper claims.
3. **Count emoji tokens.** Whether 😊 is one token or several affects interpretation.
4. **Eyeball ~5 generations per condition** for degenerate output.
5. **Log parse/failure rate per condition.** If emoji conditions fail more often and those rows get
   dropped, the result is biased. Report the rate.

## Open decisions, suggested by Claude on Web — ASK VIVEK BEFORE PROCEEDING

These are suggestions, not settled. Vivek has not verified them and explicitly wants to be consulted
rather than have them assumed. Ask; do not silently pick.

1. **Batching and context length at 4B.** fp16 4B on 16GB leaves limited headroom for activations
   and KV cache. Suggest batch size 1 and truncating over-long inputs. Confirm with Vivek before
   choosing a truncation limit — it interacts with the transform, since emoji insertion roughly
   doubles or triples token count on the same string.
2. **MLX vs transformers+MPS.** MLX is likely lower-memory and faster on Apple Silicon and exposes
   logits, but adds a dependency Vivek hasn't used. Unverified suggestion — this matters more at 4B
   than it would at 0.6B, since memory headroom is the binding constraint.
3. **Sample sizes.** Suggested n=100 harmful, n=50 benign. Not fixed.
4. **Transform rules.** Skipping words under 3 characters, and splitting at the midpoint of the
   alphabetic run rather than the raw string. The paper is vague on both; these are our choices and
   must be documented either way. Look at the github repo for information about #4
5. **Recoverability check.** Optional extension: have a clean model strip emojis and reconstruct,
   then measure similarity to the original. Tests whether the harmful payload survives the transform
   at all — the paper reports no attack-utility metric anywhere, and its own Appendix E shows the
   text getting corrupted past recovery. Nice-to-have, not core.

## Reference numbers from the paper

For sanity-checking direction of effects.

- **Open-source guardrails collapse.** ShieldLM 78.4% → 3.0% unsafe-prediction under white-box emoji
  attack. Llama Guard 81.3% → 35.1%.
- **Commercial judges do not.** On the same harmful-response set: GPT-4 96.2% → 98.2%, Claude 97.0%
  → 97.7%, Gemini 91.3% → 92.2%, GPT-3.5 58.3% → 87.7%. All *up*.
- **Emoji valence matters** (Table 3, CodeChameleon baseline): 😊 reduces detection sharply on most
  judges; 🖕 and 😈 often raise it on commercial models.
- Paper reports a headline ~12% average reduction across ten judges.

**Expect a null or attenuated effect on a 2025 guardrail.** That is a fine, publishable-as-a-README
result: "the tokenization vulnerability appears specific to earlier small guardrails and does not
reproduce on current ones." Frame it that way from the start; don't treat it as failure.

## Known gaps in the original paper (our additions)

- **No false-positive control anywhere.** The paper reports only unsafe-prediction-ratio on harmful
  content — TPR with no FPR. So it cannot distinguish "judge was fooled" from "judge became more
  permissive generally," and the latter is fixable by moving a threshold. Our benign arm addresses
  this. **This is the main thing that makes the project more than a rerun — do not cut it.**
- **No attack-utility metric.** No check that the harmful payload survives the transform.
- **Binary thresholding** discards score information.

## Limitations to state in the README

- AdvBench `harmful_strings` are short and blunt (3–44 words); unlike realistic jailbreak outputs.
- Single model family, single language.
- Ceiling/floor effects if the clean baseline sits near 0% or 100%.
- Model size: 4B was chosen to fit 16GB in fp16. Results may not transfer to the 8B variant, and
  published work suggests robustness does not scale monotonically within guardrail families.
- n and the resulting confidence intervals — do not claim precision the sample size doesn't support.

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate pandas scipy matplotlib
python -c "import torch; print(torch.backends.mps.is_available())"
```

Do not use the original repo's conda file.

Develop against the 0.6B first — it loads in seconds. Debug the chat template and sanity checks
there, then swap in the larger model.

Free memory between arms with separate script invocations rather than in-process cleanup; more
reliable on MPS.
