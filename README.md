# Does the Emoji Attack survive a 2025 guardrail?

A scoped replication and extension of **Emoji Attack: Enhancing Jailbreak Attacks Against Judge
LLM Detection** (Wei, Liu & Erichson, ICML 2025 — [arXiv](https://arxiv.org/abs/2411.01077) ·
[code](https://github.com/zhipeng-wei/EmojiAttack)), run against **Qwen3Guard-Gen** (Alibaba, Sept 2025).

**LLM Disclosure**: This README was initially written by Claude Opus 5. README was reviewed and edited by Vivek.

**Status:** results below are for **Qwen3Guard-Gen-0.6B**.

## Headline

**The emoji results replicate to Qwen3Guard as a monitor model. Additionally, monitor vulnerability through emojis seems to be primarily semantic for this model.**


| delimiter | token inflation | P(unsafe) on harmful | Δ vs clean | 95% CI |
|---|---|---|---|---|
| *clean* | 1.00x | 0.824 | — | — |
| `" "` | 1.84x | 0.494 | **−0.330** | [−0.382, −0.278] |
| `"#"` | 2.44x | 0.644 | **−0.180** | [−0.222, −0.139] |
| `"😊"` | 2.44x | 0.394 | **−0.430** | [−0.487, −0.372] |
| `"🖕"` | 2.44x | 0.784 | **−0.041** | [−0.083, +0.003] |

😊 is **2.4x more effective than `#`** at identical tokenization, and 🖕 does essentially nothing.
The effects order by *valence* — positive > neutral > negative — not by segmentation. The same
dissociation holds at token-level granularity (`#` −0.213, 😊 −0.496, 🖕 −0.078, all at 2.89x
inflation).

This directly contradicts the paper's claim that open-source judges "have a strong token
segmentation bias, while being less influenced by the specific semantic meaning of the emojis."
On Qwen3Guard-Gen-0.6B, semantics dominate.

## The false-positive control finds new results

We added a benign
arm into the paper.

| condition | harmful P(unsafe) | benign P(unsafe) | benign FPR (strict) |
|---|---|---|---|
| *clean* | 0.824 | 0.001 | 0.000 |
| `" "` | 0.494 ↓ | 0.014 | 0.000 |
| `"#"` | 0.644 ↓ | 0.014 | 0.000 |
| `"😊"` | 0.394 ↓ | 0.054 | 0.080 |
| `"🖕"` | 0.784 | 0.215 ↑ | **0.360** |

In the last row, 🖕 looks *protective* on the harmful arm (detection barely drops, strict-mode
detection actually rises to 0.993). The paper would score that as the guardrail resisting the
attack. The benign arm shows what is really happening: **🖕 pushes 36% of entirely benign text
into "unsafe"** (41% at token granularity). The guardrail did not get better at detecting harm, but rather it became indiscriminately suspicious of anything containing a middle finger.

By contrast `" "` and `"#"` lower harmful detection substantially while leaving benign FPR at
**0.000**.

## Method in brief

**Threat model.** Post-hoc output moderation. The attacker controls only the response string and
wants the guardrail to score it safe. This is the paper's Table 4 setting (direct string
transform), not Table 2 (in-context insertion via a target LLM).

**Conditions.** 9, fully paired on the same items: `clean`, plus 4 delimiters × 2 granularities.
- `word_*` — split each word at the midpoint of its alphanumeric run. This is `split_words()`
  upstream, i.e. the paper's **Figure 3** experiment.
- `tok_*` — split inside every multi-character token of the judge's *own* tokenizer. This is
  `InsertDelimiterRandomly()` upstream, i.e. the paper's **Table 4** white-box attack.

`"#"` is our addition and it is what makes the design work: without a common, valence-free,
single-token control, space-vs-emoji confounds valence with rare-token-ness and the question is
unanswerable.

**Measurement.** One forward pass, no generation. Qwen3Guard's template already ends with the
assistant prefix, so we append the literal `"Safety:"` and read the next-token distribution over
` Safe` (22291) / ` Unsafe` (73067) / ` Cont` (2093, first token of ` Controversial`), softmaxed
over those three ids. Consequence: **parse-failure rate is structurally 0% in every condition**,
so the classic bias — attack conditions produce more unparseable output, those rows get dropped,
survivors look clean — cannot occur. Both strict (controversial→unsafe) and loose
(controversial→safe) conventions are reported.

**Data.** n=150 harmful (AdvBench `harmful_strings`, 3–44 words, seed 20250813); n=75 benign, in
two arms — `benign_sent` (first sentence, median 19 words) and `benign_full` (median 71 words).
Benign numbers above are `benign_sent`. The paper has no benign arm at all, and its
`safe_responses.json` was never judged — it is Mistral-7B output capped at 100 tokens, used only
as in-context demos, which is why it needed length-matching.

**The user turn.** AdvBench strings have no paired prompt, so we pass an empty USER turn. The
paper hit the same problem and handled it less carefully: its Llama Guard template deletes the
User line, and its WildGuard call passes `prompt=None`, so that judge literally reads
`Human user:\nNone`.

## Reproducing

```bash
uv sync                       # Python 3.11.11
git clone --depth 1 https://github.com/zhipeng-wei/EmojiAttack.git work/EmojiAttack
git clone --depth 1 https://github.com/llm-attacks/llm-attacks.git work/llm-attacks
python prep_data.py
python sanity.py  --model Qwen/Qwen3Guard-Gen-0.6B    # run first — check 1 is the gate
python score.py   --model Qwen/Qwen3Guard-Gen-0.6B --out out/scores_0p6b.jsonl
python analyze.py --scores out/scores_0p6b.jsonl --tag _0p6b
```

## Limitations

- **0.6B only so far.** The headline result may not hold at 4B/8B; robustness is not known to
  scale monotonically within guardrail families. 
- **AdvBench `harmful_strings` are short and blunt** (3–32 words as sampled), unlike realistic
  jailbreak output.
- **No length control for the transform itself.** Every attack condition roughly doubles or
  triples token count, so "more tokens" is not yet separated from "split words". The clean test
  is a between-words insertion condition at matched inflation — not yet run, and the single
  biggest gap here.
- `p_controversial` is a first-token proxy (` Cont`).
- The empty user turn is not a realistic conversation.
- n=150 / 75, single language, single guardrail family.

## Ethics

Harmful strings are **not committed**, nor are raw scoring records (`out/*.jsonl`), which contain
them verbatim. The seed and filters in `prep_data.py` make the item set reproducible from the
public upstream sources. Committed artifacts are aggregate only (`out/results*.csv`,
`out/figure*.png`). This is defensive evaluation of a published attack against a publicly
released guardrail.
