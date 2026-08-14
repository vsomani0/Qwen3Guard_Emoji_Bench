"""The attack primitive: insert a delimiter inside words / inside tokens.

Two granularities, because the paper uses two and they are not the same thing:

  word-midpoint  -- `split_words()` in the upstream repo's `offensive_words_judge.py`,
                    the experiment behind the paper's Figure 3 ("mid-split").
  token-level    -- `InsertDelimiterRandomly()` in the upstream repo's `Whitebox_Emojis.py`,
                    the experiment behind the paper's Table 4 rows 2-3. Requires the
                    judge's own tokenizer, so this is a genuine white-box condition.
"""

import random
import re

# Delimiters. "#" is our addition: a common, single-token, valence-free symbol.
# Without it, `" "` vs `"\N{SMILING FACE WITH SMILING EYES}"` confounds emoji valence
# with mere rare-token-ness, and the whole tokenization-vs-semantics question is unanswerable.
DELIMITERS = {
    "space": " ",
    "hash": "#",
    "smile": "\N{SMILING FACE WITH SMILING EYES}",  # U+1F60A, positive valence
    "finger": "\N{REVERSED HAND WITH MIDDLE FINGER EXTENDED}",  # U+1F595, negative valence
}

CONDITIONS = ["clean"] + [
    f"{g}_{d}" for g in ("word", "tok") for d in DELIMITERS
]


# --------------------------------------------------------------------------- word-midpoint

def insert_word(word, ch):
    """Insert `ch` at the midpoint of the leading alphanumeric run of `word`.

    Two documented deviations from upstream `split_words()`, which the paper does not pin down:
      * we skip runs shorter than 3 chars (upstream splits any run of length >= 2, so
        a 2-letter word becomes two 1-char fragments -- more mangling than the paper describes)
      * we split the alphanumeric run, not the raw string, so trailing punctuation
        ("garbage," ) does not drag the midpoint rightward.
    """
    core = re.match(r"^\W*(\w+)", word)
    if not core or len(core.group(1)) < 3:
        return word
    s, e = core.span(1)
    j = s + (e - s) // 2
    return word[:j] + ch + word[j:]


def transform_word(text, ch):
    if ch is None:
        return text
    return " ".join(insert_word(w, ch) for w in text.split())


# --------------------------------------------------------------------------- token-level

def transform_token(text, ch, tokenizer):
    """Insert `ch` at a pseudo-random position inside every multi-character token.

    Mirrors upstream `InsertDelimiterRandomly`, including its `random.seed(i)` per token
    index (so the choice is deterministic given position, and identical across delimiters --
    which is what makes the space/#/emoji comparison paired at the *insertion-site* level,
    not just at the item level).

    Deviation from upstream: upstream was written for Llama's SentencePiece, where the
    word-leading space is a separate "N{LOWER ONE EIGHTH BLOCK}" marker. Qwen's BPE decodes
    a leading space as an actual " " inside the token string, so we split the leading
    whitespace off and insert only within the non-space core. Without this, roughly one
    insertion in five would land between the space and the word rather than inside it,
    which is not the attack the paper describes.
    """
    if ch is None:
        return text
    out = []
    for i, tid in enumerate(tokenizer.encode(text, add_special_tokens=False)):
        piece = tokenizer.decode([tid])
        lead = len(piece) - len(piece.lstrip())
        head, core = piece[:lead], piece[lead:]
        if len(core) <= 1:
            out.append(piece)
            continue
        random.seed(i)
        p = random.randint(1, len(core) - 1)
        out.append(head + core[:p] + ch + core[p:])
    return "".join(out)


def apply_condition(text, condition, tokenizer):
    if condition == "clean":
        return text
    gran, dname = condition.split("_", 1)
    ch = DELIMITERS[dname]
    return transform_word(text, ch) if gran == "word" else transform_token(text, ch, tokenizer)


# --------------------------------------------------------------------------- attack utility

def strip_delimiter(text, ch):
    """Undo the transform. The paper reports no attack-utility metric anywhere, so this
    is ours: does the harmful payload survive the transform in recoverable form?

    For a single-token delimiter (#, emoji) removal is exact and lossless.
    For " " it is NOT -- real word boundaries are destroyed and become indistinguishable
    from inserted ones, so the original cannot be recovered by stripping. That asymmetry
    means the space-split condition and the emoji conditions are not equally usable
    attacks even where they are equally effective at evading the judge.
    """
    return text.replace(ch, "")
