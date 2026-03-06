import re
from difflib import SequenceMatcher


# -----------------------------
# Tokenization
# -----------------------------
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


# -----------------------------
# LCS Alignment
# -----------------------------
def lcs_alignment(tokens_a, tokens_b):
    matcher = SequenceMatcher(None, tokens_a, tokens_b)
    blocks = matcher.get_matching_blocks()

    alignment = []
    for block in blocks:
        if block.size == 0:
            continue
        for k in range(block.size):
            alignment.append((block.a + k, block.b + k))
    return alignment


# -----------------------------
# Build maximal contiguous spans
# -----------------------------
def build_maximal_spans(alignment):

    if not alignment:
        return []

    spans = []
    current = [alignment[0]]

    for prev, cur in zip(alignment, alignment[1:]):

        # contiguous in both sentences
        if cur[0] == prev[0] + 1 and cur[1] == prev[1] + 1:
            current.append(cur)
        else:
            spans.append(current)
            current = [cur]

    spans.append(current)
    return spans


def spans_to_units(spans, source_tokens):

    units = []

    for span in spans:

        source_indices = [s[1] for s in span]

        start = min(source_indices)
        end = max(source_indices) + 1

        units.append({
            "tokens": source_tokens[start:end],
            "indices": list(range(start, end))
        })

    return units


# -----------------------------
# Expand spans with gappy units
# -----------------------------
def build_gappy_units(units, max_gap=4):
    """
    Merge non-adjacent units into gappy spans.

    If two spans are close enough (gap <= max_gap),
    merge them into a single gappy unit.
    """

    if not units:
        return []

    units = sorted(units, key=lambda x: x["indices"][0])

    gappy = []
    current = units[0]

    for next_unit in units[1:]:

        gap = next_unit["indices"][0] - current["indices"][-1] - 1

        if 0 < gap <= max_gap:
            # Merge with gap
            current["tokens"] += ["..."] + next_unit["tokens"]
            current["indices"] += next_unit["indices"]
        else:
            gappy.append(current)
            current = next_unit

    gappy.append(current)

    return gappy

# -----------------------------
# Remove contained spans
# -----------------------------
def remove_contained(units):

    units = sorted(units, key=lambda x: len(x["tokens"]), reverse=True)

    filtered = []

    for u in units:

        idx_string = " ".join(map(str, u["indices"]))

        if not any(
            idx_string in " ".join(map(str, f["indices"]))
            for f in filtered
        ):
            filtered.append(u)

    return filtered


# -----------------------------
# Format Output
# -----------------------------
def format_units(units):

    lines = []

    for u in units:

        parts = []
        for token, idx in zip(u["tokens"], u["indices"]):
            parts.append(f"{idx}:{token}")

        lines.append(" ".join(parts))

    return lines


# -----------------------------
# Main Function
# -----------------------------
def extract_source_units(input_sentence, source_sentence):

    input_tokens = tokenize(input_sentence)
    source_tokens = tokenize(source_sentence)

    alignment = lcs_alignment(input_tokens, source_tokens)

    contiguous = build_maximal_spans(alignment)

    units = spans_to_units(contiguous, source_tokens)

    units = build_gappy_units(units)

    units = remove_contained(units)

    return format_units(units)


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":

    input_sentence = "Can you give me the money back now?"
    source_sentence = "Could you give me my toy back now please?"

    units = extract_source_units(input_sentence, source_sentence)

    print("\nSource units with indices:\n")
    for u in units:
        print(u)