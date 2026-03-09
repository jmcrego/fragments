import re
import json
import argparse
from tokenizers.pre_tokenizers import Whitespace
from difflib import SequenceMatcher

class splitPunctuation():
    def __init__(self):
        self.pretok = Whitespace()

    def __call__(self, text):
        return self.pretok.pre_tokenize_str(text)


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
# Build gappy units
# -----------------------------
def build_gappy_units(units, max_gap=6):

    if not units:
        return []

    units = sorted(units, key=lambda x: x["indices"][0])

    merged = []
    current = units[0]

    for nxt in units[1:]:

        gap = nxt["indices"][0] - current["indices"][-1] - 1

        if 0 <= gap <= max_gap:

            current["tokens"] += nxt["tokens"]
            current["indices"] += nxt["indices"]

        else:
            merged.append(current)
            current = nxt

    merged.append(current)

    return merged

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
# Remove duplicate units
# -----------------------------
def remove_duplicate(units):

    seen = set()
    unique = []

    for u in units:
        key = tuple(u["indices"])
        if key not in seen:
            seen.add(key)
            unique.append(u)

    return unique

# -----------------------------
# Format Output
# -----------------------------
def format_units(units):
    lines = []

    for u in units:

        parts = [
#            f"{idx}:{tok}" for idx, tok in zip(u["indices"], u["tokens"])
            f"{{tok}" for idx, tok in zip(u["indices"], u["tokens"])
        ]

        lines.append(" ".join(parts))

    return lines

def format_units_gaps(units, gap_str="..."):

    lines = []

    for u in units:

        parts = []
        indices = u["indices"]
        tokens = u["tokens"]

        if not indices:
            continue

        # Start with first token
        parts.append(f"{indices[0]}:{tokens[0]}")

        for prev_idx, curr_idx, token in zip(indices, indices[1:], tokens[1:]):

            gap = curr_idx - prev_idx

            # If gap > 1 → we skipped tokens
            if gap > 1:
                parts.append(gap_str)

            parts.append(f"{curr_idx}:{token}")

        lines.append(" ".join(parts))

    return lines


# -----------------------------
# Main Function
# -----------------------------

def get_spans_from_files(input_file, source_file, target_file, output_file, **kwargs):
    min_tok_len = kwargs.get("min_tok_len", 1)
    min_str_len = kwargs.get("min_str_len", 3)
    max_gap = kwargs.get("max_gap", 3)
    gap_str = kwargs.get("gap_str", "")
    sp = splitPunctuation()
    with open(input_file, encoding="utf-8") as fi, open(output_file, encoding="utf-8") as fo, open(source_file, encoding="utf-8") as fs, open(target_file, encoding="utf-8") as ft:
        for idx, (i, o, s, t) in enumerate(zip(fi, fo, fs, ft)):
            i_with_offsets = sp(i.strip())
            s_with_offsets = sp(s.strip())
            i_tokens = [token for token, _ in i_with_offsets]
            s_tokens = [token for token, _ in s_with_offsets]
            # i_tokens = tokenize(i.strip())
            # s_tokens = tokenize(s.strip())
            if len(i_tokens) and len(s_tokens) and len(o.strip()) and len(t.strip()):
                i_tokens_lc = [x.lower() for x in i_tokens]
                s_tokens_lc = [x.lower() for x in s_tokens]
                alignment = lcs_alignment(i_tokens_lc, s_tokens_lc)
                contiguous = build_maximal_spans(alignment)
                units = spans_to_units(contiguous, s_tokens)
                units = build_gappy_units(units)
                units = remove_contained(units)
                units = remove_duplicate(units)
                if not gap_str:
                    units = format_units(units)
                else:
                    units = format_units_gaps(units, gap_str=gap_str)
                if len(units):
                    yield {
                        "idx": idx,
                        "input": i.strip(),
                        "source": s.strip(),
                        "target": t.strip(),
                        "output": o.strip(),
                        "spans": [unit.strip() for unit in units]
                    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run inference of EuroLLM models using vLLM.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", type=str, required=True, help="Input file (tokenized).")
    parser.add_argument("-o", type=str, required=True, help="Output file (tokenized).")
    parser.add_argument("-s", type=str, required=True, help="Source file (tokenized).")
    parser.add_argument("-t", type=str, required=True, help="Target file (tokenized).")
    parser.add_argument("-min_tok_len", type=int, default=1, help="Minimum number of tokens in a span.")
    parser.add_argument("-min_str_len", type=int, default=3, help="Minimum number of characters in a span.")
    parser.add_argument("-max_gap", type=int, default=6, help="Maximum gap size for merging units into gappy units.")
    parser.add_argument("-gap_str", type=str, default="...", help="String to use for representing gaps in the output.")
    parser.add_argument("-stop_at", type=int, default=0, help="Stop when already generated that many spans.")
    args = parser.parse_args()    

    for idx, sample in enumerate(get_spans_from_files(args.i, args.s, args.t, args.o, min_tok_len=args.min_tok_len, min_str_len=args.min_str_len, max_gap=args.max_gap, gap_str=args.gap_str)):
        # print(f"=== Sample {idx} =============================")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        if args.stop_at and idx >= args.stop_at:
            break

