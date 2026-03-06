import sys
import time
import argparse
from collections import defaultdict
from tokenizers.pre_tokenizers import Whitespace


class splitPunctuation():

    def __init__(self):
        self.pretok = Whitespace()

    def __call__(self, text):
        return self.pretok.pre_tokenize_str(text)


def get_overlapping_spans(
        input_tokens,
        source_tokens,
        min_len=1,
        min_string_len=3,
        lc=True,
):
    """find overlapping spans between lists input_tokens and source_tokens."""
    itoks = input_tokens if not lc else [x.lower() for x in input_tokens]
    stoks = source_tokens if not lc else [x.lower() for x in source_tokens]
        
    def extract_ngrams_with_position(tokens, min_len, max_len):
        spans = defaultdict(int)
        for n in range(min_len, max_len + 1):
            for i in range(len(tokens) - n + 1):
                span = tuple(tokens[i:i + n])
                spans[span] = i #('the', 'day') ==> 3 (position where the span start, only the last occurrence is saved)
        return spans 

    max_len = min(len(stoks),len(itoks))
    i_spans = extract_ngrams_with_position(itoks, min_len, max_len)
    s_spans = extract_ngrams_with_position(stoks, min_len, max_len)
    common_spans = set(s_spans.keys()) & set(i_spans.keys()) #intersection of keys

    spans = []
    for span in common_spans:
        i = s_spans[span] #source_spans
        spans.append((i, i+len(span)))

    # filter out spans that are contained within other spans (use actual token strings, not positions)
    spans_filtered_strings = []
    for span in sorted(spans, key=lambda x: x[1] - x[0], reverse=True): #larger to smaller
        span_string = " "+' '.join(stoks[span[0]:span[1]])+" " # the string corresponding to the span positions (i.e. ' the day ')
        if len(span_string)-2 < min_string_len:
            continue
        # check if any of the already added spans contains the current span tokens
        if not any(span_string in s for s in spans_filtered_strings):
            spans_filtered_strings.append(span_string)
    return spans_filtered_strings



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run inference of EuroLLM models using vLLM.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", type=str, required=True, help="Input file (tokenized).")
    parser.add_argument("-o", type=str, required=True, help="Output file (tokenized).")
    parser.add_argument("-s", type=str, required=True, help="Source file (tokenized).")
    parser.add_argument("-t", type=str, required=True, help="Target file (tokenized).")
    parser.add_argument("-stop_at", type=int, default=0, help="Stop when already generated that many spans.")
    args = parser.parse_args()    
    tic = time.time()

    sp = splitPunctuation()

    n_output = 0
    with open(args.i) as fi, open(args.o) as fo, open(args.s) as fs, open(args.t) as ft:
        idx = 0
        for i, o, s, t in zip(fi, fo, fs, ft):
            i_with_offsets = sp(i.strip())
            o_with_offsets = sp(o.strip())
            s_with_offsets = sp(s.strip())
            t_with_offsets = sp(t.strip())
            if len(i_with_offsets) and len(o_with_offsets) and len(s_with_offsets) and len(t_with_offsets):
                i_tokens = [token for token, _ in i_with_offsets]
                o_tokens = [token for token, _ in o_with_offsets]
                s_tokens = [token for token, _ in s_with_offsets]
                t_tokens = [token for token, _ in t_with_offsets]
                source_spans = get_overlapping_spans(i_tokens, s_tokens)
                if len(source_spans):
                    # print(f"I {idx}\t{' '.join(i_tokens)}")
                    # print(f"S {idx}\t{' '.join(s_tokens)}")
                    # print(f"T {idx}\t{' '.join(t_tokens)}") 
                    # print(f"O {idx}\t{' '.join(o_tokens)}")
                    print(f"I {idx}\t{i.strip()}")
                    print(f"S {idx}\t{s.strip()}")
                    print(f"M {idx}\t{[span.strip() for span in source_spans]}")
                    print(f"T {idx}\t{t.strip()}") 
                    print(f"O {idx}\t{o.strip()}")
                    n_output += 1
            idx += 1
            if args.stop_at and n_output >= args.stop_at:
                break
    sys.stderr.write(f"Done! output {n_output} out of {idx} samples\n")