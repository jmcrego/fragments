import sys
import time
import difflib
import argparse
from tqdm import tqdm
from collections import defaultdict
from tokenizers.pre_tokenizers import Whitespace


def count_lines(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)  # Lazy iteration


class splitPunctuation():

    def __init__(self):
        self.pretok = Whitespace()

    def __call__(self, text):
        return self.pretok.pre_tokenize_str(text)


def get_overlapping_spans(
        input_tokens,
        source_tokens,
        min_len=1,
        lc=True,
        filter_contained=True,
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

    if filter_contained:
        # filter out spans that are contained within other spans (use actual token strings, not positions)
        spans_filtered = []
        for span in sorted(spans, key=lambda x: x[1] - x[0], reverse=True): #larger to smaller
            span_tokens = stoks[span[0]:span[1]] # the tokens corresponding to the span positions (i.e. "['the', 'day']")
            print(f"Checking span {span} with tokens {span_tokens}")
            if not any(span_tokens in stoks[s[0]:s[1]] for s in spans_filtered):
                spans_filtered.append(span)
                print(f"Added")
            else:
                print(f"Filtered out")
        return spans_filtered

    return spans



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run inference of EuroLLM models using vLLM.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", type=str, required=True, help="Input file (tokenized).")
    parser.add_argument("-o", type=str, required=True, help="Output file (tokenized).")
    parser.add_argument("-s", type=str, required=True, help="Source file (tokenized).")
    parser.add_argument("-t", type=str, required=True, help="Target file (tokenized).")
    parser.add_argument("-min_score", type=float, default=0.5, help="Minimum matching score.")
    parser.add_argument("-stop_at", type=int, default=0, help="Stop when already generated that many spans.")
    parser.add_argument("-lc", action='store_true', help="Use lowercase to match input/source strings.")
    parser.add_argument("-detokenize", action='store_true', help="Report untokenized (raw) strings.")
    args = parser.parse_args()    
    tic = time.time()

    sp = splitPunctuation()

    # text = "Hello, world! How's everything going?"
    # print(f"Original text: {text}")

    # tokens_with_offsets = sp(text)

    # for token, (start, end) in tokens_with_offsets:
    #     print(f"({start}, {end}) -> {token}")

    # tokens = [token for token, _ in tokens_with_offsets]
    # print(f"Split tokens: {tokens}")


    n_output = 0
    with open(args.i) as fi, open(args.o) as fo, open(args.s) as fs, open(args.t) as ft:
        idx = 0
        for i, o, s, t in tqdm(zip(fi, fo, fs, ft), total=count_lines(args.i), unit=" samples", desc="Samples"):
            i_with_offsets = sp(i.strip())
            o_with_offsets = sp(o.strip())
            s_with_offsets = sp(s.strip())
            t_with_offsets = sp(t.strip())
            if len(i_with_offsets) and len(o_with_offsets) and len(s_with_offsets) and len(t_with_offsets):
                i_tokens = [token for token, _ in i_with_offsets]
                o_tokens = [token for token, _ in o_with_offsets]
                s_tokens = [token for token, _ in s_with_offsets]
                t_tokens = [token for token, _ in t_with_offsets]
                source_spans = get_overlapping_spans(i_tokens, s_tokens, lc=args.lc)
                if len(source_spans):
                    print(f"I {idx}\t{' '.join(i_tokens)}")
                    print(f"S {idx}\t{' '.join(s_tokens)}")
                    print(f"M {idx}\t{source_spans}")
                    print(f"T {idx}\t{' '.join(t_tokens)}") 
                    print(f"O {idx}\t{' '.join(o_tokens)}")
                    n_output += 1
            idx += 1
            if args.stop_at and n_output >= args.stop_at:
                break
    sys.stderr.write(f"Done! output {n_output} out of {idx} samples\n")