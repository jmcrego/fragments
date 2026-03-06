import sys
import json
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
        min_tok_len=1,
        min_str_len=3,
        lc=True,
):
    """find overlapping spans between lists input_tokens and source_tokens."""        
    def extract_ngrams_with_position(tokens, min_len, max_len):
        # extract all n-grams of length between [min_len, max_len] (both included), 
        # and save their starting position in the tokens list (the last occurrence is saved if multiple exist)
        spans = defaultdict(int) # {['the', 'day'] -> 5} means that the span 'the day' starts at position 5 in the tokens list
        for n in range(min_len, max_len + 1):
            # spans of length n
            for i in range(len(tokens) - n + 1):
                span = tuple(tokens[i:i + n])
                # save the starting position of the span in the tokens list (the last occurrence is saved if multiple exist)
                spans[span] = i 
        return spans 

    itoks = input_tokens if not lc else [x.lower() for x in input_tokens] #['the', 'day', 'is', 'sunny']
    stoks = source_tokens if not lc else [x.lower() for x in source_tokens] #['the', 'day', 'is', 'rainy']
    max_tok_len = min(len(stoks),len(itoks))
    i_spans = extract_ngrams_with_position(itoks, min_tok_len, max_tok_len) #{('the', 'day', 'is'): 0, ...}
    s_spans = extract_ngrams_with_position(stoks, min_tok_len, max_tok_len) #{('the', 'day', 'is'): 0, ...}
    common_spans = set(s_spans.keys()) & set(i_spans.keys()) #('the', 'day', 'is')

    spans = []
    for span in common_spans: #('the', 'day', 'is')
        if len(span) >= min_str_len:
            i = s_spans[span] #0
            spans.append((i, i+len(span))) #[(0, 3)] meaning that the span 'the day is' starts at position 0 and ends at position 3 in the source tokens list

    # filter out spans that are contained within other spans (use token strings, not positions)
    spans_filtered_strings = []
    for span in sorted(spans, key=lambda x: x[1] - x[0], reverse=True): #larger to smaller
        span_str = " "+' '.join(source_tokens[span[0]:span[1]])+" " # the string corresponding to the span positions (i.e. ' the day is ')
        # check if any of the already added spans contains the current span tokens
        if not any(span_str in s for s in spans_filtered_strings):
            spans_filtered_strings.append(span_str)
    return spans_filtered_strings


def get_spans_from_files(input_file, source_file, target_file, output_file, min_tok_len=1, min_str_len=3):
    sp = splitPunctuation()
    with open(input_file) as fi, open(output_file) as fo, open(source_file) as fs, open(target_file) as ft:
        for idx, (i, o, s, t) in enumerate(zip(fi, fo, fs, ft)):
            i_with_offsets = sp(i.strip())
            s_with_offsets = sp(s.strip())
            if len(i_with_offsets) and len(s_with_offsets) and len(o.strip()) and len(t.strip()):
                i_tokens = [token for token, _ in i_with_offsets]
                s_tokens = [token for token, _ in s_with_offsets]
                source_spans = get_overlapping_spans(i_tokens, s_tokens, min_tok_len=min_tok_len, min_str_len=min_str_len)
                if len(source_spans):
                    yield {
                        "idx": idx,
                        "input": i.strip(),
                        "source": s.strip(),
                        "target": t.strip(),
                        "output": o.strip(),
                        "spans": [span.strip() for span in source_spans]
                    }
                else:
                    print(f"Warning: No spans found for sample {idx} (input and source have no common spans of at least {min_tok_len} tokens and {min_str_len} characters).", file=sys.stderr)
            else:
                print(f"Warning: No spans found for sample {idx} (any of input/output/source/target is empty after tokenization).", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run inference of EuroLLM models using vLLM.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", type=str, required=True, help="Input file (tokenized).")
    parser.add_argument("-o", type=str, required=True, help="Output file (tokenized).")
    parser.add_argument("-s", type=str, required=True, help="Source file (tokenized).")
    parser.add_argument("-t", type=str, required=True, help="Target file (tokenized).")
    parser.add_argument("-min_tok_len", type=int, default=1, help="Minimum number of tokens in a span.")
    parser.add_argument("-min_str_len", type=int, default=3, help="Minimum number of characters in a span.")
    parser.add_argument("-stop_at", type=int, default=0, help="Stop when already generated that many spans.")
    args = parser.parse_args()    

    for idx, sample in enumerate(get_spans_from_files(args.i, args.s, args.t, args.o, min_tok_len=args.min_tok_len, min_str_len=args.min_str_len)):
        # print(f"=== Sample {idx} =============================")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        if args.stop_at and idx >= args.stop_at:
            break