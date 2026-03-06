import difflib
import string
import json
import sys
from collections import defaultdict

prompt = """
You are given:

1) An input sentence to be translated.

2) A translation example consisting of:
   - Source sentence (example source)
   - Target sentence (its translation)

3) A list of overlapping spans that appear in BOTH:
   - the input sentence
   - the example source sentence

Your task has two steps.

--------------------------------
Step 1 — Extract alignments
--------------------------------

For each overlapping span, identify the corresponding translation span in the example target sentence.

Output the alignments using the format:

<ALIGNS>
source span === target span
source span === target span
</ALIGNS>

Rules:
- Each source span MUST come from the list "Overlapping spans".
- Each target span MUST be a contiguous substring taken from the example target sentence.
- Prefer the translation span that best matches the semantic unit of the source span.

--------------------------------
Step 2 — Refine useful spans
--------------------------------

Evaluate the extracted alignments and produce a refined list of spans that could help translate the input sentence.

Output the refined spans using the format:

<SPANS>
source span ||| target span
source span ||| target span
</SPANS>

Rules:
- Discard spans that contain only function words or punctuation  (e.g., "of the", "in the").
- Discard pairs where the alignment is incorrect or unclear.
- Adapt the target span so that it would fit naturally when translating the input sentence (adjust morphology, determiners, or phrasing if needed).

Stop generating immediately after </SPANS>.

--------------------------------
Input
--------------------------------

Input sentence:
{input}

Example source:
{source}

Example target:
{target}

Overlapping spans:
{spans}

"""

def get_formatted_prompt(sample):
    return prompt.format(
        input=sample["input"],
        source=sample["source"],
        target=sample["target"],
    	spans='\n'.join(sample["spans"])
    )

def get_matching_spans(
    input_tokens, 
    source_tokens, 
    lc=False
):
    """                                                                                                                                                                                                                                                                                            
    Returns a list of tuples (s_beg, s_end)
    source_tokens[s_beg:s_end] esists in input_tokens
    and the span is contiguous and maximal.                                                                                                                                                                                                                                                        
    """
    itoks = input_tokens if not lc else [x.lower() for x in input_tokens]
    stoks = source_tokens if not lc else [x.lower() for x in source_tokens]
    matcher = difflib.SequenceMatcher(None, stoks, itoks)
    spans = []
    for tag, s_beg, s_end, i_beg, i_end in matcher.get_opcodes():
        if tag == 'equal':
            spans.append( (s_beg, s_end) )
    return spans

def get_overlapping_spans(
        idx,
        a, 
        b, 
        min_len=1,
        lc=True,
        filter_contained=True,
        filter_contained_strings=True,
        sort_spans=True,
):
    if lc:
        """lowercase tokens in lists."""
        a = [x.lower() for x in a]
        b = [x.lower() for x in b]
        
    def extract_ngrams_with_positions(tokens, min_len, max_len):
        spans = defaultdict(list)
        for n in range(min_len, max_len + 1):
            for i in range(len(tokens) - n + 1):
                span = tuple(tokens[i:i + n])
                spans[span].append(i)
        return spans #('the', 'day') ==> [3, 6, ...] (position/s where the span start/s)

    """find overlapping spans between lists a and b."""
    max_len = min(len(a),len(b))
    a_spans = extract_ngrams_with_positions(a, min_len, max_len)
    b_spans = extract_ngrams_with_positions(b, min_len, max_len)
    common_spans = set(a_spans.keys()) & set(b_spans.keys()) #intersection of keys
    print(f"idx: {idx} common_spans: {common_spans}")
    
    spans = []
    for span in common_spans:
        for i in a_spans[span]:
            spans.append((i, i + len(span))) 
            print(f"idx: {idx} span: {span} {spans[-1]}")

    if filter_contained:
        """Remove spans in a that are entirely contained within other spans in a."""
        filtered = []
        for i, m1 in enumerate(spans):
            a1_start, a1_end = m1
            a1_string = ' '.join(a[a1_start:a1_end])
            is_subseq = False

            for j, m2 in enumerate(spans):
                if i == j:
                    continue
                a2_start, a2_end = m2
                a2_string = ' '.join(a[a2_start:a2_end])
                if a2_start <= a1_start and a1_end <= a2_end:
                    is_subseq = True
                    break
                if filter_contained_strings and " "+a1_string+" " in " "+a2_string+" " and (a1_string != a2_string or a1_start < a2_start):
                    print(f"idx: {idx} a1_string: [{a1_start},{a1_end}) '{a1_string}' contained in a2_string: [{a2_start},{a2_end}) '{a2_string}'")
                    is_subseq = True
                    break
                    
            if not is_subseq:
                filtered.append(m1)
        spans = filtered

        
    if sort_spans:
        """Sort matches according to first position, in ascending order"""
        spans = sorted(spans, key=lambda x: x[0])  # x[0] = s_beg (ascending order)
    return spans


def detok(
    l
):
    if isinstance(l, list):
        l = ' '.join(l)
    return l.replace(' ￭', '').replace('￭ ', '').replace('￭', '')


def read_josep_file(
    TEST_FILE,
    min_span_len = 2
):
    with open(TEST_FILE, 'r') as f:
        for l in f:
            toks = l.strip().split('\t')
            if len(toks) != 2:
                sys.stderr.write(l)
                continue
            if   l.startswith("S "):
                _, source = toks
            elif l.startswith("T "):
                _, target = toks
            elif l.startswith("I "):
                _, input = toks
            elif l.startswith("O "):
                tag, output = toks

                idx = int(tag.split(' ')[1])
                ### yield sample
                print(f"idx: {idx} source.split: {source.split()} input.split: {input.split()}")
                source_spans = get_overlapping_spans(idx,source.split(), input.split())
                print(f"idx: {idx} source_spans: {source_spans}")
                if len(source_spans) == 0:
                    print(f"no spans available for idx={idx}")
                    continue
                max_span_len = max([span[1]-span[0] for span in source_spans])
                if max_span_len < min_span_len:
                    print(f"max_span_len is not enought for idx={idx} source_spans={source_spans}")
                    continue
                source_tokens = source.split()
                source_spans = [detok(' '.join(source_tokens[span[0]:span[1]])) for span in source_spans]
                yield {"idx": idx, "input": detok(input), "output": detok(output), "source": detok(source), "target": detok(target), "spans": source_spans}


