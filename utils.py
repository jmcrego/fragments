import difflib
import string
import json
import sys
from collections import defaultdict

MIN_SPAN_LEN_IN_CHARS = 7

prompt = """You are given:
- An input sentence to be translated.
- A related translation example, consisting of:
- A source sentence (which may share content with the input sentence).
- Its corresponding target translation.
- A list of overlapping spans between the input sentence and the source sentence.

Your task is to align each overlapping span (taken from the example source sentence) to its corresponding translation in the example target sentence.

Output Format:

Wrap the output in <ALIGNS> and </ALIGNS> tags without quotes. List one alignment per line using the format:

<ALIGNS>
source align1 === target align1
source align2 === target align2
</ALIGNS>

Each *source align* should be from the list of *Overlapping spans* while *target align* should be extracted from *Target*.

Then, judge the quality of the extracted aligns, you are allowed to:
- Discard uninformative spans containing only stop words ("of the") or punctuation.
- Discard failed pairs (spans without translations or not possible to align).
- Rephrase target aligns to adapt them to the morphology, context and phrasing of the required translation of the input sentence (trim/append words, update morphology).

Wrap the new list of refined aligns in <SPANS> and </SPANS> tags without quotes. List one sample per line using the format:

<SPANS>
source span1 ||| target span1
source span2 ||| target span2
</SPANS>

Input:
{input}

Source:
{source}

Target:
{target}

Overlapping spans:
{spans}

⚠️ Stop immediately after </SPANS>.
</think>"""

prompt_old2 = """You are given:
 
- An input sentence to be translated.
- A related translation example, consisting of:
- A source sentence (which may share content with the input sentence).
- Its corresponding target translation.
- A list of overlapping spans between the input sentence and the source sentence.
 
Your task is to align each overlapping span (taken from the example source sentence) to its corresponding translation in the example target sentence.
 
Apply the following rules:
- Discard uninformative spans containing only stop words ("of the"), punctuation and/or easy vocabulary ("dog").
- Discard spans without translations — if the corresponding translation is not present in the target sentence, omit the alignment.
- Do not repeat identical spans.
 
Output Format:
 
Wrap the output in <SPANS> and </SPANS> tags without quotes. List one alignment per line using the format:
 
<SPANS>
source span1 ||| target span1
source span2 ||| target span2
</SPANS>
 
Each *source span* should be from the list of *Overlapping spans* while *target span* should be extracted from *Target*.
 
Input:
{input}
 
Source:
{source}
 
Target:
{target}
 
Overlapping spans:
{spans}
 
Then, judge the quality of the extracted fragments, and discard failed pairs (cases of impossibility to extract fragments) or rephrase the fragments to adapt them to the morphology, context and phrasing of the input sentence (trim or/and prepend/append elements, change morphology like gender/plural/cases). Put the new list of fragments between the <SPANS> </SPANS> markers with the same format.

⚠️ Stop immedialty after </SPANS>.
</think>"""
 

prompt_old = """You are given:

- An input sentence to be translated.
- A related translation example, consisting of:
- A source sentence (which shares content with the input sentence).
- Its corresponding target translation.
- A list of overlapping spans between the input sentence and the source sentence.

Your task is to align each overlapping span (taken from the example source sentence) to its corresponding translation in the example target sentence.

Apply the following rules:
- Discard uninformative spans — exclude any source span that does not contain at least one noun, verb, adjective, or adverb.
- Discard spans without translations — if the corresponding translation is not present in the target sentence, omit the alignment.
- Discard context-irrelevant alignments — exclude any alignment where the translation, even if accurate, is not useful for translating the input sentence due to contextual or domain mismatch.

Output Format:

Wrap the output in <SPANS> and </SPANS> tags. List one alignment per line using the format:

<SPANS>
source span1 ||| target span1
source span2 ||| target span2
</SPANS>

Input:
{input}

Source:
{source}

Target:
{target}

Overlapping spans:
{spans}

Remember: the source side of each span pair must come from the list of overlapping spans, while the target side must use exact words from the target sentence, and both sides must convey the same meaning.
⚠️ Stop immediately after listing the span pairs.
</think>
"""

def get_formatted_prompt(
    sample
):
    print(f"my sample is: {sample}")
    #jmcc, next is to filter out some spans
    spans_filtered = []
    for span in sample["spans"]:
        if len(span) >= MIN_SPAN_LEN_IN_CHARS and not span in spans_filtered:
            spans_filtered.append(span)
    sample["spans"] = spans_filtered
    print(f"filtered spans: {spans_filtered}")
    
    formatted_prompt = prompt.format(
        input=sample["input"], 
        source=sample["source"], 
        target=sample["target"], 
        spans='\n'.join(sample["spans"])
    )
    return formatted_prompt


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


def read_newman_file(
    TEST_FILE,
    min_span_len = 2
):
    with open(TEST_FILE, 'r') as f:
        idx = 0
        for l in f:
            l = l.strip()
            if l.startswith("Source sentence #"):
                input = ' '.join(l.split()[3:])
            elif l.startswith("Ref sentence #"):
                output = ' '.join(l.split()[3:])
            elif l.startswith("Example English sentence:"):
                source = ' '.join(l.split()[3:])
            elif l.startswith("Example French sentence:"):
                target = ' '.join(l.split()[3:])
                idx += 1
                ### yield sample
                input_tokens = input.split()
                source_tokens = source.split()
                #source_spans = get_overlapping_spans(source_tokens, input_tokens)
                source_spans = get_matching_spans(input_tokens, source_tokens)
                if len(source_spans) == 0:
                    continue
                max_span_len = max([span[1]-span[0] for span in source_spans])
                if max_span_len < min_span_len:
                    continue
                source_spans = [' '.join(source_tokens[span[0]:span[1]]) for span in source_spans]
                yield {"idx": idx, "input": input, "output": output, "source": source, "target": target, "spans": source_spans}

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


