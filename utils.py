import difflib
import string
import json
import sys
from collections import defaultdict

prompt1 = """
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

prompt2 = """
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
Do not write explanations.
Do not write reasoning.

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

--------------------------------
Output
--------------------------------

</think>
"""

prompt3 = """
You are given:

1. An input sentence to be translated.
2. A translation example consisting of:
   - Source sentence (example source)
   - Target sentence (its translation)
3. A list of overlapping spans, each appearing in both the input sentence and the example source sentence.

Task:
For each overlapping span, find the corresponding span in the example target sentence.

Rules:
- The target span must be the translation of the source span as it appears in the target sentence.
- Spans may contain gaps written as "<GAP>", indicating omitted words.
- Words translating the gap <GAP> must NOT appear in the target span. 
- Use a gap <GAP> in the target span to indicate the corresponding omission.

Output format (one line per span):
source span ||| target span

Additional rules:
- Only process spans listed under "overlapping spans".
- Do not invent new spans.
- Preserve gaps <GAP> exactly as in the source span.
- Do not include reasoning or explanations.
- Stop generating after the last span.

--------------------------------
Example
--------------------------------

Input sentence:
The committee approved a policy on environmental issues and the new financial regulations

Source sentence:
The committee adopted a policy on environmental protection and financial regulations

Target sentence:
Le comité a adopté une politique sur la protection de l'environnement et les régulations financières

Overlapping spans:
The committee <GAP> policy <GAP> financial regulations

Output:
The committee <GAP> policy <GAP> financial regulations ||| Le comité <GAP> une politique <GAP> régulations financières

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

Output:
</think>
"""

def get_formatted_prompt(sample, prompt_num=1):
    prompt = prompt1 if prompt_num == 1 else (prompt2 if prompt_num == 2 else prompt3)
    return prompt.format(
        input=sample["input"],
        source=sample["source"],
        target=sample["target"],
        spans='\n'.join(sample["spans"])
    )

