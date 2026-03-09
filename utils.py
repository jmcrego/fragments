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

pormpt3 = """
You are given:

1) An input sentence to be translated.

2) A translation example consisting of:
   - Source sentence (example source)
   - Target sentence (its translation)

Your task is to identify translation units in the example that could help translate the input sentence.

Each unit must consist of:
- A source span from the example source sentence
- Its corresponding translation span from the example target sentence

The source span must match words that appear in both:
- the input sentence
- the example source sentence

The target span must appear in the example target sentence.

Output format:
source span ||| target span

Rules:
- Prefer larger units over smaller ones.
- If a span is fully contained inside a larger span, keep only the larger span.
- Prefer spans that form meaningful translation units.
- You may use units with gaps. Use <GAP> to indicate missing words on either side of the unit.

--------------------------------
Example
--------------------------------

Input sentence:
Can you give me the money back, now?

Source sentence:
Could you give me my toy back now, please?

Target sentence:
Pourrais-tu me rendre mon jouet maintenant, s'il te plaît ?

Output:

give me <GAP> back ||| me rendre
you ||| tu
now ||| maintenant

--------------------------------
Additional Rules
--------------------------------

- Do not include spans containing only punctuation or determiners (e.g. "of the", "in the").
- Do not include spans where the alignment is incorrect or unclear.
- Adapt the target span so it fits naturally when translating the input sentence (adjust morphology or phrasing if necessary).
- Do not output reasoning nor explanations.
- Output only the list of units.
- Stop generating immediately after the last unit.

--------------------------------
Input
--------------------------------

Input sentence:
{input}

Example source:
{source}

Example target:
{target}

--------------------------------
Output
--------------------------------

</think>
"""

def get_formatted_prompt(sample, prompt_num=1):
    if prompt_num == 1:
        return prompt1.format(
            input=sample["input"],
            source=sample["source"],
            target=sample["target"],
            spans='\n'.join(sample["spans"])
        )
    elif prompt_num == 2:
        return prompt2.format(
            input=sample["input"],
            source=sample["source"],
            target=sample["target"],
            spans='\n'.join(sample["spans"])
        )
    elif prompt_num == 3:
        return prompt3.format(
            input=sample["input"],
            source=sample["source"],
            target=sample["target"],
            spans='\n'.join(sample["spans"])
        )

