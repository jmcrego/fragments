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

1) An input sentence to be translated.

2) A translation example consisting of:
   - Source sentence (example source)
   - Target sentence (its translation)

3) A list of overlapping spans that appear in BOTH:
   - the input sentence
   - the example source sentence

Your task is to identify the target side spans in the example target sentence that correspond to the source side spans.
Units may contain gaps (…) on either side, indicating missing words.

Output format:
source span ||| target span

--------------------------------
Example
--------------------------------

Input sentence:
The committee has approved a new policy for protecting the environment

Source sentence:
The committee adopted a new policy on environmental protection

Target sentence:
Le comité a adopté une nouvelle politique de protection de l'environnement

Overlapping spans:
The committee (…) a new policy

Output:
The committee (…) a new policy ||| Le comité (…) une nouvelle politique

--------------------------------
Additional Rules
--------------------------------

- Do not include additional spans beyond those in the "Overlapping spans" list.
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

