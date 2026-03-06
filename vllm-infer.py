import time
import sys
import json
from typing import List, Dict, Any, Union
from vllm import LLM, SamplingParams
from utils import read_josep_file

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


def load_vllm_model(base_model_path: str, **kwargs) -> LLM:
    return LLM( # VLLM configuration - automatically handles quantization and optimization
        model=base_model_path,
        trust_remote_code=True,
        dtype="auto",  # VLLM will automatically choose the best dtype
        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        max_model_len=4096,  # Adjust based on your needs
        tensor_parallel_size=1,  # Set to number of GPUs if using multiple
        **kwargs
    )

def generate(llm: LLM, 
             prompts: List[str], 
             max_tokens: int = 256, 
             temperature: float = 0.7, 
             top_p: float = 0.9, 
             repetition_penalty: float = 1.1, 
             use_sampling: bool = False
            ) -> List[str]:
    """
    Generate responses for single prompt or batch of prompts.
    Returns: Single string if input was string, list of strings if input was list
    """    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature if use_sampling else 0.0,
        top_p=top_p if use_sampling else 1.0,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        skip_special_tokens=True,
    )
    # Generate responses in batch
    outputs = llm.generate(
        prompts, 
        sampling_params
    )
    # Return responses
    return [output.outputs[0].text for output in outputs]

def process_batch(llm, prompts):
    # Generate responses for this batch
    responses = generate(
        llm, 
        prompts, 
        max_tokens=1024,
        use_sampling=False
    )
    return responses    

def process_file(llm, IFILE, OFILE, BATCH_SIZE=32):
     
    def get_fragments(idx, prompt, result, verbose=True):
        fragments = []
        for line in result.split('\n'):
            toks = line.split(' ||| ')
            if len(toks) == 2:
                s, t = toks[0], toks[1]
                fragments.append((s.strip(), t.strip()))
        if verbose:
            print(f"=== Prompt {idx} =============================\n{prompt}")
            print(f"=== Result {idx} =============================\n{result}")
            print(f"=== Fragments {idx} =============================\n{'\n'.join(fragments)}")
        return fragments

    idxs, samples, prompts = [], [], []

    with open(OFILE, "w", encoding="utf-8") as fdo:

        def dump(idxs, samples, prompts, results):
            for k in range(len(samples)):
                samples[k]['pairs'] = get_fragments(idxs[k], prompts[k], results[k])
                fdo.write(json.dumps(samples[k], ensure_ascii=False) + "\n")
            fdo.flush()

        for idx, sample in enumerate(read_josep_file(IFILE)):

            idxs.append(idx)
            samples.append(sample)
            prompts.append(get_formatted_prompt(sample))

            if len(samples) == BATCH_SIZE:
                results = process_batch(llm, prompts)
                dump(idxs, samples, prompts, results)
                idxs, samples, promtps = [], [], []

        if len(samples):
            results = process_batch(llm, prompts)
            dump(samples, prompts, results, idxs)

    sys.stderr.write(f"Done\n")


if __name__ == "__main__":
    BASE_MODEL_PATH = "/lustre/fsmisc/dataset/HuggingFace_Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    IFILE = sys.argv[1]
    OFILE = sys.argv[2] + '.json' if not sys.argv[2].endswith('.json') else sys.argv[2]

    llm = load_vllm_model(BASE_MODEL_PATH)
    process_file(llm, IFILE, OFILE)




    
