import time
import sys
import json
import argparse
from typing import List, Dict, Any, Union
from vllm import LLM, SamplingParams
from spans import get_spans_from_files
from utils import read_josep_file, get_formatted_prompt


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

def get_fragments(result):
    fragments = []
    for line in result.split('\n'):
        toks = line.split(' ||| ')
        if len(toks) == 2:
            s, t = toks[0], toks[1]
            fragments.append((s.strip(), t.strip()))
    return fragments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run inference of EuroLLM models using vLLM.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", type=str, required=True, help="Input file.")
    parser.add_argument("-o", type=str, required=True, help="Output file.")
    parser.add_argument("-s", type=str, required=True, help="Source file.")
    parser.add_argument("-t", type=str, required=True, help="Target file.")
    parser.add_argument("-output_json", type=str, required=True, help="Output JSON file to store results.")
    parser.add_argument("-min_tok_len", type=int, default=1, help="Minimum number of tokens in a span.")
    parser.add_argument("-min_str_len", type=int, default=3, help="Minimum number of characters in a span.")
    args = parser.parse_args()    

    BATCH_SIZE = 64
    BASE_MODEL_PATH = "/lustre/fsmisc/dataset/HuggingFace_Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    llm = load_vllm_model(BASE_MODEL_PATH)

    with open(args.output_json, 'w') as fdo:

        # dump into fdo a batch of samples with their generated pairs
        def dump(samples, results):
            for k in range(len(samples)):
                samples[k]['pairs'] = get_fragments(results[k])
                fdo.write(json.dumps(samples[k], ensure_ascii=False) + "\n")
            fdo.flush()

        samples, prompts = [], []

        for sample in get_spans_from_files(args.i, args.s, args.t, args.o, min_tok_len=args.min_tok_len, min_str_len=args.min_str_len):

            samples.append(sample)
            prompts.append(get_formatted_prompt(sample))

            if len(samples) == BATCH_SIZE:
                dump(samples, process_batch(llm, prompts))
                samples, prompts = [], []

        if len(samples):
            dump(samples, process_batch(llm, prompts))
            samples, prompts = [], []

    sys.stderr.write(f"Done\n")


    
