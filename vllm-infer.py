import time
import sys
import json
from typing import List, Dict, Any, Union
from vllm import LLM, SamplingParams
from utils import read_josep_file


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


def load_vllm_model(
    base_model_path: str, 
    **kwargs
) -> LLM:
    """Load model using VLLM for optimized inference."""    
    # VLLM configuration - automatically handles quantization and optimization
    llm = LLM(
        model=base_model_path,
        trust_remote_code=True,
        dtype="auto",  # VLLM will automatically choose the best dtype
        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        max_model_len=4096,  # Adjust based on your needs
        tensor_parallel_size=1,  # Set to number of GPUs if using multiple
        **kwargs
    )    
    return llm

def generate(
    llm: LLM,
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

def print_debug(
    prompt, 
    response
):
    sys.stderr.write(f"=== Prompt {sample['idx']} =============================\n")
    sys.stderr.write(f"{prompt}\n")
    sys.stderr.write(f"=== Response {sample['idx']} =============================\n")
    sys.stderr.write(f"{response}\n")


def process_batch(
    llm, 
    prompts,
):
    # Generate responses for this batch
    responses = generate(
        llm, 
        prompts, 
        max_tokens=1024,
        use_sampling=False
    )
    return responses    


def process_file(llm, IFILE, BATCH_SIZE=32):
     
    def get_pairs(prompt, response, idx):
        print(f"prompt {idx} = {prompt}")
        print(f"response {idx} = {response}")
        pairs = []
        for pair in response.split('\n'):
            if ' ||| ' in pair:
                toks = pair.split(' ||| ')
                if len(toks) == 2:
                    s, t = toks[0], toks[1]
                    pairs.append((s.strip(), t.strip()))
                else:
                    sys.stderr.write(f"skipping pair: {pair}")
        print(f"pairs {idx} = {pairs}")
        return pairs

    samples = []
    prompts = []
    with open(IFILE+'.V3.json', "w", encoding="utf-8") as fdo:

        def dump(samples, prompts, responses, idxs):
            for k in range(len(samples)):
                samples[k]['pairs'] = get_pairs(prompts[k], responses[k], idxs[k])
                fdo.write(json.dumps(samples[k], ensure_ascii=False) + "\n")
            fdo.flush()

        idxs = []
        for idx, sample in enumerate(read_josep_file(IFILE)):
            print(f"idx: {idx} sample: {sample}")
            samples.append(sample)
            idxs.append(idx)
            prompts.append(get_formatted_prompt(sample))
            if len(samples) == BATCH_SIZE:
                responses = process_batch(llm, prompts)
                dump(samples, prompts, responses, idxs)
                samples = []
                prompts = []
                idxs = []
        if len(samples):
            responses = process_batch(llm, prompts)
            dump(samples, prompts, responses, idxs)


if __name__ == "__main__":
    BASE_MODEL_NAME = "/lustre/fsmisc/dataset/HuggingFace_Models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    IFILE = sys.argv[1]
    BATCH_SIZE=64

    llm = load_vllm_model(BASE_MODEL_NAME)
    process_file(llm, IFILE, BATCH_SIZE=BATCH_SIZE)




    
