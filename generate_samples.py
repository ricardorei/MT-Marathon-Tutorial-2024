"""
Script used to generate samples for MBR decoding using VLLM and EuroLLM.

```bash
python generate_samples.py --lp en-es --num_candidates 20 --gpus 1 --output_file data/mbr_samples/en-es-samples.txt
```

"""
import argparse
import os

import pandas as pd
from vllm import LLM, SamplingParams

CODE_MAP = {
    "uk": "Ukrainian",
    "cs": "Czech",
    "de": "German",
    "es": "Spanish (Latin America)",
    "hi": "Hindi",
    "is": "Icelandic",
    "ja": "Japanese",
    "ru": "Russian",
    "zh": "Chinese",
    "en": "English",
}

# This is the template that EuroLLM was trained with. Different models
# have different templates.
CHATML_TEMPLATE = """<|im_start|>system
You are a professional {source_language} to {target_language} translator. Your goal is to accurately convey the meaning and nuances of the original {source_language} text while adhering to {target_language} grammar, vocabulary, and cultural sensitivities.
<|im_end|>
<|im_start|>user
Translate the following {source_language} source text to {target_language}:
{source_language}: {source}
{target_language}: <|im_end|>
<|im_start|>assistant
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generates samples for MBR decoding using EuroLLM.")
    parser.add_argument(
        "--lp",
        type=str,
        required=True,
        help="Language pair code from WMT24.",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=20,
        help="Number of samples to generate for each source sentence.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs used to generate the candidates (default: 1).",
    )
    parser.add_argument(
        "--output_file", type=str, help="Output file with translation candidates."
    )
    return parser.parse_args()


def divide_list_into_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

def main():
    args = parse_arguments()
    sampling_params = SamplingParams(
        use_beam_search=False,
        best_of=1,
        # Since we are doing sentence level we can stop at newlines. This avoids the model trying to continue
        # generating translations
        stop=["<|im_end|>", "\n"], 
        min_p=0.02,
        temperature=1,
        max_tokens=4096,
    )
    llm = LLM(model="utter-project/EuroLLM-1.7B-Instruct", tensor_parallel_size=args.gpus)
    source_sentences = [s.strip() for s in open(f"data/sources/{args.lp}.txt").readlines()]

    model_inputs = []
    for source in source_sentences:
        # Build MBR Inputs!
        source_language = CODE_MAP[args.lp.split("-")[0]]
        target_language = CODE_MAP[args.lp.split("-")[1]]

        source_prompt = CHATML_TEMPLATE.format(
            source_language=source_language,
            target_language=target_language,
            source=source,
        )
        # Each prompt will appear several times. VLLM also performs some prompt caching thus
        # this is actually fast.
        model_inputs.extend([source_prompt] * args.num_candidates)

    print("Model inputs are built. Starting generate! This can take some time!")
    # Generate translations
    outputs = llm.generate(model_inputs, sampling_params)
    generations = [o.outputs[0].text for o in outputs]
    # Extract the directory from the output_file path
    output_directory = os.path.dirname(args.output_file)
    # Create the directory if it does not exist
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    print (f"Saving translations to {args.output_file}")
    with open(args.output_file, 'w') as file:
        for line in generations:
            file.write(line + '\n')


if __name__ == "__main__":
    main()
