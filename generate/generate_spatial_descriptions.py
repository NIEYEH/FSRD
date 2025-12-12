import json
import os
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['VLLM_RPC_TIMEOUT'] = '600' 
os.environ['NCCL_TIMEOUT'] = '600'     
os.environ['VLLM_NCCL_TIMEOUT'] = '600'

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Large language models
MODEL_PATH = ""

# Output file path
OUTPUT_FILE = ""

TARGET_TOTAL_COUNT = 10000
BATCH_SIZE_HINT = 10

# Prompt Template
SYSTEM_PROMPT_TEMPLATE = """### System Instruction
You are a specialized AI dataset generator. Your task is to generate precise, objective, and strictly physical English descriptions of spatial relationships between objects.

### Task
Generate a list of {count} sentences containing complex spatial relationships based on the **Domain/Theme** provided below.

### Requirements
1.  **Domain/Theme:** Freely choose diverse scenes (Industrial, Nature, Home, Sci-fi, etc.).
2.  **Mandatory Spatial Combination:**
    Every sentence MUST explicitly combine:
    * **[Direction]:** (e.g., to the left, right, in front, behind, diagonal).
    * **[Fuzzy Distance]:** (e.g., tightly attached, grazing, within reach, squeezed against).
3.  **Tone & Style (CRITICAL):**
    * **Strictly Physical & Objective:** Describe **ONLY** the visual arrangement.
    * **NO Fluff/Context:** Do NOT describe the purpose, atmosphere, or emotion.
        * *NO:* "...inviting lazy afternoons."
        * *NO:* "...looking like it's been abandoned."
        * *NO:* "...perfect for working."
    * **Concise:** Keep the sentence focused solely on Object A's position relative to Object B.
4.  **Diversity:** Vary the sentence structures so they don't all look identical, but keep them factual.

### Examples
* **Bad (Too flowery):** "To the left of the sofa, a coffee table sits within reach, inviting lazy afternoons of sipping tea."
* **Bad (Too simple):** "The table is left. The table is close."
* **Good (Target):** "To the left of the sofa, a coffee table sits within reach, its edge aligned parallel to the cushion."
* **Good (Target):** "A heavy tool chest is located directly behind the workbench, squeezed tightly against the metal frame."

### Output Format
Strictly output a JSON list.
[
  {{
    "scene": "Scene Name",
    "objects": ["Object A", "Object B"],
    "direction": "The directional keyword used",
    "fuzzy_distance": "The fuzzy keyword used",
    "sentence": "The strictly physical sentence."
  }},
  ...
]"""

def load_model(model_path):
    print(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

    print(f"Loading vLLM engine from: {model_path}")
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPUs. Using tensor_parallel_size={gpu_count}")

    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=gpu_count,
            gpu_memory_utilization=0.90,
            quantization="awq", 
            dtype="float16",    
            max_model_len=8192,
            enforce_eager=True,
            enable_chunked_prefill=False, 
            disable_custom_all_reduce=True 
        )
    except Exception as e:
        print(f"Error loading vLLM: {e}")
        sys.exit(1)
        
    return tokenizer, llm

def extract_json_list(text):
    """Extracts and parses a JSON list from a string that might contain other text."""
    try:
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"Error parsing JSON: {e}")
    return []

def main():
    if not torch.cuda.is_available():
        print("Error: CUDA is required for vLLM.")
        sys.exit(1)

    tokenizer, llm = load_model(MODEL_PATH)
    
    print(f"Output will be saved to: {os.path.abspath(OUTPUT_FILE)}")
    
    stop_token_ids = [tokenizer.eos_token_id]
    if tokenizer.convert_tokens_to_ids("<|endoftext|>") is not None:
         stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|endoftext|>"))

    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.95,
        max_tokens=2048,
        stop_token_ids=stop_token_ids
    )

    total_generated = 0
    batch_index = 0

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        while total_generated < TARGET_TOTAL_COUNT:
            batch_index += 1
            print(f"\n--- Batch {batch_index} (Total Generated: {total_generated}/{TARGET_TOTAL_COUNT}) ---")
            
            prompt_content = SYSTEM_PROMPT_TEMPLATE.format(count=BATCH_SIZE_HINT)
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt_content}
            ]
            
            text_input = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            try:
                outputs = llm.generate([text_input], sampling_params)
                generated_text = outputs[0].outputs[0].text
                
                items = extract_json_list(generated_text)
                
                if items:
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                    count = len(items)
                    total_generated += count
                    print(f"  Batch {batch_index}: Saved {count} sentences.")
                else:
                    print(f"  Batch {batch_index}: No valid JSON found in output.")
                    
            except Exception as e:
                print(f"Error during generation: {e}")
                continue

    print(f"\nDone! Total sentences generated: {total_generated}")

if __name__ == "__main__":
    main()
