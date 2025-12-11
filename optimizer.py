import os
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

def find_json_files(datasets_root: str) -> List[Tuple[str, str, str]]:
    root = Path(datasets_root)
    results = []
    if not root.exists():
        raise FileNotFoundError(f"datasets_root not found: {datasets_root}")

    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        if ds_dir.name not in {"spright0", "spright1", "spright2", "spright3", "sprightcoco"}:
            continue
        llava_dir = ds_dir / "llava_output"
        if not llava_dir.exists():
            continue
        for p in sorted(llava_dir.glob("*.json")):
            results.append((ds_dir.name, str(p), p.name))
    return results

def extract_index_from_filename(filename: str) -> Optional[int]:
    m = re.match(r"(\d+)_expanded\.json$", filename)
    if m:
        return int(m.group(1))
    m2 = re.match(r"(\d+)", filename)
    if m2:
        return int(m2.group(1))
    return None

def load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_json(path: str, data: Dict) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save the file {path}: {e}")
        return False

def load_llama3(model_path: str, use_4bit: bool = True, force_fp16: bool = False, device: str = "auto", single_device: str = None):
    use_cuda = torch.cuda.is_available()
    if single_device is None:
        single_device = "cuda:0" if (device != "cpu" and use_cuda) else "cpu"
    if single_device.startswith("cuda") and not use_cuda:
        raise RuntimeError("GPU requested but not available in the current environment.")

    if single_device == "cpu":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    quantization_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gpu_index = 0
    if single_device.startswith("cuda:"):
        try:
            gpu_index = int(single_device.split(":")[1])
        except Exception:
            gpu_index = 0

    common_kwargs = dict(
        torch_dtype=torch_dtype,
        trust_remote_code=False,
        device_map=None,  
    )

    model = AutoModelForCausalLM.from_pretrained(model_path, **common_kwargs)
    model.eval()
    model.to(single_device)

    try:
        test_inputs = tokenizer("Hello", return_tensors="pt")
        dev = single_device if single_device != "cpu" else "cpu"
        test_inputs = {k: v.to(dev) for k, v in test_inputs.items()}
        with torch.no_grad():
            out = model(**test_inputs)
        if torch.isnan(out.logits).any() or torch.isinf(out.logits).any():
            print("[WARN] NaN/Inf detected in forward pass, possible precision overflow.")
    except Exception as chk_e:
        print(f"[WARN] Health check exception: {chk_e}")

    return tokenizer, model, single_device

def safe_generate(model_path: str, tokenizer, model, prompt: str,
                  max_new_tokens: int, temperature: float, top_p: float,
                  do_sample: bool, run_device: str):
    def _gen(tkzr, mdl, use_sample: bool, dev: str):
        gen_cfg = dict(
            max_new_tokens=max_new_tokens,
            do_sample=use_sample,
            temperature=(temperature if use_sample else None),
            top_p=(top_p if use_sample else None),
            eos_token_id=tkzr.eos_token_id,
            pad_token_id=tkzr.pad_token_id,
            use_cache=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        inputs = tkzr(prompt, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            logits_test = mdl(**inputs).logits
        if torch.isnan(logits_test).any() or torch.isinf(logits_test).any():
            raise RuntimeError("NaN/Inf logits detected, possible precision overflow.")
        with torch.no_grad():
            return mdl.generate(**inputs, **gen_cfg)[0], inputs["input_ids"].shape[-1]

    try:
        return _gen(tokenizer, model, do_sample, run_device)
    except Exception as e:
        msg = str(e)
        print(f"[WARN] First generation failed: {msg}")
        need_fallback = ("device-side assert" in msg) or ("CUDA error" in msg) or ("probability" in msg) or ("nan" in msg.lower()) or ("Inf" in msg)
        if not need_fallback:
            raise

    try:
        print("[INFO] Fallback: Forcing greedy decoding.")
        return _gen(tokenizer, model, False, run_device)
    except Exception as e1:
        print(f"[WARN] Greedy fallback failed: {e1}")

    if run_device.startswith("cuda"):
        try:
            print("[INFO] Fallback: Reloading GPU non-quantized model.")
            torch.cuda.empty_cache()
            tk2, m2, _dev2 = load_llama3(model_path, use_4bit=False, force_fp16=True, device="cuda", single_device=run_device)
            return _gen(tk2, m2, False, run_device)
        except Exception as e2:
            print(f"[WARN] GPU non-quantized fallback failed: {e2}")

    try:
        print("[INFO] Final fallback: CPU float32.")
        tk3, m3, _dev3 = load_llama3(model_path, use_4bit=False, force_fp16=True, device="cpu", single_device="cpu")
        return _gen(tk3, m3, False, "cpu")
    except Exception as e3:
        print(f"[ERROR] CPU fallback still failed: {e3}")
        raise RuntimeError(f"All generation attempts failed: {e3}")

def build_prompt(expanded_caption: str) -> str:
    cap = str(expanded_caption).strip()
    return (
        f"Simplify the following description by removing redundant and non-essential details. "
        f"Requirements:\n"
        f"1. Keep all spatial relations with FUZZY descriptions only\n"
        f"2. Convert ALL numeric measurements to vague terms:\n"
        f"   - Distance numbers (e.g., '3/4 width apart', '1/3 width away', '100 meters') → use 'very close', 'close', 'nearby', 'far', 'very far', etc.\n"
        f"   - Size numbers (e.g., '100 meters tall', 'twice as large') → use 'very tall', 'tall', 'large', 'small', 'tiny', etc.\n"
        f"   - Position numbers → use 'near', 'behind', 'in front of', 'beside', 'above', 'below', etc.\n"
        f"3. Keep only main objects and their fuzzy spatial relationships\n"
        f"4. Remove artistic embellishments and unnecessary attributes\n"
        f"5. Ensure LOGICAL CONSISTENCY - avoid contradictions:\n"
        f"   - Do NOT describe the same object with conflicting positions (e.g., both 'behind' and 'in front of')\n"
        f"   - Do NOT repeat the same spatial relationship multiple times\n"
        f"6. Avoid any repetition\n"
        f"7. Ensure the final text is fluent and natural, suitable for use in a dataset description.\n\n"
        f"Original description:\n{cap}\n\n"
        f"Simplified description with fuzzy semantics:"
    )

def generate_optimized_caption(tokenizer, model, expanded_caption: str, model_path: str,
                               max_resp_tokens: int = 512, temperature: float = 0.2,
                               do_sample: bool = False, run_device: str = "cpu") -> str:
    prompt = build_prompt(expanded_caption)
    try:
        output_ids, prompt_len = safe_generate(
            model_path=model_path,
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            max_new_tokens=max_resp_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=do_sample,
            run_device=run_device,
        )
    except Exception as e:
        raise RuntimeError(f"safe_generate final failure: {e}")

    new_tokens = output_ids[prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    if "Note:" in text:
        text = text.split("Note:")[0].strip()
    if "```" in text:
        text = text.split("```")[0].strip()
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        text = lines[0]
    
    prefixes_to_remove = [
        "Here is the simplified description:",
        "Simplified description:",
        "Output:",
        "Result:",
    ]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    return text

def save_results_tsv(rows: List[Dict], out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("dataset\tfilename\tindex\toptimized_caption\n")
        for r in rows:
            idx_str = "" if r["index"] is None else str(r["index"])
            cap = r["optimized_caption"].replace("\t", " ").replace("\n", " ").strip()
            f.write(f'{r["dataset"]}\t{r["filename"]}\t{idx_str}\t{cap}\n')

class GPUWorker:
    def __init__(self, worker_id: int, gpu_id: int, model_path: str, use_4bit: bool, temperature: float, no_greedy: bool):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.use_4bit = use_4bit
        self.temperature = temperature
        self.no_greedy = no_greedy
        self.device = f"cuda:{gpu_id}"
        
        print(f"[Worker {worker_id} @ GPU {gpu_id}] Loading model...")
        self.tokenizer, self.model, self.run_device = load_llama3(
            model_path,
            use_4bit=use_4bit,
            device="cuda",
            single_device=self.device
        )
        print(f"[Worker {worker_id} @ GPU {gpu_id}] Model loaded")
    
    def warmup(self):
        try:
            print(f"[Worker {self.worker_id}] Warming up...")
            test_prompt = "Test warmup prompt."
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            print(f"[Worker {self.worker_id}] Preheating complete.")
        except Exception as e:
            print(f"[Worker {self.worker_id}] Preheating failed: {e}")
    
    def process_file(self, dataset: str, path: str, fname: str, skip_existing: bool) -> Tuple[bool, str, str]:
        data = load_json(path)
        if not data:
            return False, "fail", "Failed to read file"
        
        if "expanded_caption" not in data:
            return False, "skip", "No expanded_caption"
        
        if skip_existing and "optimize_caption" in data and data["optimize_caption"]:
            return False, "skip", "Existing optimize_caption"
        
        cap = str(data["expanded_caption"])
        
        try:
            optimized = generate_optimized_caption(
                self.tokenizer,
                self.model,
                cap,
                model_path=self.model_path,
                max_resp_tokens=512,
                temperature=self.temperature,
                do_sample=(not self.no_greedy and self.temperature > 0.0),
                run_device=self.run_device,
            )
            
            data["optimize_caption"] = optimized
            if save_json(path, data):
                return True, "success", optimized
            else:
                return False, "fail", "Save failed"
                
        except Exception as e:
            return False, "fail", str(e)

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

_process_model = None
_process_tokenizer = None
_process_device = None

def init_process(gpu_id: int, model_path: str):
    """
    Process initialization function, called once when each process starts
    """
    global _process_model, _process_tokenizer, _process_device
    
    _process_device = f"cuda:{gpu_id}"
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    print(f"[Process [{mp.current_process().name}] is loading the model on GPU {gpu_id}...")
    
    _process_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if _process_tokenizer.pad_token is None:
        _process_tokenizer.pad_token = _process_tokenizer.eos_token
    
    _process_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=False,
    )
    _process_model.eval()
    _process_model.to(_process_device)
    
    print(f"[Process {mp.current_process().name}] Model loaded")

def process_single_file(args_tuple):
    dataset, path, fname, skip_existing, temperature, no_greedy = args_tuple
    
    global _process_model, _process_tokenizer, _process_device
    
    try:
        data = load_json(path)
        if not data:
            return False, "fail", "Failed to read file", fname
        
        if "expanded_caption" not in data:
            return False, "skip", "No expanded_caption", fname
        
        if skip_existing and "optimize_caption" in data and data["optimize_caption"]:
            return False, "skip", "Existing optimize_caption", fname
        
        cap = str(data["expanded_caption"])
        
        prompt = build_prompt(cap)
        inputs = _process_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(_process_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = _process_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=(not no_greedy and temperature > 0.0),
                temperature=(temperature if (not no_greedy and temperature > 0.0) else None),
                top_p=(0.9 if (not no_greedy and temperature > 0.0) else None),
                eos_token_id=_process_tokenizer.eos_token_id,
                pad_token_id=_process_tokenizer.pad_token_id,
                use_cache=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_len:]
        text = _process_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        if "Note:" in text:
            text = text.split("Note:")[0].strip()
        if "```" in text:
            text = text.split("```")[0].strip()
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            text = lines[0]
        
        prefixes_to_remove = [
            "Here is the simplified description:",
            "Simplified description:",
            "Output:",
            "Result:",
        ]
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        data["optimize_caption"] = text
        if save_json(path, data):
            return True, "success", text, fname
        else:
            return False, "fail", "Failed to save", fname
            
    except Exception as e:
        return False, "fail", str(e), fname

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description="Optimize the expanded_caption field in all JSON files by adding an optimize_caption field.")
    parser.add_argument("--datasets-root", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-greedy", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--processes-per-gpu", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    files = find_json_files(args.datasets_root)
    if not files:
        print("No JSON files found.")
        return

    total_files = len(files)
    print(f"Found {total_files} JSON files, processing all.")
    if not torch.cuda.is_available():
        print("[ERROR] No available GPU")
        return
    
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        args.num_gpus = available_gpus
    
    total_processes = args.num_gpus * args.processes_per_gpu
    
    print(f"[INFO] Configuration overview:")
    print(f"  - Usage: Multi-process (one model instance per process)")
    print(f"  - Number of GPUs: {args.num_gpus}")
    print(f"  - Processes per GPU: {args.processes_per_gpu}")
    print(f"  - Total processes: {total_processes}")
    print(f"  - Estimated VRAM per GPU: ~{args.processes_per_gpu * 18}GB / 80GB")
    
    if args.processes_per_gpu * 18 > 75:
        print(f"[WARN] Estimated VRAM {args.processes_per_gpu * 18}GB may exceed the limit!")
        print(f"[SUGGESTION] Use --processes-per-gpu 2 or 3")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\nClearing GPU cache...")
    torch.cuda.empty_cache()
    
    gpu_tasks = [[] for _ in range(args.num_gpus)]
    for i, (dataset, path, fname) in enumerate(files):
        gpu_id = i % args.num_gpus
        gpu_tasks[gpu_id].append((
            dataset, path, fname, args.skip_existing,
            args.temperature, args.no_greedy
        ))
    
    for i, tasks in enumerate(gpu_tasks):
        print(f"GPU {i}: Assigned {len(tasks)} tasks")
    
    print("\n" + "="*60)
    print("Starting file processing...")
    print("="*60 + "\n")
    
    start_time = time.time()
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    import threading
    from queue import Queue
    
    result_queue = Queue()
    
    def process_gpu_tasks(gpu_id, tasks, result_q):
        try:
            with ProcessPoolExecutor(
                max_workers=args.processes_per_gpu,
                initializer=init_process,
                initargs=(gpu_id, args.model_path)
            ) as executor:
                futures = [executor.submit(process_single_file, task) for task in tasks]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        result_q.put(result)
                    except Exception as e:
                        result_q.put((False, "fail", str(e), "unknown"))
        except Exception as e:
            print(f"[ERROR] GPU {gpu_id} process pool exception: {e}")
    
    gpu_threads = []
    for gpu_id in range(args.num_gpus):
        thread = threading.Thread(
            target=process_gpu_tasks,
            args=(gpu_id, gpu_tasks[gpu_id], result_queue),
            daemon=True
        )
        thread.start()
        gpu_threads.append(thread)
        print(f"GPU {gpu_id}: Starting processing thread, {len(gpu_tasks[gpu_id])} tasks.")
    
    print("\nCollecting processing results...\n")
    while any(t.is_alive() for t in gpu_threads) or not result_queue.empty():
        try:
            success, status, message, fname = result_queue.get(timeout=1)
            
            if status == "success":
                success_count += 1
                if success_count % 10 == 0:
                    elapsed = time.time() - start_time
                    completed = success_count + skip_count + fail_count
                    progress = (completed / total_files) * 100
                    speed = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_files - completed) / speed if speed > 0 else 0
                    
                    print(f"[{completed}/{total_files}] ✓ Progress: {progress:.1f}%")
                    print(f"    Success: {success_count} | Skipped: {skip_count} | Failed: {fail_count}")
                    print(f"    Speed: {speed:.2f} items/sec ({speed * 60:.1f} items/min)")
                    print(f"    Elapsed: {format_time(elapsed)} | Estimated remaining: {format_time(remaining)}\n")
                    
            elif status == "skip":
                skip_count += 1
            else:
                fail_count += 1
                if fail_count <= 10:
                    print(f"Failed: {fname} - {message[:100]}")
                    
        except:
            continue
    
    for thread in gpu_threads:
        thread.join(timeout=60)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"Processing complete！")
    print("="*60)
    print(f"success: {success_count} items")
    print(f"skipped: {skip_count} items")
    print(f"failed: {fail_count} items")
    print(f"total time: {format_time(total_time)}")
    if success_count > 0:
        print(f"  processing speed: {success_count / total_time * 60:.1f} items/min")
    print("="*60)

if __name__ == "__main__":
    main()
