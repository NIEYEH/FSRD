import os
import json
import random
import torch
from diffusers import (
    StableDiffusionPipeline, 
    DPMSolverMultistepScheduler,
    PixArtSigmaPipeline,
    PixArtAlphaPipeline,
    FluxPipeline
)
from pathlib import Path
from tqdm import tqdm

CONFIG = {
    "JSONL_PATH": "spatial_descriptions.jsonl",
    
    "MODELS": {
        "model_1": "",
        "model_2": "sprightsd2",
        "model_3": "stable_diffusion",
        "model_4": "PixArt-Sigma",
        "model_5": "PixArt-alpha",
        "model_6": "FLUX.1-schnell"
    },
    
    "OUTPUT_BASE_DIR": "images",
    
    "NUM_SENTENCES": 10, 
    "IMAGES_PER_SENTENCE": 3, 
    "IMAGE_SIZE": 512,
    "NUM_INFERENCE_STEPS": 30,
    "GUIDANCE_SCALE": 7.5,
    "SEED": -1, 
}


def load_sentences(jsonl_path, num_sentences, seed=42):
    print(f"Loading sentences from {jsonl_path}...")
    
    sentences = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            sentences.append(data)
    
    print(f"Loaded a total of {len(sentences)} sentences")
    
    random.seed(seed)
    selected = random.sample(sentences, min(num_sentences, len(sentences)))
    
    print(f"Randomly selected {len(selected)} sentences")
    return selected


def detect_model_type(model_path):
    model_path_obj = Path(model_path)
    model_name = model_path_obj.name.lower()
    
    if "pixartPixArt-Sigma" in model_name:
        return "PixArt-Sigma"
    elif "pixart-alpha" in model_name:
        return "PixArt-alpha"
    elif "flux" in model_name:
        return "flux"
    else:
        return "stable_diffusion"


def load_model(model_path, device):
    print(f"\nLoading model: {model_path}")
    
    try:
        model_path_obj = Path(model_path)
        model_type = detect_model_type(model_path)
        print(f"Detected model type: {model_type}")
        
        if model_type == "PixArt-Sigma":
            pipe = PixArtSigmaPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()
        elif model_type == "PixArt-alpha":
            pipe = PixArtAlphaPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()
        elif model_type == "flux":
            print("Using FluxPipeline...")
            pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            )
            pipe = pipe.to(device)
            
        else:  # stable_diffusion
            print("Using StableDiffusionPipeline...")
            if (model_path_obj / "checkpoint-15000").exists():
                checkpoint_path = model_path_obj / "checkpoint-15000"
                print(f"Detected checkpoint directory, using: {checkpoint_path}")
                pipe = StableDiffusionPipeline.from_pretrained(
                    str(model_path_obj),
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)
            pipe.enable_attention_slicing()
        
        print(f"Model loading complete.")
        return pipe
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_images_for_sentence(pipe, prompt, output_dir, sentence_idx, num_images, config):
    
    os.makedirs(output_dir, exist_ok=True)
    
    generated_paths = []
    
    is_flux = isinstance(pipe, FluxPipeline)
    is_pixartsigma = isinstance(pipe, PixArtSigmaPipeline)
    is_pixartalpha = isinstance(pipe, PixArtAlphaPipeline)
    
    for img_idx in range(num_images):
        seed = config["SEED"] + sentence_idx * 100 + img_idx
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        try:
            if is_flux:
                image = pipe(
                    prompt=prompt,
                    height=config["IMAGE_SIZE"],
                    width=config["IMAGE_SIZE"],
                    num_inference_steps=config["NUM_INFERENCE_STEPS"],
                    generator=generator,
                    guidance_scale=0.0  
                ).images[0]
            elif is_pixartsigma or is_pixartalpha:
                image = pipe(
                    prompt=prompt,
                    height=config["IMAGE_SIZE"],
                    width=config["IMAGE_SIZE"],
                    num_inference_steps=config["NUM_INFERENCE_STEPS"],
                    generator=generator
                ).images[0]
            else:
                image = pipe(
                    prompt=prompt,
                    height=config["IMAGE_SIZE"],
                    width=config["IMAGE_SIZE"],
                    num_inference_steps=config["NUM_INFERENCE_STEPS"],
                    guidance_scale=config["GUIDANCE_SCALE"],
                    generator=generator
                ).images[0]
            
            output_filename = f"sentence_{sentence_idx:03d}_image_{img_idx + 1}.png"
            output_path = os.path.join(output_dir, output_filename)
            image.save(output_path)
            
            generated_paths.append(output_path)
            
        except Exception as e:
            print(f"Generation failed (sentence {sentence_idx}, image {img_idx + 1}): {e}")
            import traceback
            traceback.print_exc()
    
    return generated_paths


def main():
    if not torch.cuda.is_available():
        print("Error: CUDA support is required")
        return
    
    device = "cuda"
    print(f"Using device: {device}")
    
    sentences = load_sentences(
        CONFIG["JSONL_PATH"],
        CONFIG["NUM_SENTENCES"],
        CONFIG["SEED"]
    )
    
    selected_sentences_path = os.path.join(CONFIG["OUTPUT_BASE_DIR"], "selected_sentences.json")
    os.makedirs(CONFIG["OUTPUT_BASE_DIR"], exist_ok=True)
    with open(selected_sentences_path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"\nSaved selected sentences to: {selected_sentences_path}")
    
    for model_name, model_path in CONFIG["MODELS"].items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        pipe = load_model(model_path, device)
        if pipe is None:
            print(f"Skipping model {model_name}")
            continue
        
        model_output_dir = os.path.join(CONFIG["OUTPUT_BASE_DIR"], model_name)
        
        for idx, sentence_data in enumerate(tqdm(sentences, desc=f"{model_name} generating")):
            sentence = sentence_data["sentence"]
            
            print(f"\n[{idx + 1}/{len(sentences)}] Sentence: {sentence}")
            
            generated_paths = generate_images_for_sentence(
                pipe,
                sentence,
                model_output_dir,
                idx,
                CONFIG["IMAGES_PER_SENTENCE"],
                CONFIG
            )
            
            print(f"  ✓ Generated {len(generated_paths)} images")
        
        del pipe
        torch.cuda.empty_cache()
        
        print(f"\n✓ Model {model_name} processing completed")
    
    print(f"\n{'='*60}")
    print("All images have been generated.！")
    print(f"Output directory: {CONFIG['OUTPUT_BASE_DIR']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
