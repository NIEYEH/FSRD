import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Any

CONFIG = {
    "CLIP_MODEL_PATH": "",
    
    "SELECTED_SENTENCES_PATH": "",
    "IMAGES_BASE_DIR": "",
    
    "OUTPUT_DIR": "",
    
    "BATCH_SIZE": 8,  
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}


class CLIPEvaluator:
    
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading CLIP model: {model_path}")
        self.device = device
        
        try:
            self.model = CLIPModel.from_pretrained(model_path).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_path)
            self.model.eval()
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Failed to load CLIP model: {e}")
            raise
    
    def generate_evaluation_prompts(self, sentence_data: Dict) -> List[str]:
        sentence = sentence_data["sentence"]
        scene = sentence_data.get("scene", "")
        objects = sentence_data.get("objects", [])
        fuzzy_term = sentence_data.get("fuzzy_term", "")
        
        prompts = []
        
        prompts.append({
            "type": "full_match",
            "text": sentence,
            "weight": 0.4
        })
        
        if objects and len(objects) >= 2:
            obj_prompt = f"An image showing {objects[0]} and {objects[1]}"
            prompts.append({
                "type": "objects",
                "text": obj_prompt,
                "weight": 0.25
            })
        
        if fuzzy_term:
            spatial_prompt = f"{objects[0]} {fuzzy_term} {objects[1]}" if len(objects) >= 2 else sentence
            prompts.append({
                "type": "spatial_relation",
                "text": spatial_prompt,
                "weight": 0.25
            })
        
        if scene:
            scene_prompt = f"A {scene} scene with {objects[0]} and {objects[1]}" if len(objects) >= 2 else f"A {scene} scene"
            prompts.append({
                "type": "scene",
                "text": scene_prompt,
                "weight": 0.1
            })
        
        return prompts
    
    def compute_similarity(self, image_path: str, prompts: List[Dict]) -> Dict[str, float]:
        try:
            image = Image.open(image_path).convert("RGB")
            
            texts = [p["text"] for p in prompts]
            
            inputs = self.processor(
                text=texts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            scores = {}
            weighted_score = 0.0
            
            for idx, prompt_info in enumerate(prompts):
                score = probs[0][idx].item()
                scores[prompt_info["type"]] = score
                weighted_score += score * prompt_info["weight"]
            
            final_score = 1 + weighted_score * 4  
            
            return {
                "final_score": final_score,
                "dimension_scores": scores,
                "weighted_average": weighted_score
            }
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {
                "final_score": 0.0,
                "dimension_scores": {},
                "weighted_average": 0.0,
                "error": str(e)
            }
    
    def generate_reasoning(self, scores: Dict, sentence_data: Dict) -> str:
        final_score = scores["final_score"]
        dim_scores = scores.get("dimension_scores", {})
        
        reasoning_parts = []
        
        if final_score >= 4.5:
            reasoning_parts.append("The image highly matches the description.")
        elif final_score >= 3.5:
            reasoning_parts.append("The image matches the description well.")
        elif final_score >= 2.5:
            reasoning_parts.append("The image partially matches the description.")
        else:
            reasoning_parts.append("The image matches the description poorly.")
        
        if "full_match" in dim_scores:
            score = dim_scores["full_match"]
            if score > 0.7:
                reasoning_parts.append(f"Full description match is high ({score:.2f}).")
            elif score > 0.5:
                reasoning_parts.append(f"Full description match is moderate ({score:.2f}).")
            else:
                reasoning_parts.append(f"Full description match is low ({score:.2f}).")
        
        if "objects" in dim_scores:
            score = dim_scores["objects"]
            objects = sentence_data.get("objects", [])
            if score > 0.7:
                reasoning_parts.append(f"Objects are accurately recognized, {objects[0]} and {objects[1]} are clearly visible.")
            elif score > 0.5:
                reasoning_parts.append(f"Object recognition is generally accurate but may not be clear.")
            else:
                reasoning_parts.append(f"Object recognition has issues.")
        
        if "spatial_relation" in dim_scores:
            score = dim_scores["spatial_relation"]
            fuzzy_term = sentence_data.get("fuzzy_term", "")
            if score > 0.7:
                reasoning_parts.append(f"Spatial relation '{fuzzy_term}' is clearly represented.")
            elif score > 0.5:
                reasoning_parts.append(f"Spatial relation '{fuzzy_term}' is moderately represented.")
            else:
                reasoning_parts.append(f"Spatial relation '{fuzzy_term}' is not clearly represented.")
        
        if "scene" in dim_scores:
            score = dim_scores["scene"]
            scene = sentence_data.get("scene", "")
            if score > 0.7:
                reasoning_parts.append(f"Scene '{scene}' is highly matched.")
            elif score > 0.5:
                reasoning_parts.append(f"Scene '{scene}' is moderately matched.")
            else:
                reasoning_parts.append(f"Scene '{scene}' is poorly matched.")
        
        return " ".join(reasoning_parts)


def load_sentences(json_path: str) -> List[Dict]:
    print(f"Loading sentences: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        sentences = json.load(f)
    print(f"Loaded {len(sentences)} sentences")
    return sentences


def evaluate_model(evaluator: CLIPEvaluator, model_name: str, 
                   sentences: List[Dict], images_base_dir: str) -> List[Dict]:

    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*60}")
    
    results = []
    model_dir = os.path.join(images_base_dir, model_name)
    
    if not os.path.exists(model_dir):
        print(f"Model directory does not exist: {model_dir}")
        return results
    
    for sent_idx, sentence_data in enumerate(tqdm(sentences, desc=f"Evaluating {model_name}")):
        sentence_result = {
            "sentence_idx": sent_idx,
            "sentence": sentence_data["sentence"],
            "scene": sentence_data.get("scene", ""),
            "objects": sentence_data.get("objects", []),
            "fuzzy_distance": sentence_data.get("fuzzy_term", ""),
            "images": []
        }
        
        prompts = evaluator.generate_evaluation_prompts(sentence_data)
        
        image_pattern = f"sentence_{sent_idx:03d}_image_*.png"
        image_files = sorted(Path(model_dir).glob(image_pattern))
        
        for img_path in image_files:
            img_idx = int(img_path.stem.split("_")[-1])
            
            scores = evaluator.compute_similarity(str(img_path), prompts)
            
            reasoning = evaluator.generate_reasoning(scores, sentence_data)
            
            image_result = {
                "image_idx": img_idx,
                "image_filename": img_path.name,
                "clip_score": round(scores["final_score"], 2),
                "weighted_average": round(scores["weighted_average"], 4),
                "dimension_scores": {k: round(v, 4) for k, v in scores.get("dimension_scores", {}).items()},
                "reasoning": reasoning,
                "error": scores.get("error", None)
            }
            
            sentence_result["images"].append(image_result)
        
        results.append(sentence_result)
    
    return results


def generate_summary_report(all_results: Dict[str, List[Dict]], output_dir: str):
    summary_path = os.path.join(output_dir, "clip_evaluation_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CLIP Model Evaluation Summary Report\n")
        f.write("="*80 + "\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Model: {model_name}\n")
            f.write("-"*80 + "\n")
            
            all_scores = []
            error_count = 0
            
            for sent_result in results:
                for img_result in sent_result["images"]:
                    if img_result.get("error") is None:
                        all_scores.append(img_result["clip_score"])
                    else:
                        error_count += 1
            
            if not all_scores:
                f.write("No valid scores available\n")
                continue
            
            total_images = len(all_scores) + error_count
            avg_score = np.mean(all_scores)
            std_score = np.std(all_scores)
            
            f.write(f"Total images: {total_images}\n")
            f.write(f"Valid scores: {len(all_scores)}\n")
            f.write(f"Evaluation errors: {error_count}\n")
            f.write(f"Average score: {avg_score:.2f}/5.0\n")
            f.write(f"Standard deviation: {std_score:.2f}\n\n")
            
            f.write("Score distribution:\n")
            bins = [1, 2, 3, 4, 5]
            for i in range(len(bins)):
                lower = bins[i]
                upper = bins[i] + 1 if i < len(bins) - 1 else 6
                count = sum(1 for s in all_scores if lower <= s < upper)
                percentage = (count / len(all_scores)) * 100
                bar = "█" * int(percentage / 2)
                f.write(f"  {lower}-{upper} points: {count:3d} ({percentage:5.1f}%) {bar}\n")
            
            f.write(f"\nHighest score: {max(all_scores):.2f}\n")
            f.write(f"Lowest score: {min(all_scores):.2f}\n")
            
            f.write("\nAverage dimension scores:\n")
            dimension_totals = {}
            dimension_counts = {}
            
            for sent_result in results:
                for img_result in sent_result["images"]:
                    if img_result.get("error") is None:
                        for dim, score in img_result.get("dimension_scores", {}).items():
                            dimension_totals[dim] = dimension_totals.get(dim, 0) + score
                            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
            
            for dim in sorted(dimension_totals.keys()):
                avg = dimension_totals[dim] / dimension_counts[dim]
                f.write(f"  {dim}: {avg:.4f}\n")
    
    print(f"\n✓ Summary report saved to: {summary_path}")


def main():
    print("="*80)
    print("CLIP Image Evaluation System")
    print("="*80)
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (slower speed)")
    
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    
    evaluator = CLIPEvaluator(
        CONFIG["CLIP_MODEL_PATH"],
        CONFIG["DEVICE"]
    )
    
    sentences = load_sentences(CONFIG["SELECTED_SENTENCES_PATH"])
    
    all_results = {}
    
    images_base_dir = CONFIG["IMAGES_BASE_DIR"]
    model_dirs = [d for d in os.listdir(images_base_dir) 
                  if os.path.isdir(os.path.join(images_base_dir, d)) and d.startswith("model_")]
    
    if not model_dirs:
        print(f"No model directories found in {images_base_dir}")
        return
    
    print(f"\nFound {len(model_dirs)} models: {', '.join(model_dirs)}")
    
    for model_name in model_dirs:
        results = evaluate_model(evaluator, model_name, sentences, images_base_dir)
        all_results[model_name] = results
    
    results_path = os.path.join(CONFIG["OUTPUT_DIR"], "clip_evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {results_path}")
    
    generate_summary_report(all_results, CONFIG["OUTPUT_DIR"])
    
    print("\n" + "="*80)
    print("Evaluation completed!")
    print("="*80)
    print(f"\nOutput directory: {CONFIG['OUTPUT_DIR']}")
    print(f"  - Detailed results: clip_evaluation_results.json")
    print(f"  - Summary report: clip_evaluation_summary.txt")

if __name__ == "__main__":
    main()
