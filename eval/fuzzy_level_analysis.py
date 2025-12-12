import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

CONFIG = {
    "FUZZY_TERM_MAPPING": "",
    
    "HARD_EVAL_RESULTS": "",
    "SELECTED_SENTENCES": "",
    "IMAGES_BASE_DIR": "",
    
    "CLIP_MODEL_PATH": "",
    
    "OUTPUT_DIR": "",
    
    "CONTACT_IOU_THRESHOLD": 0.0, 
    "CONTACT_PIXEL_EPSILON": 5,    
    
    "LEVEL_NAMES": {
        "L1": "Contact & Pressure",
        "L2": "Grazing & Zero-Gap", 
        "L3": "Proximity & Reach"
    },
    
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}


class FuzzyTermAnalyzer:
    def __init__(self):
        self.fuzzy_mapping = None
        self.level_definitions = None
        self.hard_results = None
        self.sentences = None
        
        print("Loading CLIP model...")
        self.clip_processor = CLIPProcessor.from_pretrained(CONFIG["CLIP_MODEL_PATH"])
        self.clip_model = CLIPModel.from_pretrained(CONFIG["CLIP_MODEL_PATH"]).to(CONFIG["DEVICE"])
        self.clip_model.eval()
        print("✓ CLIP model loaded")
    
    def load_fuzzy_mapping(self, mapping_path: str):
        print(f"\nLoading fuzzy term mapping: {mapping_path}")
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.fuzzy_mapping = data["mapping"]
        self.level_definitions = data["meta_info"]["definitions"]
        
        level_counts = {}
        for term, level in self.fuzzy_mapping.items():
            level_counts[level] = level_counts.get(level, 0) + 1
        
        print(f"Loaded {len(self.fuzzy_mapping)} fuzzy terms")
        for level, count in sorted(level_counts.items()):
            print(f"  {level}: {count} terms - {self.level_definitions[level]}")
    
    def load_evaluation_results(self, hard_path: str, sentences_path: str):
        print("\nLoading evaluation results...")
        
        with open(hard_path, 'r', encoding='utf-8') as f:
            self.hard_results = json.load(f)
        print(f"✓ Hard metric results: {hard_path}")
        
        with open(sentences_path, 'r', encoding='utf-8') as f:
            self.sentences = json.load(f)
        print(f"✓ Sentence data: {sentences_path}")
    
    def get_fuzzy_level(self, fuzzy_term: str) -> str:
        term_lower = fuzzy_term.lower().strip()
        return self.fuzzy_mapping.get(term_lower, "Unknown")
    
    def check_contact_detection(self, detection_result: Dict, fuzzy_term: str) -> Dict:
        level = self.get_fuzzy_level(fuzzy_term)
        
        scores = detection_result.get("scores", {})
        detections = scores.get("detections", [])
        
        if len(detections) < 2:
            return {
                "level": level,
                "contact_expected": bool(level == "L1"),
                "contact_detected": False,
                "iou": 0.0,
                "pixel_distance": float('inf'),
                "pass_contact_check": False,
                "reason": "Less than 2 objects detected"
            }
        
        box1 = detections[0]["box"]
        box2 = detections[1]["box"]
        
        iou = self._calculate_iou(box1, box2)
        
        pixel_dist = self._calculate_pixel_distance(box1, box2)
        
        contact_detected = (iou > CONFIG["CONTACT_IOU_THRESHOLD"] or 
                          pixel_dist < CONFIG["CONTACT_PIXEL_EPSILON"])
        
        contact_expected = (level == "L1")
        pass_check = (not contact_expected) or contact_detected
        
        return {
            "level": level,
            "fuzzy_term": fuzzy_term,
            "contact_expected": bool(contact_expected),
            "contact_detected": bool(contact_detected),
            "iou": round(float(iou), 4),
            "pixel_distance": round(float(pixel_dist), 2),
            "pass_contact_check": bool(pass_check),
            "reason": self._get_contact_reason(level, contact_detected, iou, pixel_dist)
        }
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_pixel_distance(self, box1: List[float], box2: List[float]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        if not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1):
            return 0.0
        
        dx = max(0, max(x1_1 - x2_2, x1_2 - x2_1))
        dy = max(0, max(y1_1 - y2_2, y1_2 - y2_1))
        
        return float(np.sqrt(dx**2 + dy**2))
    
    def _get_contact_reason(self, level: str, detected: bool, iou: float, dist: float) -> str:
        if level == "L1":
            if detected:
                if iou > 0:
                    return f"Contact detected via IoU={iou:.4f}"
                else:
                    return f"Contact detected via pixel distance={dist:.2f}px"
            else:
                return f"L1 requires contact but IoU={iou:.4f}, dist={dist:.2f}px"
        elif level == "L2":
            if iou > 0:
                return f"L2 should have IoU≈0 but got {iou:.4f}"
            elif dist < 10:
                return f"L2 grazing/zero-gap: dist={dist:.2f}px"
            else:
                return f"Distance={dist:.2f}px"
        elif level == "L3":
            if dist < 50:
                return f"L3 proximity: dist={dist:.2f}px"
            else:
                return f"Distance={dist:.2f}px (may be too far for L3)"
        else:
            return "Unknown level"
    
    def predict_level_from_image(self, image_path: str, objects: List[str]) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            
            obj_a, obj_b = objects[0], objects[1] if len(objects) > 1 else "object"
            
            candidates = [
                f"{obj_a} tightly pressed against {obj_b}, physical contact",  # L1
                f"{obj_a} very close to {obj_b}, almost touching",              # L2
                f"{obj_a} near {obj_b}, within reach"                           # L3
            ]
            
            inputs = self.clip_processor(
                text=candidates,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(CONFIG["DEVICE"])
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1)[0]
            
            predicted_idx = probs.argmax().item()
            level_map = ["L1", "L2", "L3"]
            predicted_level = level_map[predicted_idx]
            
            confidence = probs[predicted_idx].item()
            
            return {
                "predicted_level": predicted_level,
                "confidence": round(float(confidence), 4),
                "probabilities": {
                    "L1": round(float(probs[0]), 4),
                    "L2": round(float(probs[1]), 4),
                    "L3": round(float(probs[2]), 4)
                }
            }
            
        except Exception as e:
            print(f"VLM Prediction failed: {e}")
            return {
                "predicted_level": "Unknown",
                "confidence": 0.0,
                "probabilities": {"L1": 0.0, "L2": 0.0, "L3": 0.0}
            }
    
    def analyze_all_models(self) -> Dict:
        print("\n" + "="*80)
        print("Starting fuzzy word grading analysis.")
        print("="*80)
        
        all_results = {}
        
        model_names = list(self.hard_results.keys())
        print(f"\nFound {len(model_names)} models: {', '.join(model_names)}")
        
        for model_name in model_names:
            print(f"\n{'='*60}")
            print(f"Analyzing model: {model_name}")
            print(f"{'='*60}")
            
            model_results = self._analyze_single_model(model_name)
            all_results[model_name] = model_results
        
        return all_results
    
    def _analyze_single_model(self, model_name: str) -> Dict:
        model_hard_results = self.hard_results[model_name]
        model_dir = os.path.join(CONFIG["IMAGES_BASE_DIR"], model_name)
        
        results = {
            "model_name": model_name,
            "sentences": [],
            "confusion_matrix_data": []  
        }
        
        for sent_idx, sent_data in enumerate(self.sentences):
            fuzzy_distance = sent_data.get("fuzzy_distance", "")
            if not fuzzy_distance:
                print(f" Sentence {sent_idx}: Missing fuzzy_distance field")
                continue
            
            ground_truth_level = self.get_fuzzy_level(fuzzy_distance)
            if ground_truth_level == "Unknown":
                print(f"Sentence {sent_idx}: Unknown fuzzy term '{fuzzy_distance}'")
                continue
            
            print(f"  Processing sentence {sent_idx}: '{fuzzy_distance}' -> {ground_truth_level}")
            
            sentence_result = {
                "sentence_idx": sent_idx,
                "sentence": sent_data["sentence"],
                "fuzzy_term": fuzzy_distance,
                "ground_truth_level": ground_truth_level,
                "objects": sent_data.get("objects", []),
                "images": []
            }
            
            hard_sent = model_hard_results[sent_idx]
            
            for img_idx, img_result in enumerate(hard_sent.get("images", [])):
                image_filename = f"sentence_{sent_idx:03d}_image_{img_idx + 1}.png"
                image_path = os.path.join(model_dir, image_filename)
                
                if not os.path.exists(image_path):
                    continue
                
                contact_check = self.check_contact_detection(
                    img_result,
                    fuzzy_distance
                )
                
                vlm_prediction = self.predict_level_from_image(
                    image_path,
                    sent_data.get("objects", [])
                )
                
                image_analysis = {
                    "image_idx": img_idx + 1,
                    "image_path": image_filename,
                    "contact_check": contact_check,
                    "vlm_prediction": vlm_prediction
                }
                
                sentence_result["images"].append(image_analysis)
                
                results["confusion_matrix_data"].append({
                    "ground_truth": ground_truth_level,
                    "predicted": vlm_prediction["predicted_level"],
                    "sentence_idx": sent_idx,
                    "image_idx": img_idx + 1
                })
            
            results["sentences"].append(sentence_result)
        
        results["statistics"] = self._compute_statistics(results)
        
        return results
    
    def _compute_statistics(self, model_results: Dict) -> Dict:
        l1_total = 0
        l1_correct = 0
        
        level_correct = 0
        level_total = 0
        
        level_stats = {
            "L1": {"total": 0, "correct": 0},
            "L2": {"total": 0, "correct": 0},
            "L3": {"total": 0, "correct": 0}
        }
        
        for sent in model_results["sentences"]:
            gt_level = sent["ground_truth_level"]
            
            for img in sent["images"]:
                contact = img["contact_check"]
                if contact["contact_expected"]:
                    l1_total += 1
                    if contact["pass_contact_check"]:
                        l1_correct += 1
                
                pred_level = img["vlm_prediction"]["predicted_level"]
                if pred_level != "Unknown":
                    level_total += 1
                    level_stats[gt_level]["total"] += 1
                    
                    if pred_level == gt_level:
                        level_correct += 1
                        level_stats[gt_level]["correct"] += 1
        
        return {
            "contact_detection": {
                "l1_total": l1_total,
                "l1_correct": l1_correct,
                "l1_accuracy": round(l1_correct / l1_total, 4) if l1_total > 0 else 0.0
            },
            "level_prediction": {
                "total": level_total,
                "correct": level_correct,
                "accuracy": round(level_correct / level_total, 4) if level_total > 0 else 0.0
            },
            "per_level_accuracy": {
                level: round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0.0
                for level, stats in level_stats.items()
            }
        }
    
    def generate_confusion_matrix(self, all_results: Dict, output_dir: str):
        print("\nGenerating confusion matrix...")
        
        levels = ["L1", "L2", "L3"]
        n_models = len(all_results)
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(all_results.items()):
            confusion_data = results["confusion_matrix_data"]
            matrix = np.zeros((3, 3), dtype=int)
            
            for item in confusion_data:
                gt = item["ground_truth"]
                pred = item["predicted"]
                
                if gt in levels and pred in levels:
                    gt_idx = levels.index(gt)
                    pred_idx = levels.index(pred)
                    matrix[gt_idx, pred_idx] += 1
            
            ax = axes[idx]
            
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix_pct = np.divide(matrix, row_sums, 
                                  out=np.zeros_like(matrix, dtype=float), 
                                  where=row_sums!=0) * 100
            
            sns.heatmap(
                matrix_pct,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                xticklabels=levels,
                yticklabels=levels,
                ax=ax,
                cbar_kws={'label': 'Percentage (%)'},
                vmin=0,
                vmax=100
            )
            
            ax.set_xlabel('Predicted Level', fontsize=12)
            ax.set_ylabel('Ground Truth Level', fontsize=12)
            ax.set_title(f'{model_name}\nAccuracy: {results["statistics"]["level_prediction"]["accuracy"]:.1%}', 
                        fontsize=13, fontweight='bold')
            
            for i in range(3):
                for j in range(3):
                    count = matrix[i, j]
                    if count > 0:
                        ax.text(j + 0.5, i + 0.7, f'(n={count})', 
                               ha='center', va='center', 
                               fontsize=9, color='gray')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Confusion matrix: {output_path}")
    
    def generate_contact_detection_report(self, all_results: Dict, output_dir: str):
        report_path = os.path.join(output_dir, "fuzzy_analysis_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Fuzzy Term Level-based Analysis Report\n")
            f.write("="*80 + "\n\n")
            
            f.write("Level Definitions:\n")
            f.write("-"*80 + "\n")
            for level, definition in CONFIG["LEVEL_NAMES"].items():
                f.write(f"  {level}: {definition}\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("Model Comparison\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Model':<20}{'L1 Contact':<15}{'Level Acc':<15}{'L1 Acc':<12}{'L2 Acc':<12}{'L3 Acc':<12}\n")
            f.write("-"*80 + "\n")
            
            for model_name, results in all_results.items():
                stats = results["statistics"]
                contact = stats["contact_detection"]
                level_acc = stats["level_prediction"]["accuracy"]
                per_level = stats["per_level_accuracy"]
                
                f.write(f"{model_name:<20}"
                       f"{contact['l1_accuracy']:<15.2%}"
                       f"{level_acc:<15.2%}"
                       f"{per_level['L1']:<12.2%}"
                       f"{per_level['L2']:<12.2%}"
                       f"{per_level['L3']:<12.2%}\n")
            
            for model_name, results in all_results.items():
                f.write(f"\n\n{'='*80}\n")
                f.write(f"Detailed Analysis: {model_name}\n")
                f.write(f"{'='*80}\n\n")
                
                stats = results["statistics"]
                
                f.write("Contact Detection (L1 Specific):\n")
                f.write("-"*40 + "\n")
                contact = stats["contact_detection"]
                f.write(f"  Total L1 images: {contact['l1_total']}\n")
                f.write(f"  Correctly detected: {contact['l1_correct']}\n")
                f.write(f"  Accuracy: {contact['l1_accuracy']:.2%}\n\n")
                
                f.write("Level Prediction:\n")
                f.write("-"*40 + "\n")
                level_pred = stats["level_prediction"]
                f.write(f"  Total predictions: {level_pred['total']}\n")
                f.write(f"  Correct predictions: {level_pred['correct']}\n")
                f.write(f"  Overall accuracy: {level_pred['accuracy']:.2%}\n\n")
                
                f.write("Per-Level Accuracy:\n")
                for level, acc in stats["per_level_accuracy"].items():
                    level_name = CONFIG["LEVEL_NAMES"][level]
                    f.write(f"  {level} ({level_name}): {acc:.2%}\n")
        
        print(f"  ✓ Analysis report: {report_path}")
    
    def generate_visualizations(self, all_results: Dict, output_dir: str):
        
        print("\nGenerating visualizations...")
        
        self.generate_confusion_matrix(all_results, output_dir)
        
        self._plot_contact_detection(all_results, output_dir)
        
        self._plot_level_accuracy(all_results, output_dir)
        
        print("Visualizations completed")
    
    def _plot_contact_detection(self, all_results: Dict, output_dir: str):
        model_names = []
        accuracies = []
        
        for model_name, results in all_results.items():
            model_names.append(model_name)
            acc = results["statistics"]["contact_detection"]["l1_accuracy"]
            accuracies.append(acc)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(model_names, accuracies, color='steelblue', alpha=0.7)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1%}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Contact Detection Accuracy', fontsize=12)
        ax.set_title('L1 Contact Detection Performance\n(IoU > 0 OR Pixel Distance < 5px)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% Target')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "contact_detection_accuracy.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Contact detection chart: {output_path}")
    
    def _plot_level_accuracy(self, all_results: Dict, output_dir: str):
        
        model_names = list(all_results.keys())
        levels = ["L1", "L2", "L3"]
        
        data = {level: [] for level in levels}
        for model_name in model_names:
            per_level = all_results[model_name]["statistics"]["per_level_accuracy"]
            for level in levels:
                data[level].append(per_level[level])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(model_names))
        width = 0.25
        
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        
        for idx, level in enumerate(levels):
            offset = (idx - 1) * width
            bars = ax.bar(x + offset, data[level], width, 
                         label=f'{level} ({CONFIG["LEVEL_NAMES"][level]})',
                         color=colors[idx], alpha=0.8)
            
            for bar, val in zip(bars, data[level]):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.0%}',
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Level Prediction Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "per_level_accuracy.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Per-level accuracy chart: {output_path}")


def main():
    
    print("="*80)
    print("Fuzzy Term Level-based Analysis System")
    print("="*80)
    
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    
    analyzer = FuzzyTermAnalyzer()
    
    analyzer.load_fuzzy_mapping(CONFIG["FUZZY_TERM_MAPPING"])
    
    analyzer.load_evaluation_results(
        CONFIG["HARD_EVAL_RESULTS"],
        CONFIG["SELECTED_SENTENCES"]
    )
    
    all_results = analyzer.analyze_all_models()
    
    results_path = os.path.join(CONFIG["OUTPUT_DIR"], "fuzzy_analysis_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Detailed results saved: {results_path}")
    
    analyzer.generate_contact_detection_report(all_results, CONFIG["OUTPUT_DIR"])
    
    analyzer.generate_visualizations(all_results, CONFIG["OUTPUT_DIR"])
    
    print("\n" + "="*80)
    print("✅ Fuzzy Term Analysis Completed!")
    print("="*80)
    print(f"\nOutput Directory: {CONFIG['OUTPUT_DIR']}")
    print("  - fuzzy_analysis_results.json (Detailed Results)")
    print("  - fuzzy_analysis_report.txt (Analysis Report)")
    print("  - confusion_matrix.png (Confusion Matrix)")
    print("  - contact_detection_accuracy.png (Contact Detection)")
    print("  - per_level_accuracy.png (Per-Level Accuracy)")


if __name__ == "__main__":
    main()
