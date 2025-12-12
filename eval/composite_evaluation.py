import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple
from pathlib import Path

matplotlib.use('Agg')

CONFIG = {
    "HARD_EVAL_RESULTS": "",
    "SOFT_EVAL_RESULTS": "",
    
    "OUTPUT_DIR": "",
    
    "ALPHA": 0.6,  # Weighting of hard indicators (0.6 means 60% weight is given to hard indicators, and 40% to soft indicators)
    
    "PASS_THRESHOLD": 10,  
}


class CompositeEvaluator:
    
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.hard_results = None
        self.soft_results = None
        
    def load_results(self, hard_path: str, soft_path: str):    
        print("Loading evaluation results...")
        
        with open(hard_path, 'r', encoding='utf-8') as f:
            self.hard_results = json.load(f)
        print(f"✓ Hard indicator results: {hard_path}")
        
        with open(soft_path, 'r', encoding='utf-8') as f:
            self.soft_results = json.load(f)
        print(f"✓ Soft indicator results: {soft_path}")
        
    def compute_hard_score(self, hard_eval: Dict) -> float:
        scores = hard_eval.get("scores", {})
        total = sum([
            scores.get("quantity", 0),
            scores.get("category", 0),
            scores.get("direction", 0),
            scores.get("distance", 0),
            scores.get("scene", 0)
        ])
        return float(total)
    
    def compute_soft_score(self, soft_eval: Dict) -> float:
        return float(soft_eval.get("clip_score", 0))
    
    def compute_composite_score(self, hard_score: float, soft_score: float) -> Dict:
        s_hard = hard_score / 5.0
        s_soft = (soft_score - 1) / 4.0 
        
        score_final = self.alpha * s_hard + (1 - self.alpha) * s_soft
        
        return {
            "s_hard_normalized": round(s_hard, 4),
            "s_soft_normalized": round(s_soft, 4),
            "score_final": round(score_final, 4),
            "score_final_scaled": round(score_final * 15, 2) 
        }
    
    def compute_consistency_cv(self, scores: List[float]) -> Dict:
        if len(scores) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "cv": 0.0,
                "consistency": 0.0
            }
        
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        
        if mean_score == 0:
            cv = 0.0
            consistency = 0.0
        else:
            cv = std_score / mean_score
            consistency = max(0.0, 1.0 - cv)
        
        return {
            "mean": round(float(mean_score), 4),
            "std": round(float(std_score), 4),
            "cv": round(float(cv), 4),
            "consistency": round(float(consistency), 4)
        }
    
    def compute_consistency_pass_rate(self, scores: List[float], threshold: float) -> Dict:
        if len(scores) == 0:
            return {
                "min_score": 0.0,
                "all_pass": False,
                "pass_count": 0,
                "pass_rate": 0.0
            }
        
        min_score = min(scores)
        all_pass = min_score >= threshold
        pass_count = sum(1 for s in scores if s >= threshold)
        pass_rate = pass_count / len(scores)
        
        return {
            "min_score": round(float(min_score), 2),
            "all_pass": bool(all_pass),
            "pass_count": int(pass_count),
            "pass_rate": round(float(pass_rate), 4)
        }
    
    def evaluate_model(self, model_name: str) -> Dict:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")
        
        hard_model_results = self.hard_results.get(model_name, [])
        soft_model_results = self.soft_results.get(model_name, [])
        
        if not hard_model_results or not soft_model_results:
            print(f"Assessment results are missing.")
            return {}
        
        model_evaluation = {
            "model_name": model_name,
            "alpha": self.alpha,
            "sentences": []
        }
        
        for sent_idx in range(len(hard_model_results)):
            hard_sent = hard_model_results[sent_idx]
            soft_sent = soft_model_results[sent_idx]
            
            sentence_eval = {
                "sentence_idx": sent_idx,
                "sentence": hard_sent.get("sentence", ""),
                "images": [],
                "composite_scores": [], 
            }
            
            hard_images = hard_sent.get("images", [])
            soft_images = soft_sent.get("images", [])
            
            for img_idx in range(min(len(hard_images), len(soft_images))):
                hard_img = hard_images[img_idx]
                soft_img = soft_images[img_idx]
                
                hard_score = self.compute_hard_score(hard_img)
                soft_score = self.compute_soft_score(soft_img)
                
                composite = self.compute_composite_score(hard_score, soft_score)
                
                image_eval = {
                    "image_idx": img_idx + 1,
                    "hard_score": hard_score,
                    "soft_score": soft_score,
                    **composite
                }
                
                sentence_eval["images"].append(image_eval)
                sentence_eval["composite_scores"].append(composite["score_final_scaled"])
            
            composite_scores = sentence_eval["composite_scores"]
            
            cv_metrics = self.compute_consistency_cv(composite_scores)
            sentence_eval["consistency_cv"] = cv_metrics
            
            pass_metrics = self.compute_consistency_pass_rate(
                composite_scores, 
                CONFIG["PASS_THRESHOLD"]
            )
            sentence_eval["consistency_pass"] = pass_metrics
            
            model_evaluation["sentences"].append(sentence_eval)
        
        model_evaluation["statistics"] = self.compute_model_statistics(model_evaluation)
        
        return model_evaluation
    
    def compute_model_statistics(self, model_eval: Dict) -> Dict:
        all_composite_scores = []
        all_hard_scores = []
        all_soft_scores = []
        all_consistency_cv = []
        all_consistency_pass = []
        sentence_pass_count = 0
        
        for sent in model_eval["sentences"]:
            for img in sent["images"]:
                all_composite_scores.append(img["score_final_scaled"])
                all_hard_scores.append(img["hard_score"])
                all_soft_scores.append(img["soft_score"])
            
            all_consistency_cv.append(sent["consistency_cv"]["consistency"])
            
            if sent["consistency_pass"]["all_pass"]:
                sentence_pass_count += 1
        
        total_sentences = len(model_eval["sentences"])
        
        return {
            "total_sentences": total_sentences,
            "total_images": len(all_composite_scores),
            
            "avg_composite_score": round(float(np.mean(all_composite_scores)), 2),
            "avg_hard_score": round(float(np.mean(all_hard_scores)), 2),
            "avg_soft_score": round(float(np.mean(all_soft_scores)), 2),
            
            "std_composite_score": round(float(np.std(all_composite_scores)), 2),
            "std_hard_score": round(float(np.std(all_hard_scores)), 2),
            "std_soft_score": round(float(np.std(all_soft_scores)), 2),
            
            "avg_consistency_cv": round(float(np.mean(all_consistency_cv)), 4),
            "sentence_pass_count": sentence_pass_count,
            "sentence_pass_rate": round(sentence_pass_count / total_sentences, 4) if total_sentences > 0 else 0.0,
        }
    
    def evaluate_all_models(self) -> Dict[str, Dict]:
        all_evaluations = {}
        
        model_names = set(self.hard_results.keys()) & set(self.soft_results.keys())
        
        print(f"\nFound {len(model_names)} models: {', '.join(model_names)}")
        
        for model_name in sorted(model_names):
            evaluation = self.evaluate_model(model_name)
            if evaluation:
                all_evaluations[model_name] = evaluation
        
        return all_evaluations
    
    def generate_comparison_report(self, all_evaluations: Dict, output_dir: str):
        report_path = os.path.join(output_dir, "composite_evaluation_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Composite Scoring System - Model Comparison Report\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Scoring Formula:\n")
            f.write(f"  Score_final = α × S_hard + (1-α) × S_soft\n")
            f.write(f"  Where α = {self.alpha} (weight for hard metrics)\n\n")
            
            f.write(f"Consistency Metrics:\n")
            f.write(f"  A. Coefficient of Variation: Consistency = max(0, 1 - σ/μ)\n")
            f.write(f"  B. Pass Rate: All 3 images must have scores ≥ {CONFIG['PASS_THRESHOLD']} to pass\n\n")
            
            f.write("="*80 + "\n")
            f.write("Model Composite Score Ranking\n")
            f.write("="*80 + "\n\n")
            
            sorted_models = sorted(
                all_evaluations.items(),
                key=lambda x: x[1]["statistics"]["avg_composite_score"],
                reverse=True
            )
            
            f.write(f"{'Rank':<6}{'Model':<15}{'Composite Score':<12}{'Hard Score':<10}{'Soft Score':<10}"
                   f"{'Consistency (CV)':<15}{'Pass Rate':<10}\n")
            f.write("-"*80 + "\n")
            
            for rank, (model_name, eval_data) in enumerate(sorted_models, 1):
                stats = eval_data["statistics"]
                f.write(f"{rank:<6}{model_name:<15}"
                       f"{stats['avg_composite_score']:<12.2f}"
                       f"{stats['avg_hard_score']:<10.2f}"
                       f"{stats['avg_soft_score']:<10.2f}"
                       f"{stats['avg_consistency_cv']:<15.4f}"
                       f"{stats['sentence_pass_rate']:<10.2%}\n")
            
            for model_name, eval_data in sorted_models:
                f.write(f"\n\n{'='*80}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"{'='*80}\n\n")
                
                stats = eval_data["statistics"]
                
                f.write("Overall Statistics:\n")
                f.write("-"*40 + "\n")
                f.write(f"  Total Sentences: {stats['total_sentences']}\n")
                f.write(f"  Total Images: {stats['total_images']}\n\n")
                
                f.write("Average Scores:\n")
                f.write(f"  Overall score: {stats['avg_composite_score']:.2f} ± {stats['std_composite_score']:.2f}\n")
                f.write(f"  Hard Score: {stats['avg_hard_score']:.2f} ± {stats['std_hard_score']:.2f} \n")
                f.write(f"  Soft Score: {stats['avg_soft_score']:.2f} ± {stats['std_soft_score']:.2f} \n\n")
                
                f.write("Consistency indicators:\n")
                f.write(f"  Average Consistency (CV): {stats['avg_consistency_cv']:.4f}\n")
                f.write(f"  Sentences Passed: {stats['sentence_pass_count']}/{stats['total_sentences']}\n")
                f.write(f"  Pass Rate: {stats['sentence_pass_rate']:.2%}\n\n")
                
                f.write("Detailed Scores for Each Sentence:\n")
                f.write("-"*40 + "\n")
                
                for sent in eval_data["sentences"]:
                    f.write(f"\nSentence {sent['sentence_idx']}: {sent['sentence']}\n")
                    
                    for img in sent["images"]:
                        f.write(f"  Image {img['image_idx']}: "
                               f"Composite={img['score_final_scaled']:.2f}, "
                               f"Hard={img['hard_score']:.1f}, "
                               f"Soft={img['soft_score']:.1f}\n")
                    
                    cv = sent["consistency_cv"]
                    pass_info = sent["consistency_pass"]
                    
                    f.write(f"  Consistency: CV={cv['consistency']:.4f} "
                           f"(μ={cv['mean']:.2f}, σ={cv['std']:.2f})\n")
                    f.write(f"  Pass Status: {'✓ Passed' if pass_info['all_pass'] else '✗ Failed'} "
                           f"(Min Score={pass_info['min_score']:.2f}, "
                           f"Pass Count={pass_info['pass_count']}/3)\n")
        
        print(f"\nThe comparison report has been saved: {report_path}")
    
    def generate_visualizations(self, all_evaluations: Dict, output_dir: str):
        print("\nGenerating visualizations...")
        self._plot_composite_scores(all_evaluations, output_dir)
        self._plot_boxplots(all_evaluations, output_dir)
        self._plot_hard_vs_soft(all_evaluations, output_dir)
        self._plot_consistency_metrics(all_evaluations, output_dir)
        print("✓ Visualizations generated successfully")
    
    def _plot_composite_scores(self, all_evaluations: Dict, output_dir: str):
        model_names = []
        composite_scores = []
        hard_scores = []
        soft_scores = []
        
        for model_name, eval_data in all_evaluations.items():
            stats = eval_data["statistics"]
            model_names.append(model_name)
            composite_scores.append(stats["avg_composite_score"])
            hard_scores.append(stats["avg_hard_score"])
            soft_scores.append(stats["avg_soft_score"])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(model_names))
        width = 0.25
        
        ax.bar(x - width, hard_scores, width, label='Hard Score', color='steelblue')
        ax.bar(x, soft_scores, width, label='Soft Score (CLIP)', color='coral')
        ax.bar(x + width, composite_scores, width, label='Composite Score', color='seagreen')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Composite Score Comparison (α={self.alpha})', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "composite_scores_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Composite scores comparison: {output_path}")
    
    def _plot_boxplots(self, all_evaluations: Dict, output_dir: str):
        data_by_model = {}
        
        for model_name, eval_data in all_evaluations.items():
            scores = []
            for sent in eval_data["sentences"]:
                for img in sent["images"]:
                    scores.append(img["score_final_scaled"])
            data_by_model[model_name] = scores
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        positions = range(1, len(data_by_model) + 1)
        bp = ax.boxplot(
            data_by_model.values(),
            labels=data_by_model.keys(),
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=8)
        )
        
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow']):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Composite Score (out of 15)', fontsize=12)
        ax.set_title('Score Distribution Boxplot (Shorter box = Higher consistency)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        ax.axhline(y=CONFIG["PASS_THRESHOLD"], color='r', linestyle='--', 
                   label=f'Pass Threshold ({CONFIG["PASS_THRESHOLD"]} pts)')
        ax.legend()
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "score_distribution_boxplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Boxplot: {output_path}")
    
    def _plot_hard_vs_soft(self, all_evaluations: Dict, output_dir: str):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['steelblue', 'coral', 'seagreen', 'purple']
        
        for idx, (model_name, eval_data) in enumerate(all_evaluations.items()):
            hard_scores = []
            soft_scores = []
            
            for sent in eval_data["sentences"]:
                for img in sent["images"]:
                    hard_scores.append(img["hard_score"])
                    soft_scores.append(img["soft_score"])
            
            ax.scatter(hard_scores, soft_scores, 
                      label=model_name, 
                      alpha=0.6, 
                      s=100,
                      color=colors[idx % len(colors)])
        
        ax.set_xlabel('Hard Score (0-5)', fontsize=12)
        ax.set_ylabel('Soft Score (1-5)', fontsize=12)
        ax.set_title('Hard vs Soft Score Scatter Plot', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add diagonal reference line
        ax.plot([0, 5], [1, 5], 'k--', alpha=0.3, label='Ideal Alignment')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "hard_vs_soft_scatter.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Scatter plot: {output_path}")
    
    def _plot_consistency_metrics(self, all_evaluations: Dict, output_dir: str):
        model_names = []
        cv_consistency = []
        pass_rates = []
        
        for model_name, eval_data in all_evaluations.items():
            stats = eval_data["statistics"]
            model_names.append(model_name)
            cv_consistency.append(stats["avg_consistency_cv"])
            pass_rates.append(stats["sentence_pass_rate"])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.bar(model_names, cv_consistency, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Consistency Score (CV)', fontsize=12)
        ax1.set_title('CV-based Consistency (Higher is better)', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(cv_consistency):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        ax2.bar(model_names, pass_rates, color='seagreen', alpha=0.7)
        ax2.set_ylabel('All-Pass Rate', fontsize=12)
        ax2.set_title(f'All-Pass Rate (All 3 images ≥ {CONFIG["PASS_THRESHOLD"]} pts)', 
                     fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(pass_rates):
            ax2.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "consistency_metrics_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Consistency metrics: {output_path}")


def main():
    print("="*80)
    print("Composite Scoring System")
    print("="*80)
    
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    
    evaluator = CompositeEvaluator(alpha=CONFIG["ALPHA"])
    
    evaluator.load_results(
        CONFIG["HARD_EVAL_RESULTS"],
        CONFIG["SOFT_EVAL_RESULTS"]
    )
    
    print("\nStarting composite evaluation...")
    all_evaluations = evaluator.evaluate_all_models()
    
    results_path = os.path.join(CONFIG["OUTPUT_DIR"], "composite_evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_evaluations, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Detailed results saved: {results_path}")
    
    evaluator.generate_comparison_report(all_evaluations, CONFIG["OUTPUT_DIR"])
    
    evaluator.generate_visualizations(all_evaluations, CONFIG["OUTPUT_DIR"])
    
    print("\n" + "="*80)
    print("Composite evaluation completed!")
    print("="*80)
    print(f"\nOutput directory: {CONFIG['OUTPUT_DIR']}")
    print("  - composite_evaluation_results.json ")
    print("  - composite_evaluation_report.txt ")
    print("  - *.png")


if __name__ == "__main__":
    main()
