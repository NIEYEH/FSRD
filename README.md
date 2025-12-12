# FSRD: A Large-Scale Dataset for Fuzzy Spatial Relationship Understanding

<div align="center">
  
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)]
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green)](LICENSE)

**[Tianjiao Liang, Qinlong Li, Honggang Qi]**

</div>

---

> [cite_start]**Abstract:** Recent text-to-image (T2I) diffusion models achieve impressive visual fidelity but consistently struggle with interpreting and generating images from natural language prompts containing fuzzy spatial relationships (e.g., "somewhere to the right," "fairly close"). This gap arises because standard training datasets predominantly feature precise spatial descriptions, neglecting the graded, context-dependent nature of human spatial language. To address this, we introduce the Fuzzy Spatial Relationship Dataset (FSRD), a large-scale vision-language corpus designed to bridge this divide. FSRD comprises approximately eight million images from five public sources, each paired with synthetically generated captions that replace crisp spatial predicates with calibrated vague expressions. We detail an automated pipeline for constructing these captions, which involves a high-recall multi-detector ensemble for object localisation and a two-stage captioning process using vision-language models to generate globally coherent and pairwise fuzzy spatial descriptions. To evaluate model performance under fuzzy instructions, we propose a novel Spatial Fuzziness Metric (SFM) that combines deterministic geometric verification with semantic alignment scoring. Fine-tuning Stable Diffusion 2.1 on FSRD demonstrates consistent and substantial improvements: our model achieves a state-of-the-art VISOR accuracy of 68.0% on the SR2D benchmark and superior performance on the spatial subset of T2I-CompBench. Qualitative analyses further confirm its enhanced ability to faithfully render instructions like "roughly above" or "very close to." By explicitly modelling spatial vagueness, FSRD advances T2I generation towards more robust and human-aligned spatial understanding.

---

## 🔥 Highlights

- [cite_start]**📚 Large-Scale:** Roughly **8 million images** sourced from COCO, CC12M, SA-1B, etc., re-captioned with rich fuzzy spatial semantics.
- [cite_start]**🧠 Automated Pipeline:** A robust two-stage framework combining ensemble detection (Grounding DINO, OWL-ViT) and VLM-based fuzzy captioning.
- [cite_start]**📏 New Metric:** **Spatial Fuzziness Metric (SFM)**, combining deterministic geometric verification with semantic alignment to evaluate compliance with vague instructions.
- [cite_start]**🚀 SOTA Performance:** Fine-tuning SD 2.1 on FSRD achieves state-of-the-art results on the **SR2D benchmark** and **T2I-CompBench**.

---

## 🏗️ The Pipeline

[cite_start]Our fully automated construction pipeline ensembles detection modules for high-recall detection, followed by a two-stage captioning process to produce fluent spatial descriptions with controllable fuzziness.

---

## 📂 Dataset Access

The FSRD dataset contains images paired with global scene descriptions and structured pairwise relations.

### Download
- **HuggingFace:** [Link to your HF repo]

