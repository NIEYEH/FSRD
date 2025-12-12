## Quantitative Evaluations

To evaluate the performance on our dataset, *Learning from Ambiguity: A Fuzzy Spatial Relationship Dataset for Human-Aligned Text-to-Image Generation*, we employed the following quantitative metrics and their corresponding official implementations.

| Evaluation Metric | Implementation Source | Description |
| :--- | :--- | :--- |
| **VISOR** | [Official Repo](https://github.com/microsoft/VISOR) | Metric for evaluating spatial relationship consistency |
| **T2I-CompBench** | [Official Repo](https://github.com/Karine-Huang/T2I-CompBench) | Benchmark for open-world compositional text-to-image generation |
| **T2I-CompBench++** | [Official Repo](https://github.com/Karine-Huang/T2I-CompBench) | Enhanced benchmark with more complex compositional prompts |
| **ZS-FID** | [pytorch-fid](https://github.com/mseitzer/pytorch-fid) | Zero-Shot Fréchet Inception Distance (assessing image fidelity) |
| **CMMD** | [Google Research](https://github.com/google-research/google-research/tree/master/cmmd) | CLIP-based Maximum Mean Discrepancy metric |
| **NR-IQA** | [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) | Non-Reference Image Quality Assessment (e.g., MUSIQ, CLIPIQA) |
| **Attention Maps** | [Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite) | Visualization and control of cross-attention maps |
