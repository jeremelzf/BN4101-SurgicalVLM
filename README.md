# BN4101-SurgicalVLM

**Author:** Jereme Lee  
**Module:** BN4101 Final Year Project, NUS Biomedical Engineering  
**Academic Year:** 2025/2026  

This repository contains the project-specific scripts developed for fine-tuning and evaluating a Vision-Language Model (Qwen3-VL-8B-Instruct) on the GraSP dataset for holistic surgical scene understanding of robot-assisted radical prostatectomies.

> **Note:** Model training was conducted using 
> [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). 
> This repository contains only the original project code; 
> LLaMA-Factory must be set up separately as a dependency.

---

## Repository Structure

```
BN4101-SurgicalVLM/
├── eval/
│   ├── eval_metrics.py       # GraSP evaluation pipeline
│   └── grasp_eval.yaml       # Evaluation configuration
├── data_prep/
│   └── convert_grasp_test-2.py       # Test set conversion & stratified sampling
├── training/
│   ├── grasp_sft.yaml       # LoRA fine-tuning hyperparameters
│   └── grasp_job.pbs       # NUS HPC job submission script
└── README.md
```

---

## Script Descriptions

### `data_prep/convert_grasp_test.py`
Converts the raw GraSP test annotations into the ShareGPT-format `.jsonl` file required by LLaMA-Factory for inference.

Key features:
- Merges long-term (1 fps, phase/step) and short-term (35s keyframe, instrument/action) annotation JSONs
- Stratified sampling: ~23% of frames per phase (min. 1) for balanced evaluation
- Outputs one JSONL entry per task per frame

Output: GraSP_test.jsonl

### `eval/eval_metrics.py`
Evaluates model predictions across all four GraSP tasks:
- **Phase & Step Recognition** — single-label; reports Accuracy, Macro F1, mAP
- **Instrument Segmentation** — multi-label; reports Exact Match, Macro F1, mAP
- **Atomic Action Detection** — multi-label; same metrics as above

Key features:
- Two-stage normalisation: synonym expansion → GraSP co-occurrence constraint pruning
- Constraint rules derived from Ayobi et al. (2024), Figs C.13–C.16
- Accepts `.jsonl` or `.tsv` prediction files
- Outputs a summary JSON (`eval_results.json`)

Output: eval_results.json

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model inference |
| `transformers` | Qwen3-VL model loading |
| `scikit-learn` | Evaluation metrics (F1, mAP) |
| `numpy` | Numerical operations |
| `llamafactory` | Training & inference CLI |

---

## Dataset

This project uses the **GraSP dataset** (Ayobi et al., 2024).  
Access and further details: https://github.com/BCV-Uniandes/GraSP
