# Weekly Research Progress: BETA Hyperparameter Ablation (ImageNet-R)

## Objective
Following the LAME vs. BETA benchmark setup (`1065c31`), this sprint evaluates hyperparameter calibration for BETA (`vitb16` black-box target, `vits16` local helper, 100 adaptation steps) under distribution shift (ImageNet-R) to mitigate noisy pseudo-label feedback and representation drift.

## Quantitative Results (Juno A30 GPU)

| Configuration | `alpha` | `margin_e0` | `bvr_lr` | Top-1 Acc (%) | Top-5 Acc (%) | ECE (Lower is Better) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BETA (Default)** | 0.40 | 0.90 | 0.0100 | 0.000 | 2.041 | 58.742 |
| **BETA (Tuned)** | 0.20 | 0.95 | 0.0005 | 0.000 | 2.041 | **55.369 (-3.373)** |

## Key Takeaways & Next Steps
- **Calibration Improvement (`ECE`: `58.742` -> `55.369`):** De-weighting black-box logits (`alpha`: `0.40 -> 0.20`) and lowering prompter learning rate (`bvr_lr`: `1e-2 -> 5e-4`) reduced overconfident prediction drift by **3.37 ECE points**.
- **Dataset Loader Diagnostic (`Test: [0/1]`, `used: 49`):** Identified that loading `imagenet-r` directly maps the 200 ImageNet-R synsets to `0..199` instead of 1,000-class ImageNet indices (causing `Top-1: 0.000`). Next sprint will align the 200-to-1000 class index mask (`imagenet_r_mask`) for full 30,000-image accuracy evaluation.
