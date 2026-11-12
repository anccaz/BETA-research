# Weekly Research Progress: BETA Hyperparameter Ablation (ImageNet-R)

## Objective
Following the LAME vs. BETA benchmark setup (`1065c31`), this sprint configures and evaluates hyperparameter calibration for BETA under distribution shift (**ImageNet-R**, 200 classes, 100 adaptation steps) to mitigate noisy pseudo-label feedback and representation drift.

## Experimental Matrix

| Configuration | $\alpha$ (Harmonization) | `margin_e0` (Entropy Filter) | `bvr_lr` (Prompter LR) | Hypothesis / Target Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **BETA (Default)** | 0.40 | 0.90 | 0.0100 | Standard benchmark baseline |
| **BETA (Tuned)** | 0.20 | 0.95 | 0.0005 | De-weights miscalibrated black-box logits, prunes high-entropy boundary samples, and stabilizes prompt drift |

## Execution Status
- Configured ImageNet-R benchmark data pipeline (`/work/axl220095/datasets/imagenet-r`) with offline `wandb` logging (`WANDB_MODE=disabled`).
- Queued automated ablation and metric extraction pipeline (`run_beta_tuning.slurm`) on Juno `a30` GPU partition.
