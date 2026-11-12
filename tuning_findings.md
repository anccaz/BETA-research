# Weekly Research Progress: BETA Hyperparameter Ablation (ImageNet-R)

## Objective
Following the LAME vs. BETA benchmark setup (`1065c31`), this sprint configures hyperparameter calibration for BETA under distribution shift (**ImageNet-R**, 200 classes, 100 adaptation steps).

## Experimental Matrix

| Configuration | alpha | margin_e0 | bvr_lr | Hypothesis |
| :--- | :---: | :---: | :---: | :--- |
| **BETA (Default)** | 0.40 | 0.90 | 0.0100 | Standard benchmark baseline |
| **BETA (Tuned)** | 0.20 | 0.95 | 0.0005 | De-weights miscalibrated logits, prunes high-entropy samples, stabilizes prompt drift |

## Execution Status
- Configured ImageNet-R benchmark data pipeline (`/work/axl220095/datasets/imagenet-r`) with offline `wandb` logging (`WANDB_MODE=disabled`).
- Queued automated ablation pipeline (`run_beta_tuning.slurm`) on Juno `a30` GPU partition.
