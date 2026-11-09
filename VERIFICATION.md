# Verification — November 9, 2026

Local verification of the statistical workflow; neural experiments remain unverified.

- Six unittest tests passed: category validation, input-copy/mapping behavior, duplicate-ID rejection, schema rejection, hand-calculated metrics, and a fitted synthetic pipeline that handles an unseen category.
- All Python source and all notebook code cells passed AST syntax parsing.
- Every code cell in the derivative notebook has empty outputs and null execution count.
- Actual source ZIP passed the recorded SHA256 and 30,000-row, 25-column schema checks.
- Full `run(frame, tune=True)` completed with exit code 0: five-fold cross-validation of three statistical models; 24-setting/five-fold logistic grid; threshold selection; and 6,000-row exploratory holdout evaluation.
- Selected grid settings: C=10, balanced class weights, L2 (l1_ratio=0), lbfgs.
- Selected threshold: 0.24266113877086257.

## Newly reproduced exploratory-holdout results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic | 0.820333 | 0.679654 | 0.354936 | 0.466337 | 0.757473 |
| Depth-5 tree | 0.816667 | 0.699473 | 0.299925 | 0.419831 | 0.736693 |
| Random forest | 0.816000 | 0.652531 | 0.359457 | 0.463557 | 0.753989 |
| Grid-tuned logistic | 0.774167 | 0.490765 | 0.560663 | 0.523391 | 0.757389 |
| Threshold-tuned logistic | 0.785333 | 0.514141 | 0.534288 | 0.524021 | 0.757473 |

## Not verified

Full notebook execution, neural retraining, bootstrap intervals and coefficient plots were not rerun. A clean virtual-environment installation was not performed; exact pins were inspected and execution used existing installed packages. CLI --help exited 0, and missing required --data-zip exited 2 with the expected validation message. The expensive model workflow was called through its Python API. No claim of unbiased post-selection evaluation or deployment readiness is made.

No original coursework was changed.

## User provenance clarification
Benjamin confirms no instructor code was used in this project and all sources used are referenced. Instructor code belongs only to the separate reinforcement-learning assignment, excluded from this candidate.
