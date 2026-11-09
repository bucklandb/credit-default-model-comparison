# Credit-default model comparison

A portfolio adaptation of Benjamin Buckland's AAI6610 credit-default modeling work. Original coursework remains unchanged. See the provenance section for references and AI assistance.

## What this shows
Compare logistic regression, decision trees, random forests and an optional neural-network experiment on the UCI Default of Credit Card Clients dataset. Study imbalance, precision/recall tradeoffs, cross-validation, regularization and threshold tuning rather than treating accuracy as the only objective.

The notebook retains the original modeling implementation, with outputs cleared and evaluation language corrected. The separate command-line runner reproduces the three baseline statistical pipelines and the original 24-setting logistic grid and threshold tuning, without requiring TensorFlow. It is a portfolio refactor, not a claim of a new independent experiment.

## Setup
Use Python 3.13 in a fresh virtual environment. Install `requirements.txt` and, to execute the full notebook, `requirements-neural.txt`. Exact pins reflect the inspected local environment; cross-platform installation and TensorFlow hardware behavior are not guaranteed.

    python -m venv .venv
    # Windows PowerShell
    .venv\Scripts\Activate.ps1
    # macOS/Linux
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v

## Data input
Download the ZIP from the official [UCI dataset page](https://doi.org/10.24432/C55S3H), under CC BY 4.0. Keep it outside this repository directory. The code reads `default of credit card clients.xls` from that archive. No data is bundled and no download occurs automatically.

    python credit_default.py --data-zip "C:\path\default+of+credit+card+clients.zip" --output results.json
    python credit_default.py --data-zip "C:\path\default+of+credit+card+clients.zip" --tune --output tuned-results.json

For the notebook, set CREDIT_DEFAULT_DATA_ZIP to the absolute ZIP path in the launching environment, then open notebooks/credit_default_comparison.ipynb with Jupyter. Install Jupyter separately if needed. The notebook includes all original neural experiments and is substantially more expensive than the baseline runner.

Expected source ZIP SHA256: `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`. A mismatch raises an error; review an official source change before updating the expected digest. Input shape and labels are checked. Models use the original 80/20 stratified split and seed 20.

## Evaluation limits
The same test set was examined across earlier coursework. Test results also informed the later choice to tune logistic regression. Therefore it is a reused exploratory holdout, **not an untouched unbiased final test**. A new nested evaluation or genuinely untouched external set is needed before claiming generalization after selection.

The neural architecture was selected before its subsequent CV; that CV is not nested. Its categories are taken from an earlier broader training subset. Its fitting sample sizes differ from those of the statistical models. Retained code documents the original experiment; these limitations are not silently rewritten away. Seed settings do not guarantee bitwise replication across platforms.

The statistical runner was executed on the complete dataset on November 9, 2026. It reproduced logistic F1 0.4663, balanced grid-search F1 0.5234 and threshold-tuned F1 0.5240. See VERIFICATION.md for the exact tested scope. Neural results and the complete notebook have not been rerun. GitHub Actions installs the pinned statistical dependencies, runs the six synthetic unit tests, checks CLI help, and parses the notebook code. It does not run the full dataset or neural experiments. Runtime timings are hardware-dependent.

This 2005 Taiwan dataset has population/time limitations. Sex, age, education and marital status are predictors. No fairness, calibration, business-cost or deployment validation is claimed. This is educational analysis, not an automated lending system.

## Provenance and assistance
Derived from Benjamin Buckland's Module 8 notebook and the Module 3/5 experiment lineage. The Module 3 disclosure says Codex assisted with concepts, code generation/troubleshooting, interpretation, editing, citations and drafting the evaluation section; Benjamin wrote initial drafts of several narrative sections and reviewed the results. This refactor was also AI-assisted. Do not describe the work as wholly unaided authorship.

Public libraries are imported, not vendored. Course prompts, rubrics, textbook notebooks and the research-paper PDF are excluded. Benjamin confirms that no instructor code was used in this project and all sources used are already referenced. His reinforcement-learning assignment is the separate exception containing instructor code; it is not included here. 

## References and rights
- Yeh, I.-C. (2009). Default of Credit Card Clients. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. Dataset license: https://creativecommons.org/licenses/by/4.0/
- Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. https://doi.org/10.1016/j.eswa.2007.12.020
- Geron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd ed. O'Reilly. Original notebooks cite this book for methods; Benjamin confirms all sources used are already referenced.
- Kavlakoglu, E. (2021). What is random forest? IBM. https://www.ibm.com/think/topics/random-forest
