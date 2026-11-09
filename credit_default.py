"""Educational portfolio refactor; never an untouched final test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, TunedThresholdClassifierCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 20
TARGET = "default payment next month"
ZIP_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
EDUCATION = {0:"Unknown", 1:"Graduate school", 2:"University", 3:"High school", 4:"Other", 5:"Unknown", 6:"Unknown"}
MARRIAGE = {0:"Unknown", 1:"Married", 2:"Single", 3:"Other"}


def clean_data(frame):
    """Validate input and retain the original categorical mapping."""
    required = {"ID", TARGET, "LIMIT_BAL", "AGE", *CATEGORICAL,
                *[f"BILL_AMT{i}" for i in range(1,7)], *[f"PAY_AMT{i}" for i in range(1,7)]}
    if set(frame.columns) != required:
        raise ValueError("Dataset columns do not match the expected UCI schema")
    if frame.isna().any().any() or frame.ID.duplicated().any():
        raise ValueError("Missing values or duplicate client IDs require review")
    if set(frame[TARGET].unique()) != {0,1}:
        raise ValueError("Expected both binary target classes")
    if not frame.EDUCATION.isin(EDUCATION).all() or not frame.MARRIAGE.isin(MARRIAGE).all():
        raise ValueError("Unexpected education/marriage categories")
    result = frame.copy()
    result["EDUCATION"] = result.EDUCATION.map(EDUCATION)
    result["MARRIAGE"] = result.MARRIAGE.map(MARRIAGE)
    return result


def load_data(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ZIP_SHA256:
        raise ValueError("Dataset ZIP checksum differs; verify official provenance before proceeding")
    with ZipFile(path) as archive:
        with archive.open("default of credit card clients.xls") as handle:
            frame = pd.read_excel(handle, header=1, engine="xlrd")
    if frame.shape != (30000,25):
        raise ValueError("Expected 30,000 rows and 25 columns")
    return clean_data(frame)


def make_preprocessor(columns):
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numerical", StandardScaler(), [c for c in columns if c not in CATEGORICAL])])


def make_pipeline(preprocessor, model):
    return Pipeline([("preprocessing", clone(preprocessor)), ("model", model)])


def metrics(actual, predicted, probabilities):
    return {"accuracy":float(accuracy_score(actual,predicted)),
            "precision":float(precision_score(actual,predicted,zero_division=0)),
            "recall":float(recall_score(actual,predicted,zero_division=0)),
            "f1":float(f1_score(actual,predicted,zero_division=0)),
            "roc_auc":float(roc_auc_score(actual,probabilities))}


def run(frame, tune=False):
    X, y = frame.drop(columns=["ID",TARGET]), frame[TARGET]
    train, test, ytrain, ytest = train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    preprocessing = make_preprocessor(X.columns)
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    models = {"logistic":LogisticRegression(max_iter=1000),
              "tree_depth_5":DecisionTreeClassifier(max_depth=5,random_state=SEED),
              "random_forest":RandomForestClassifier(n_estimators=100,random_state=SEED,n_jobs=-1)}
    report = {"evaluation":"Reused exploratory holdout; selection bias is possible", "seed":SEED,
              "train_rows":len(train), "holdout_rows":len(test), "models":{}}
    for name, estimator in models.items():
        model = make_pipeline(preprocessing,estimator)
        scores = cross_validate(model,train,ytrain,cv=cv,scoring=["accuracy","precision","recall","f1"],error_score="raise")
        model.fit(train,ytrain)
        report["models"][name] = {"cv_mean":{k:float(v.mean()) for k,v in scores.items() if k.startswith("test_")},
             "exploratory_holdout":metrics(ytest,model.predict(test),model.predict_proba(test)[:,1])}
    if tune:
        grid = [{"model__l1_ratio":[ratio],"model__solver":[solver],
                 "model__C":[.001,.01,.1,1,10,100],"model__class_weight":[None,"balanced"]}
                for ratio,solver in [(0,"lbfgs"),(1,"liblinear")]]
        base = make_pipeline(preprocessing,LogisticRegression(max_iter=1000,random_state=SEED))
        search = GridSearchCV(base,grid,cv=cv,scoring="f1",refit=True,error_score="raise")
        search.fit(train,ytrain)
        report["selected_parameters"] = search.best_params_
        threshold = TunedThresholdClassifierCV(make_pipeline(preprocessing,LogisticRegression(max_iter=1000)),scoring="f1",cv=cv)
        threshold.fit(train,ytrain)
        report["selected_threshold"] = float(threshold.best_threshold_)
        for name, model in [("grid_tuned",search),("threshold_tuned",threshold)]:
            report["models"][name] = {"exploratory_holdout":metrics(ytest,model.predict(test),model.predict_proba(test)[:,1])}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-zip",required=True,type=Path)
    parser.add_argument("--output",type=Path,default=Path("results.json"))
    parser.add_argument("--tune",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path to preserve prior results")
    report = run(load_data(args.data_zip),tune=args.tune)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
