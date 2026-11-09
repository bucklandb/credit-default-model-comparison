import unittest
import pandas as pd
import numpy as np
from credit_default import clean_data, make_preprocessor, make_pipeline, metrics, TARGET, CATEGORICAL
from sklearn.linear_model import LogisticRegression

class PortfolioTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(20)
        columns=["ID",TARGET,"LIMIT_BAL","AGE",*CATEGORICAL,*[f"BILL_AMT{i}" for i in range(1,7)],*[f"PAY_AMT{i}" for i in range(1,7)]]
        data=pd.DataFrame({c:rng.integers(1,10,40) for c in columns})
        data["ID"]=range(40); data[TARGET]=[0,1]*20
        data["EDUCATION"]=1; data["MARRIAGE"]=2
        return data
    def test_clean_copy_and_unknown(self):
        raw=self.fixture(); raw.loc[0,"EDUCATION"]=6
        clean=clean_data(raw)
        self.assertEqual(clean.loc[0,"EDUCATION"],"Unknown")
        self.assertEqual(raw.loc[0,"EDUCATION"],6)
    def test_schema_rejected(self):
        with self.assertRaises(ValueError): clean_data(self.fixture().drop(columns=["AGE"]))
    def test_duplicate_rejected(self):
        data=self.fixture(); data.loc[1,"ID"]=0
        with self.assertRaises(ValueError): clean_data(data)
    def test_categories_rejected(self):
        data=self.fixture(); data.loc[1,"EDUCATION"]=999
        with self.assertRaises(ValueError): clean_data(data)
    def test_hand_metrics(self):
        values=metrics([1,1,1,1,0,0],[1,1,0,0,1,0],[.9,.8,.3,.2,.7,.1])
        self.assertAlmostEqual(values["f1"],4/7)
        self.assertAlmostEqual(values["recall"],.5)
    def test_pipeline_smoke_and_unknown_category(self):
        data=clean_data(self.fixture()); X=data.drop(columns=["ID",TARGET])
        model=make_pipeline(make_preprocessor(X.columns),LogisticRegression(max_iter=1000))
        model.fit(X.iloc[:30],data[TARGET].iloc[:30])
        test=X.iloc[30:].copy(); test["PAY_0"]=999
        self.assertEqual(model.predict(test).shape,(10,))
        self.assertNotIn("ID",model.named_steps["preprocessing"].feature_names_in_)

if __name__ == "__main__": unittest.main()
