
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

# 1. Load data
df = pd.read_csv("data/Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

mask = df["tenure"].eq(0) & df["TotalCharges"].isna()
df.loc[mask, "TotalCharges"] = 0

# Drop IDs and missing target rows
df = df.drop(columns=["customerID"])
df = df.dropna(subset=["Churn"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values in Churn column")

# 2. Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 3. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 4. Compare two models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = {}
best_name = None
best_auc = -1
best_pipeline = None

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)
    results[name] = auc

    print(f"\n{name}")
    print("ROC-AUC:", round(auc, 4))
    print(classification_report(
        y_test, predictions, zero_division=0
    ))
    print("Confusion matrix:\n",
          confusion_matrix(y_test, predictions))

    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_pipeline = pipeline

# 5. Save the winning model
joblib.dump(best_pipeline, "churn_model.joblib")

print("\nBest model:", best_name)
print("Best test ROC-AUC:", round(best_auc, 4))
print("Saved: churn_model.joblib")
