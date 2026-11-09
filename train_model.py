# train_model.py
# Student Performance Prediction System - data retrieval, training, evaluation
import io
import zipfile
from urllib.request import urlopen

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Activity A - Retrieve and inspect the dataset
# ---------------------------------------------------------------------------
ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def _find_math_csv(zf):
    """Look for student-mat.csv inside the zip (also inside nested zips)."""
    for name in zf.namelist():
        if name.endswith("student-mat.csv"):
            return zf.read(name)
    for name in zf.namelist():
        if name.endswith(".zip"):
            found = _find_math_csv(zipfile.ZipFile(io.BytesIO(zf.read(name))))
            if found:
                return found
    return None


def load_dataset():
    """Return the Mathematics records (student-mat.csv, 395 rows).

    Primary: download the official UCI zip and read student-mat.csv.
    Fallback: the ucimlrepo interface given in the lab handout.
    """
    try:
        raw = urlopen(ZIP_URL, timeout=30).read()
        csv_bytes = _find_math_csv(zipfile.ZipFile(io.BytesIO(raw)))
        if csv_bytes is None:
            raise FileNotFoundError("student-mat.csv not found in zip")
        print("Source: UCI zip -> student-mat.csv (Mathematics)")
        return pd.read_csv(io.BytesIO(csv_bytes), sep=";")
    except Exception as err:
        print("Zip download failed (", err, ") - falling back to ucimlrepo")
        from ucimlrepo import fetch_ucirepo

        student = fetch_ucirepo(id=320)
        print("Source: ucimlrepo id=320")
        return pd.concat([student.data.features, student.data.targets], axis=1)


df = load_dataset()
df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)  # Mathematics file = (395, 33)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())
assert {"G1", "G2", "G3"}.issubset(df.columns), "G1, G2, G3 must exist"

# ---------------------------------------------------------------------------
# Activity B - Preprocess, train, evaluate
# ---------------------------------------------------------------------------
# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor (target leakage)
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
report = classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
)
print(report)

# Keep the evaluation as evidence for the submission
with open("model_evaluation.txt", "w") as f:
    f.write(f"Rows used: {len(df)}\n")
    f.write(f"Accuracy: {accuracy_score(y_test, predictions):.3f}\n\n")
    f.write(report)

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")
