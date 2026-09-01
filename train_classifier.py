import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

DATASET_NAME = "deepset/prompt-injections"
MODEL_OUTPUT_PATH = "security/ml_model.joblib"

def load_data():
    from datasets import load_dataset
    train_ds = load_dataset(DATASET_NAME, split="train")
    test_ds = load_dataset(DATASET_NAME, split="test")

    def normalize(raw):
        if isinstance(raw, str):
            return 1 if raw.strip().upper() in ("INJECTION", "1", "TRUE") else 0
        return int(raw)

    X_train = [row["text"] for row in train_ds]
    y_train = [normalize(row["label"]) for row in train_ds]
    X_test = [row["text"] for row in test_ds]
    y_test = [normalize(row["label"]) for row in test_ds]

    return X_train, y_train, X_test, y_test


def train_and_evaluate():
    X_train, y_train, X_test, y_test = load_data()
    print(f"Train set: {len(X_train)} examples | Test set: {len(X_test)} examples")
    print(f"Train label balance: {sum(y_train)} injection / {len(y_train) - sum(y_train)} legit\n")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("=" * 60)
    print(f"RESULTS on {DATASET_NAME} [test split]")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["legit", "injection"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")

    print_top_features(pipeline)

    return pipeline


def print_top_features(pipeline, top_n=15):
    
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]

    top_injection_idx = coefs.argsort()[-top_n:][::-1]
    top_legit_idx = coefs.argsort()[:top_n]

    print(f"\nTop {top_n} features pushing toward 'injection':")
    for i in top_injection_idx:
        print(f"  {feature_names[i]:30s} weight={coefs[i]:.3f}")

    print(f"\nTop {top_n} features pushing toward 'legit':")
    for i in top_legit_idx:
        print(f"  {feature_names[i]:30s} weight={coefs[i]:.3f}")


if __name__ == "__main__":
    train_and_evaluate()
