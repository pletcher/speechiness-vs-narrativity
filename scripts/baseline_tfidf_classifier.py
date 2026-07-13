"""TF-IDF + logistic regression baseline for epic/tragedy classification."""

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from data_split import ID2LABEL, load_and_split

ROOT_DIR = Path(__file__).parent.parent
IN_CSV = ROOT_DIR / "csv" / "epic_tragedy_sentences.csv"

N_TOP_FEATURES = 25

train_df, eval_df, test_df = load_and_split(IN_CSV)

print(
    f"Train size = {len(train_df)}; eval size = {len(eval_df)}; test size = {len(test_df)}"
)

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2)
X_train = vectorizer.fit_transform(train_df["text"])
X_eval = vectorizer.transform(eval_df["text"])
X_test = vectorizer.transform(test_df["text"])

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train, train_df["label"])

for split_name, X, split_df in [("eval", X_eval, eval_df), ("test", X_test, test_df)]:
    preds = clf.predict(X)
    accuracy = accuracy_score(split_df["label"], preds)
    f1 = f1_score(split_df["label"], preds, average="weighted", zero_division=0)
    print(f"\n{split_name}: accuracy={accuracy:.4f} f1={f1:.4f}")
    print(
        classification_report(
            split_df["label"],
            preds,
            target_names=[ID2LABEL[0], ID2LABEL[1]],
            zero_division=0,
        )
    )

feature_names = vectorizer.get_feature_names_out()
coefs = clf.coef_[0]
top_epic = coefs.argsort()[-N_TOP_FEATURES:][::-1]
top_tragedy = coefs.argsort()[:N_TOP_FEATURES]

print(f"\nTop {N_TOP_FEATURES} words pushing toward 'epic':")
print(", ".join(feature_names[i] for i in top_epic))

print(f"\nTop {N_TOP_FEATURES} words pushing toward 'tragedy':")
print(", ".join(feature_names[i] for i in top_tragedy))
