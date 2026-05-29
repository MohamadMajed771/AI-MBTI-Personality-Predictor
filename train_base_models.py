import pandas as pd
import re
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

SEED = 42
np.random.seed(SEED)

df = pd.read_csv("mbti_1.csv")

MBTI_TYPES = [
    "infj", "entp", "intp", "intj", "entj", "enfj", "infp", "enfp",
    "isfp", "istp", "isfj", "istj", "estp", "esfp", "estj", "esfj"
]

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.replace("|||", " ")
    text = text.lower()

    for t in MBTI_TYPES:
        text = re.sub(rf"\b{t}\b", " ", text)

    text = re.sub(r"[^a-z\s]", " ", text) #keeps only letters and one space
    text = re.sub(r"\s+", " ", text).strip() #remove extra spaces
    return text

df["clean_posts"] = df["posts"].apply(clean_text)

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["clean_posts"])

df["type"] = df["type"].astype(str).str.upper()
df["IE"] = df["type"].str[0]
df["NS"] = df["type"].str[1]
df["TF"] = df["type"].str[2]
df["JP"] = df["type"].str[3]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    df[["IE", "NS", "TF", "JP", "type"]],
    test_size=0.2,
    random_state=SEED,
    stratify=df["type"]
)

model_IE = LogisticRegression(max_iter=2000, class_weight="balanced")
model_NS = LogisticRegression(max_iter=2000, class_weight="balanced")
model_TF = LogisticRegression(max_iter=2000, class_weight="balanced")
model_JP = LogisticRegression(max_iter=2000, class_weight="balanced")

model_IE.fit(X_train, y_train["IE"])
model_NS.fit(X_train, y_train["NS"])
model_TF.fit(X_train, y_train["TF"])
model_JP.fit(X_train, y_train["JP"])

print("\nTEXT Trait Accuracies:")
pred_IE = model_IE.predict(X_test)
pred_NS = model_NS.predict(X_test)
pred_TF = model_TF.predict(X_test)
pred_JP = model_JP.predict(X_test)

print("IE Accuracy:", accuracy_score(y_test["IE"], pred_IE))
print("NS Accuracy:", accuracy_score(y_test["NS"], pred_NS))
print("TF Accuracy:", accuracy_score(y_test["TF"], pred_TF))
print("JP Accuracy:", accuracy_score(y_test["JP"], pred_JP))

final_predictions = [
    pred_IE[i] + pred_NS[i] + pred_TF[i] + pred_JP[i]
    for i in range(len(pred_IE))
]
print("TEXT Final MBTI Accuracy:", accuracy_score(y_test["type"], final_predictions))


df_q = pd.read_csv("mbti_synthetic_starter.csv")

IE_COLUMNS = ["IE01", "IE02", "IE03", "IE04"]
NS_COLUMNS = ["NS01", "NS02", "NS03", "NS04"]
TF_COLUMNS = ["TF01", "TF02", "TF03", "TF04"]
JP_COLUMNS = ["JP01", "JP02", "JP03", "JP04"]

QUESTION_COLUMNS = IE_COLUMNS + NS_COLUMNS + TF_COLUMNS + JP_COLUMNS

def make_questionnaire_more_realistic(df_q, question_columns, seed=42):
    rng = np.random.default_rng(seed)
    df_q = df_q.copy()

    # Add stronger noise to all answers
    for col in question_columns:
        noise = rng.choice(
            [-2, -1, 0, 1, 2],
            size=len(df_q),
            p=[0.10, 0.20, 0.40, 0.20, 0.10]
        )
        df_q[col] = df_q[col] + noise
        df_q[col] = df_q[col].clip(1, 5)

    # Randomize 55% of answers to simulate real human inconsistency
    random_mask = rng.random((len(df_q), len(question_columns))) < 0.55
    random_values = rng.integers(1, 6, size=(len(df_q), len(question_columns)))

    df_q.loc[:, question_columns] = np.where(
        random_mask,
        random_values,
        df_q[question_columns].values
    )

    return df_q

for col in QUESTION_COLUMNS:
    df_q[col] = pd.to_numeric(df_q[col], errors="coerce")
df_q = make_questionnaire_more_realistic(df_q, QUESTION_COLUMNS, seed=42)

df_q["mbti_label"] = df_q["mbti_label"].astype(str).str.upper()

# Derive labels from mbti_label instead of trusting separate label columns
df_q["ie_label"] = df_q["mbti_label"].str[0]
df_q["ns_label"] = df_q["mbti_label"].str[1]
df_q["tf_label"] = df_q["mbti_label"].str[2]
df_q["jp_label"] = df_q["mbti_label"].str[3]

X_ie_train, X_ie_test, y_ie_train, y_ie_test = train_test_split(
    df_q[IE_COLUMNS],
    df_q["ie_label"],
    test_size=0.2,
    random_state=SEED,
    stratify=df_q["ie_label"]
)

X_ns_train, X_ns_test, y_ns_train, y_ns_test = train_test_split(
    df_q[NS_COLUMNS],
    df_q["ns_label"],
    test_size=0.2,
    random_state=SEED,
    stratify=df_q["ns_label"]
)

X_tf_train, X_tf_test, y_tf_train, y_tf_test = train_test_split(
    df_q[TF_COLUMNS],
    df_q["tf_label"],
    test_size=0.2,
    random_state=SEED,
    stratify=df_q["tf_label"]
)

X_jp_train, X_jp_test, y_jp_train, y_jp_test = train_test_split(
    df_q[JP_COLUMNS],
    df_q["jp_label"],
    test_size=0.2,
    random_state=SEED,
    stratify=df_q["jp_label"]
)

question_model_IE = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
question_model_NS = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
question_model_TF = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
question_model_JP = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

question_model_IE.fit(X_ie_train, y_ie_train)
question_model_NS.fit(X_ns_train, y_ns_train)
question_model_TF.fit(X_tf_train, y_tf_train)
question_model_JP.fit(X_jp_train, y_jp_train)

print("\nQUESTION Trait Accuracies:")
q_pred_IE = question_model_IE.predict(X_ie_test)
q_pred_NS = question_model_NS.predict(X_ns_test)
q_pred_TF = question_model_TF.predict(X_tf_test)
q_pred_JP = question_model_JP.predict(X_jp_test)

print("IE Accuracy:", accuracy_score(y_ie_test, q_pred_IE))
print("NS Accuracy:", accuracy_score(y_ns_test, q_pred_NS))
print("TF Accuracy:", accuracy_score(y_tf_test, q_pred_TF))
print("JP Accuracy:", accuracy_score(y_jp_test, q_pred_JP))


joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

joblib.dump(model_IE, "model_IE.pkl")
joblib.dump(model_NS, "model_NS.pkl")
joblib.dump(model_TF, "model_TF.pkl")
joblib.dump(model_JP, "model_JP.pkl")

joblib.dump(question_model_IE, "question_model_IE.pkl")
joblib.dump(question_model_NS, "question_model_NS.pkl")
joblib.dump(question_model_TF, "question_model_TF.pkl")
joblib.dump(question_model_JP, "question_model_JP.pkl")

joblib.dump(QUESTION_COLUMNS, "question_columns.pkl")

RUNTIME_TO_DATASET_COLUMN = {
    "ie1": "IE01",
    "ie2": "IE02",
    "ie3": "IE03",
    "ie4": "IE04",
    "ns1": "NS01",
    "ns2": "NS02",
    "ns3": "NS03",
    "ns4": "NS04",
    "tf1": "TF01",
    "tf2": "TF02",
    "tf3": "TF03",
    "tf4": "TF04",
    "jp1": "JP01",
    "jp2": "JP02",
    "jp3": "JP03",
    "jp4": "JP04"
}

joblib.dump(RUNTIME_TO_DATASET_COLUMN, "runtime_to_dataset_column.pkl")

print("\nSaved successfully.")