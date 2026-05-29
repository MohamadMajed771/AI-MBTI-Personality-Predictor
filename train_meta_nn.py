import re
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 42
PAIR_SAMPLES_PER_LABEL = 1500
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MBTI_TYPES = [
    "infj", "entp", "intp", "intj", "entj", "enfj", "infp", "enfp",
    "isfp", "istp", "isfj", "istj", "estp", "esfp", "estj", "esfj"
]

vectorizer = joblib.load("tfidf_vectorizer.pkl")

model_IE = joblib.load("model_IE.pkl")
model_NS = joblib.load("model_NS.pkl")
model_TF = joblib.load("model_TF.pkl")
model_JP = joblib.load("model_JP.pkl")

question_model_IE = joblib.load("question_model_IE.pkl")
question_model_NS = joblib.load("question_model_NS.pkl")
question_model_TF = joblib.load("question_model_TF.pkl")
question_model_JP = joblib.load("question_model_JP.pkl")

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.replace("|||", " ")
    text = text.lower()

    for t in MBTI_TYPES:
        text = re.sub(rf"\b{t}\b", " ", text)

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prob_for_letter(model, X, first_letter, second_letter):
    probs = model.predict_proba(X)
    classes = [str(c).upper() for c in model.classes_]

    idx_first = classes.index(first_letter)
    idx_second = classes.index(second_letter)

    return probs[:, idx_first], probs[:, idx_second]

df_text = pd.read_csv("mbti_1.csv")
df_text["type"] = df_text["type"].astype(str).str.upper()
df_text["clean_posts"] = df_text["posts"].apply(clean_text)

df_q = pd.read_csv("mbti_synthetic_starter.csv")
df_q["mbti_label"] = df_q["mbti_label"].astype(str).str.upper()

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

shared_labels = sorted(set(df_text["type"]).intersection(set(df_q["mbti_label"])))
print("Shared labels:", shared_labels)

meta_rows = []

for label in shared_labels:
    text_group = df_text[df_text["type"] == label].reset_index(drop=True)
    q_group = df_q[df_q["mbti_label"] == label].reset_index(drop=True)

    if len(text_group) == 0 or len(q_group) == 0:
        continue

    pair_count = min(PAIR_SAMPLES_PER_LABEL, len(text_group) * len(q_group))

    for _ in range(pair_count):
        text_row = text_group.iloc[random.randrange(len(text_group))]
        q_row = q_group.iloc[random.randrange(len(q_group))]

        X_text = vectorizer.transform([text_row["clean_posts"]])

        t_i, t_e = prob_for_letter(model_IE, X_text, "I", "E")
        t_n, t_s = prob_for_letter(model_NS, X_text, "N", "S")
        t_t, t_f = prob_for_letter(model_TF, X_text, "T", "F")
        t_j, t_p = prob_for_letter(model_JP, X_text, "J", "P")

        X_ie = pd.DataFrame([[q_row[c] for c in IE_COLUMNS]], columns=IE_COLUMNS)
        X_ns = pd.DataFrame([[q_row[c] for c in NS_COLUMNS]], columns=NS_COLUMNS)
        X_tf = pd.DataFrame([[q_row[c] for c in TF_COLUMNS]], columns=TF_COLUMNS)
        X_jp = pd.DataFrame([[q_row[c] for c in JP_COLUMNS]], columns=JP_COLUMNS)

        q_i, q_e = prob_for_letter(question_model_IE, X_ie, "I", "E")
        q_n, q_s = prob_for_letter(question_model_NS, X_ns, "N", "S")
        q_t, q_f = prob_for_letter(question_model_TF, X_tf, "T", "F")
        q_j, q_p = prob_for_letter(question_model_JP, X_jp, "J", "P")

        meta_rows.append({
            "type": label,

            "IE_t1": float(t_i[0]), "IE_t2": float(t_e[0]),
            "IE_q1": float(q_i[0]), "IE_q2": float(q_e[0]),

            "NS_t1": float(t_n[0]), "NS_t2": float(t_s[0]),
            "NS_q1": float(q_n[0]), "NS_q2": float(q_s[0]),

            "TF_t1": float(t_t[0]), "TF_t2": float(t_f[0]),
            "TF_q1": float(q_t[0]), "TF_q2": float(q_f[0]),

            "JP_t1": float(t_j[0]), "JP_t2": float(t_p[0]),
            "JP_q1": float(q_j[0]), "JP_q2": float(q_p[0]),
        })

meta_df = pd.DataFrame(meta_rows)
print("Meta dataset shape:", meta_df.shape)

meta_df["IE_label"] = meta_df["type"].str[0]
meta_df["NS_label"] = meta_df["type"].str[1]
meta_df["TF_label"] = meta_df["type"].str[2]
meta_df["JP_label"] = meta_df["type"].str[3]

class MetaNN(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)

def encode_binary_labels(labels, first_letter, second_letter):
    mapping = {first_letter: 0, second_letter: 1}
    return np.array([mapping[x] for x in labels], dtype=np.int64)

def train_one_dimension(X, y_letters, first_letter, second_letter, save_path):
    y = encode_binary_labels(y_letters, first_letter, second_letter)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model = MetaNN(input_dim=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()

        indices = torch.randperm(X_train.size(0))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for start in range(0, X_train.size(0), BATCH_SIZE):
            end = start + BATCH_SIZE
            xb = X_train_shuffled[start:end]
            yb = y_train_shuffled[start:end]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        preds = torch.argmax(test_logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test.cpu().numpy(), preds)
    print(f"{save_path} accuracy: {acc:.4f}")

    torch.save(model.state_dict(), save_path)

X_IE = meta_df[["IE_t1", "IE_t2", "IE_q1", "IE_q2"]].values
X_NS = meta_df[["NS_t1", "NS_t2", "NS_q1", "NS_q2"]].values
X_TF = meta_df[["TF_t1", "TF_t2", "TF_q1", "TF_q2"]].values
X_JP = meta_df[["JP_t1", "JP_t2", "JP_q1", "JP_q2"]].values

train_one_dimension(X_IE, meta_df["IE_label"].values, "I", "E", "meta_IE.pt")
train_one_dimension(X_NS, meta_df["NS_label"].values, "N", "S", "meta_NS.pt")
train_one_dimension(X_TF, meta_df["TF_label"].values, "T", "F", "meta_TF.pt")
train_one_dimension(X_JP, meta_df["JP_label"].values, "J", "P", "meta_JP.pt")

print("Saved:")
print("- meta_IE.pt")
print("- meta_NS.pt")
print("- meta_TF.pt")
print("- meta_JP.pt")