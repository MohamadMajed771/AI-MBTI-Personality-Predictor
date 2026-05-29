import re
import joblib
import pandas as pd

MBTI_TYPES = [
    "infj", "entp", "intp", "intj", "entj", "enfj", "infp", "enfp",
    "isfp", "istp", "isfj", "istj", "estp", "esfp", "estj", "esfj"
]

DIMENSIONS = {
    "IE": ("I", "E"),
    "NS": ("N", "S"),
    "TF": ("T", "F"),
    "JP": ("J", "P")
}

QUESTION_BANK = {
    "IE": [
        {"id": "ie1", "text": "I feel energized after spending time alone."},
        {"id": "ie2", "text": "I usually prefer a few close friends over a large social circle."},
        {"id": "ie3", "text": "I often think before speaking rather than speaking immediately."},
        {"id": "ie4", "text": "Too much social interaction can leave me mentally tired."}
    ],
    "NS": [
        {"id": "ns1", "text": "I am more interested in ideas and possibilities than concrete facts."},
        {"id": "ns2", "text": "I enjoy imagining future possibilities more than focusing on present realities."},
        {"id": "ns3", "text": "Patterns and hidden meanings usually attract my attention."},
        {"id": "ns4", "text": "I trust inspiration and intuition more than direct observation."}
    ],
    "TF": [
        {"id": "tf1", "text": "When making decisions, I value logic more than personal feelings."},
        {"id": "tf2", "text": "I prefer honest and objective feedback even if it feels harsh."},
        {"id": "tf3", "text": "I usually judge situations by fairness and reason rather than emotion."},
        {"id": "tf4", "text": "In conflict, I focus first on what is correct rather than what is compassionate."}
    ],
    "JP": [
        {"id": "jp1", "text": "I feel better when my work is planned and organized in advance."},
        {"id": "jp2", "text": "I prefer schedules and deadlines over last-minute flexibility."},
        {"id": "jp3", "text": "I like finishing tasks early rather than keeping options open."},
        {"id": "jp4", "text": "Clear structure helps me feel more comfortable and productive."}
    ]
}

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


def get_letter_probability(model, user_vector, first_letter, second_letter):
    probs = model.predict_proba(user_vector)[0]
    classes = model.classes_

    prob_map = {str(c).upper(): float(p) for c, p in zip(classes, probs)}

    return prob_map.get(first_letter, 0.5), prob_map.get(second_letter, 0.5)


def get_text_analysis(user_text):
    cleaned = clean_text(user_text)
    vec = vectorizer.transform([cleaned])

    p_i, p_e = get_letter_probability(model_IE, vec, "I", "E")
    p_n, p_s = get_letter_probability(model_NS, vec, "N", "S")
    p_t, p_f = get_letter_probability(model_TF, vec, "T", "F")
    p_j, p_p = get_letter_probability(model_JP, vec, "J", "P")

    scores = {
        "IE": {"I": p_i, "E": p_e},
        "NS": {"N": p_n, "S": p_s},
        "TF": {"T": p_t, "F": p_f},
        "JP": {"J": p_j, "P": p_p}
    }

    prediction = ""
    for dim, (a, b) in DIMENSIONS.items():
        prediction += a if scores[dim][a] >= scores[dim][b] else b

    return prediction, scores


def get_all_questions():
    return (
        QUESTION_BANK["IE"] +
        QUESTION_BANK["NS"] +
        QUESTION_BANK["TF"] +
        QUESTION_BANK["JP"]
    )


def get_valid_answer(q):
    while True:
        ans = input(q).strip()
        if ans in ["1", "2", "3", "4"]:
            return ans
        print("Enter 1–4 only.")


def get_question_analysis(answer_map):
    ie = [float(answer_map[f"ie{i}"]) for i in range(1, 5)]
    ns = [float(answer_map[f"ns{i}"]) for i in range(1, 5)]
    tf = [float(answer_map[f"tf{i}"]) for i in range(1, 5)]
    jp = [float(answer_map[f"jp{i}"]) for i in range(1, 5)]

    q_i, q_e = get_letter_probability(question_model_IE, pd.DataFrame([ie], columns=["IE01", "IE02", "IE03", "IE04"]), "I", "E")
    q_n, q_s = get_letter_probability(question_model_NS, pd.DataFrame([ns], columns=["NS01", "NS02", "NS03", "NS04"]), "N", "S")
    q_t, q_f = get_letter_probability(question_model_TF, pd.DataFrame([tf], columns=["TF01", "TF02", "TF03", "TF04"]), "T", "F")
    q_j, q_p = get_letter_probability(question_model_JP, pd.DataFrame([jp], columns=["JP01", "JP02", "JP03", "JP04"]), "J", "P")

    scores = {
        "IE": {"I": q_i, "E": q_e},
        "NS": {"N": q_n, "S": q_s},
        "TF": {"T": q_t, "F": q_f},
        "JP": {"J": q_j, "P": q_p}
    }

    pred = ""
    for dim, (a, b) in DIMENSIONS.items():
        pred += a if scores[dim][a] >= scores[dim][b] else b

    return pred, scores


def combine(text_scores, q_scores):
    final = ""
    for dim, (a, b) in DIMENSIONS.items():
        val = 0.7 * text_scores[dim][a] + 0.3 * q_scores[dim][a]
        final += a if val >= 0.5 else b
    return final


print("=" * 50)
print("MBTI PREDICTION (16 QUESTIONS)")
print("=" * 50)

text = input("\nEnter a paragraph about yourself:\n\n")
text_pred, text_scores = get_text_analysis(text)

print("\nText prediction:", text_pred)

print("\nAnswer all 16 questions (1–4):")

answers = {}
questions = get_all_questions()

for i, q in enumerate(questions, 1):
    answers[q["id"]] = get_valid_answer(f"\n{i}) {q['text']}\nYour answer: ")

q_pred, q_scores = get_question_analysis(answers)
final = combine(text_scores, q_scores)

print("\nText prediction:", text_pred)
print("Question prediction:", q_pred)
print("Final MBTI:", final)