from flask import Flask, render_template, request
import re
import joblib
import pandas as pd
import torch
import torch.nn as nn

app = Flask(__name__)

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
        {"id": "ie1", "text": "I feel more comfortable with a small circle of close friends than with a large social group."},
        {"id": "ie2", "text": "I gain energy from being around people and joining active conversations."},
        {"id": "ie3", "text": "I usually need quiet time alone to recharge after busy days."},
        {"id": "ie4", "text": "I often think better by talking things out with others."}
    ],
    "NS": [
        {"id": "ns1", "text": "I am more interested in ideas, patterns, and future possibilities than in concrete details."},
        {"id": "ns2", "text": "I trust facts, real examples, and practical experience more than abstract ideas."},
        {"id": "ns3", "text": "I naturally notice hidden meanings and connections between things."},
        {"id": "ns4", "text": "I focus more on what is real and observable than on imagination or speculation."}
    ],
    "TF": [
        {"id": "tf1", "text": "When making decisions, I usually rely more on logic than on emotions."},
        {"id": "tf2", "text": "I often choose based on compassion and people’s feelings more than strict logic."},
        {"id": "tf3", "text": "Fairness and objective reasoning are more important to me than emotional comfort."},
        {"id": "tf4", "text": "I care strongly about maintaining harmony and protecting others’ feelings."}
    ],
    "JP": [
        {"id": "jp1", "text": "I feel better when things are planned, organized, and decided in advance."},
        {"id": "jp2", "text": "I prefer to keep my options open instead of following a fixed plan."},
        {"id": "jp3", "text": "I like finishing tasks early and having clear structure."},
        {"id": "jp4", "text": "I feel more comfortable adapting as I go rather than sticking to schedules."}
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


meta_IE = MetaNN()
meta_NS = MetaNN()
meta_TF = MetaNN()
meta_JP = MetaNN()

meta_IE.load_state_dict(torch.load("meta_IE.pt", map_location="cpu"))
meta_NS.load_state_dict(torch.load("meta_NS.pt", map_location="cpu"))
meta_TF.load_state_dict(torch.load("meta_TF.pt", map_location="cpu"))
meta_JP.load_state_dict(torch.load("meta_JP.pt", map_location="cpu"))

meta_IE.eval()
meta_NS.eval()
meta_TF.eval()
meta_JP.eval()


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
    probabilities = model.predict_proba(user_vector)[0]
    classes = model.classes_

    prob_map = {}
    for cls, prob in zip(classes, probabilities):
        prob_map[str(cls).upper()] = float(prob)

    p_first = prob_map.get(first_letter, 0.5)
    p_second = prob_map.get(second_letter, 0.5)
    return p_first, p_second


def get_text_analysis(user_text: str):
    cleaned = clean_text(user_text)
    user_vector = vectorizer.transform([cleaned])

    p_i, p_e = get_letter_probability(model_IE, user_vector, "I", "E")
    p_n, p_s = get_letter_probability(model_NS, user_vector, "N", "S")
    p_t, p_f = get_letter_probability(model_TF, user_vector, "T", "F")
    p_j, p_p = get_letter_probability(model_JP, user_vector, "J", "P")

    scores = {
        "IE": {"I": p_i, "E": p_e},
        "NS": {"N": p_n, "S": p_s},
        "TF": {"T": p_t, "F": p_f},
        "JP": {"J": p_j, "P": p_p}
    }

    prediction = ""
    confidence = {}

    for dim, (first_letter, second_letter) in DIMENSIONS.items():
        if scores[dim][first_letter] >= scores[dim][second_letter]:
            prediction += first_letter
            confidence[dim] = scores[dim][first_letter]
        else:
            prediction += second_letter
            confidence[dim] = scores[dim][second_letter]

    return prediction, scores, confidence


def get_all_questions():
    return (
        QUESTION_BANK["IE"] +
        QUESTION_BANK["NS"] +
        QUESTION_BANK["TF"] +
        QUESTION_BANK["JP"]
    )


def get_question_analysis(answer_map):
    ie_values = [float(answer_map[f"ie{i}"]) for i in range(1, 5)]
    X_ie = pd.DataFrame([ie_values], columns=["IE01", "IE02", "IE03", "IE04"])
    q_i, q_e = get_letter_probability(question_model_IE, X_ie, "I", "E")

    ns_values = [float(answer_map[f"ns{i}"]) for i in range(1, 5)]
    X_ns = pd.DataFrame([ns_values], columns=["NS01", "NS02", "NS03", "NS04"])
    q_n, q_s = get_letter_probability(question_model_NS, X_ns, "N", "S")

    tf_values = [float(answer_map[f"tf{i}"]) for i in range(1, 5)]
    X_tf = pd.DataFrame([tf_values], columns=["TF01", "TF02", "TF03", "TF04"])
    q_t, q_f = get_letter_probability(question_model_TF, X_tf, "T", "F")

    jp_values = [float(answer_map[f"jp{i}"]) for i in range(1, 5)]
    X_jp = pd.DataFrame([jp_values], columns=["JP01", "JP02", "JP03", "JP04"])
    q_j, q_p = get_letter_probability(question_model_JP, X_jp, "J", "P")

    scores = {
        "IE": {"I": q_i, "E": q_e},
        "NS": {"N": q_n, "S": q_s},
        "TF": {"T": q_t, "F": q_f},
        "JP": {"J": q_j, "P": q_p}
    }

    prediction = ""
    for dim, (first_letter, second_letter) in DIMENSIONS.items():
        if scores[dim][first_letter] >= scores[dim][second_letter]:
            prediction += first_letter
        else:
            prediction += second_letter

    return prediction, scores


def predict_meta_dimension(meta_model, values, first_letter, second_letter):
    x = torch.tensor([values], dtype=torch.float32)

    with torch.no_grad():
        logits = meta_model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    prob_first = float(probs[0])
    prob_second = float(probs[1])

    pred = first_letter if prob_first >= prob_second else second_letter
    return pred, prob_first, prob_second


def combine_with_meta_nn(text_scores, question_scores):
    pred_ie, p_i, p_e = predict_meta_dimension(
        meta_IE,
        [
            text_scores["IE"]["I"], text_scores["IE"]["E"],
            question_scores["IE"]["I"], question_scores["IE"]["E"]
        ],
        "I", "E"
    )

    pred_ns, p_n, p_s = predict_meta_dimension(
        meta_NS,
        [
            text_scores["NS"]["N"], text_scores["NS"]["S"],
            question_scores["NS"]["N"], question_scores["NS"]["S"]
        ],
        "N", "S"
    )

    pred_tf, p_t, p_f = predict_meta_dimension(
        meta_TF,
        [
            text_scores["TF"]["T"], text_scores["TF"]["F"],
            question_scores["TF"]["T"], question_scores["TF"]["F"]
        ],
        "T", "F"
    )

    pred_jp, p_j, p_p = predict_meta_dimension(
        meta_JP,
        [
            text_scores["JP"]["J"], text_scores["JP"]["P"],
            question_scores["JP"]["J"], question_scores["JP"]["P"]
        ],
        "J", "P"
    )

    final_mbti = pred_ie + pred_ns + pred_tf + pred_jp

    final_details = {
        "IE": {
            "letters": ("I", "E"),
            "text_probs": {
                "I": round(text_scores["IE"]["I"], 4),
                "E": round(text_scores["IE"]["E"], 4)
            },
            "question_probs": {
                "I": round(question_scores["IE"]["I"], 4),
                "E": round(question_scores["IE"]["E"], 4)
            },
            "meta_probs": {
                "I": round(p_i, 6),
                "E": round(p_e, 6)
            },
            "prediction": pred_ie
        },
        "NS": {
            "letters": ("N", "S"),
            "text_probs": {
                "N": round(text_scores["NS"]["N"], 4),
                "S": round(text_scores["NS"]["S"], 4)
            },
            "question_probs": {
                "N": round(question_scores["NS"]["N"], 4),
                "S": round(question_scores["NS"]["S"], 4)
            },
            "meta_probs": {
                "N": round(p_n, 6),
                "S": round(p_s, 6)
            },
            "prediction": pred_ns
        },
        "TF": {
            "letters": ("T", "F"),
            "text_probs": {
                "T": round(text_scores["TF"]["T"], 4),
                "F": round(text_scores["TF"]["F"], 4)
            },
            "question_probs": {
                "T": round(question_scores["TF"]["T"], 4),
                "F": round(question_scores["TF"]["F"], 4)
            },
            "meta_probs": {
                "T": round(p_t, 6),
                "F": round(p_f, 6)
            },
            "prediction": pred_tf
        },
        "JP": {
            "letters": ("J", "P"),
            "text_probs": {
                "J": round(text_scores["JP"]["J"], 4),
                "P": round(text_scores["JP"]["P"], 4)
            },
            "question_probs": {
                "J": round(question_scores["JP"]["J"], 4),
                "P": round(question_scores["JP"]["P"], 4)
            },
            "meta_probs": {
                "J": round(p_j, 6),
                "P": round(p_p, 6)
            },
            "prediction": pred_jp
        }
    }

    return final_mbti, final_details

@app.route("/", methods=["GET", "POST"])
def home():
    stage = "text_input"
    user_text = ""
    text_prediction = None
    question_prediction = None
    final_prediction = None
    selected_questions = []
    final_details = None
    error_message = None

    if request.method == "POST":
        form_stage = request.form.get("form_stage")

        if form_stage == "text_predict":
            user_text = request.form.get("user_text", "").strip()

            if user_text:
                text_prediction, text_scores, confidence = get_text_analysis(user_text)
                selected_questions = get_all_questions()
                stage = "questions"
            else:
                error_message = "Please enter a paragraph before continuing."

        elif form_stage == "final_predict":
            user_text = request.form.get("user_text", "").strip()

            if user_text:
                text_prediction, text_scores, confidence = get_text_analysis(user_text)
                selected_questions = get_all_questions()

                answer_map = {}
                all_answered = True

                for question in selected_questions:
                    value = request.form.get(question["id"])
                    if not value:
                        all_answered = False
                        break
                    answer_map[question["id"]] = value

                if all_answered:
                    question_prediction, question_scores = get_question_analysis(answer_map)
                    final_prediction, final_details = combine_with_meta_nn(
                        text_scores,
                        question_scores
                    )
                    stage = "result"
                else:
                    error_message = "Please answer all 16 questions."
                    stage = "questions"
            else:
                error_message = "Please enter a paragraph before continuing."

    return render_template(
        "index.html",
        stage=stage,
        user_text=user_text,
        text_prediction=text_prediction,
        question_prediction=question_prediction,
        final_prediction=final_prediction,
        selected_questions=selected_questions,
        final_details=final_details,
        error_message=error_message
    )


if __name__ == "__main__":
    app.run(debug=True)
