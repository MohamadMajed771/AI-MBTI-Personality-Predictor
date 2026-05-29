import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

MBTI_TYPES = [
    "INFJ", "ENTP", "INTP", "INTJ", "ENTJ", "ENFJ", "INFP", "ENFP",
    "ISFP", "ISTP", "ISFJ", "ISTJ", "ESTP", "ESFP", "ESTJ", "ESFJ"
]

SAMPLES_PER_TYPE = 80

IE_COLUMNS = ["IE01", "IE02", "IE03", "IE04"]
NS_COLUMNS = ["NS01", "NS02", "NS03", "NS04"]
TF_COLUMNS = ["TF01", "TF02", "TF03", "TF04"]
JP_COLUMNS = ["JP01", "JP02", "JP03", "JP04"]

def answer_for_trait(trait, positive_trait):
    """
    Creates realistic overlapping answers.
    Not always clear 1 or 5.
    """
    if trait == positive_trait:
        return rng.choice([2, 3, 4, 5], p=[0.15, 0.25, 0.35, 0.25])
    else:
        return rng.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15])

def create_dimension_answers(trait, first_trait, second_trait):
    answers = []

    answers.append(answer_for_trait(trait, first_trait))
    answers.append(answer_for_trait(trait, second_trait))
    answers.append(answer_for_trait(trait, first_trait))
    answers.append(answer_for_trait(trait, second_trait))

    for i in range(len(answers)):
        if rng.random() < 0.30:
            answers[i] = rng.integers(1, 6)

    return answers

rows = []

for mbti in MBTI_TYPES:
    for i in range(SAMPLES_PER_TYPE):
        ie = mbti[0]
        ns = mbti[1]
        tf = mbti[2]
        jp = mbti[3]

        ie_answers = create_dimension_answers(ie, "I", "E")
        ns_answers = create_dimension_answers(ns, "N", "S")
        tf_answers = create_dimension_answers(tf, "T", "F")
        jp_answers = create_dimension_answers(jp, "J", "P")

        row = {
            "respondent_id": f"SYN-{mbti}-{i+1:03d}",
            "mbti_label": mbti,

            "IE01": ie_answers[0],
            "IE02": ie_answers[1],
            "IE03": ie_answers[2],
            "IE04": ie_answers[3],

            "NS01": ns_answers[0],
            "NS02": ns_answers[1],
            "NS03": ns_answers[2],
            "NS04": ns_answers[3],

            "TF01": tf_answers[0],
            "TF02": tf_answers[1],
            "TF03": tf_answers[2],
            "TF04": tf_answers[3],

            "JP01": jp_answers[0],
            "JP02": jp_answers[1],
            "JP03": jp_answers[2],
            "JP04": jp_answers[3],

            "ie_label": ie,
            "ns_label": ns,
            "tf_label": tf,
            "jp_label": jp
        }

        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("mbti_synthetic_starter.csv", index=False)

print("New questionnaire dataset created successfully.")
print("Shape:", df.shape)
print(df.head())