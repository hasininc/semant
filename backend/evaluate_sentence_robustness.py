import json
import torch
import torch.nn.functional as F
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
distilbert = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
distilbert.eval()

word_model = joblib.load("backend/models/sentiment_pipeline.joblib")
char_model = joblib.load("backend/models/sentiment_char_pipeline.joblib")

test_sentences = [
    "The movie was amazing.",
    "The movie was amazig.",
    "The movie was amazng.",
    "The movie was excellent.",
    "The movie was terrible.",
    "The movie was not good.",
    "The movie was okay."
]

T = 1.50

results = []

for text in test_sentences:
    # 1. Word TF-IDF
    w_pred = int(word_model.predict([text])[0])
    w_probs = word_model.predict_proba([text])[0]
    w_label = "Positive" if w_pred == 1 else "Negative"
    w_conf = round(float(w_probs[w_pred]) * 100, 2)
    w_neg = round(float(w_probs[0]) * 100, 2)
    w_pos = round(float(w_probs[1]) * 100, 2)

    # 2. Char TF-IDF
    c_pred = int(char_model.predict([text])[0])
    c_probs = char_model.predict_proba([text])[0]
    c_label = "Positive" if c_pred == 1 else "Negative"
    c_conf = round(float(c_probs[c_pred]) * 100, 2)
    c_neg = round(float(c_probs[0]) * 100, 2)
    c_pos = round(float(c_probs[1]) * 100, 2)

    # 3. DistilBERT
    tokens = tokenizer.tokenize(text)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = distilbert(**inputs).logits[0].cpu()
        raw_probs = F.softmax(logits, dim=-1)
        cal_probs = F.softmax(logits / T, dim=-1)

    d_pred = int(torch.argmax(logits).item())
    d_label = "Positive" if d_pred == 1 else "Negative"
    d_raw_neg = round(float(raw_probs[0]) * 100, 2)
    d_raw_pos = round(float(raw_probs[1]) * 100, 2)
    d_raw_conf = round(float(max(raw_probs)) * 100, 2)

    d_cal_neg = round(float(cal_probs[0]) * 100, 2)
    d_cal_pos = round(float(cal_probs[1]) * 100, 2)
    d_cal_conf = round(float(max(cal_probs)) * 100, 2)

    res = {
        "text": text,
        "tokens": tokens,
        "raw_logits": [round(float(logits[0]), 4), round(float(logits[1]), 4)],
        "logit_margin": round(float(abs(logits[1] - logits[0])), 4),
        "word_tfidf": {
            "prediction": w_label,
            "confidence": w_conf,
            "probabilities": {"negative": w_neg, "positive": w_pos}
        },
        "char_tfidf": {
            "prediction": c_label,
            "confidence": c_conf,
            "probabilities": {"negative": c_neg, "positive": c_pos}
        },
        "distilbert": {
            "prediction": d_label,
            "raw_confidence": d_raw_conf,
            "raw_probabilities": {"negative": d_raw_neg, "positive": d_raw_pos},
            "calibrated_confidence": d_cal_conf,
            "calibrated_probabilities": {"negative": d_cal_neg, "positive": d_cal_pos},
            "temperature": T
        }
    }
    results.append(res)

print(json.dumps(results, indent=2))
with open("backend/models/controlled_sentences_eval.json", "w") as f:
    json.dump(results, f, indent=2)
