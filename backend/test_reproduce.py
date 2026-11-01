import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
ckpt = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
model.eval()

T = 1.50

test_sentences = [
    "The movie was amazing.",
    "The movie was amazig.",
    "The movie was amazng.",
    "The movie was excellent.",
    "The movie was terrible.",
    "The movie was not good.",
    "The movie was okay."
]

out_f = open("backend/test_out.txt", "w")
def log_print(*args):
    s = " ".join(str(a) for a in args)
    print(s, flush=True)
    out_f.write(s + "\n")
    out_f.flush()

log_print("=== DISTILBERT DIAGNOSTIC EVALUATION (RAW vs CALIBRATED T=1.50) ===")
log_print(f"Checkpoint: {ckpt}")
log_print(f"id2label: {model.config.id2label}")
log_print(f"label2id: {model.config.label2id}")

for text in test_sentences:
    tokens = tokenizer.tokenize(text)
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu()
        raw_probs = F.softmax(logits, dim=-1)
        cal_probs = F.softmax(logits / T, dim=-1)
    
    pred_idx = torch.argmax(logits).item()
    pred_label = model.config.id2label[pred_idx]
    
    raw_neg = raw_probs[0].item() * 100
    raw_pos = raw_probs[1].item() * 100
    raw_conf = max(raw_neg, raw_pos)
    
    cal_neg = cal_probs[0].item() * 100
    cal_pos = cal_probs[1].item() * 100
    cal_conf = max(cal_neg, cal_pos)
    
    log_print(f"\nText: \"{text}\"")
    log_print(f"  Tokens: {tokens}")
    log_print(f"  Raw Logits [neg, pos]: [{logits[0].item():.4f}, {logits[1].item():.4f}] (Logit Margin: {abs(logits[1].item() - logits[0].item()):.4f})")
    log_print(f"  Raw Probabilities: Neg={raw_neg:.2f}%, Pos={raw_pos:.2f}% (Sum: {(raw_neg + raw_pos)/100:.6f})")
    log_print(f"  Raw Confidence: {raw_conf:.2f}%")
    log_print(f"  Calibrated Probabilities (T={T}): Neg={cal_neg:.2f}%, Pos={cal_pos:.2f}% (Sum: {(cal_neg + cal_pos)/100:.6f})")
    log_print(f"  Calibrated Confidence: {cal_conf:.2f}%")
    log_print(f"  Predicted Label: {pred_label}")

out_f.close()
