#!/usr/bin/env python3
import json, os
from pathlib import Path
import numpy as np
import onnxruntime as ort

MODEL_PATH = "./network.onnx"     
VOCAB_FILENAME = "vocab.json"     
MAX_LEN = 5

# --- resolve paths relative to this script file ---
HERE = Path(__file__).resolve().parent
MODEL_PATH = str((HERE / MODEL_PATH).resolve())
VOCAB_PATH = (HERE / VOCAB_FILENAME).resolve()  # <— key change

# --- tokenization helpers ---
def load_vocab(path: Path):
    if not path.exists():
        # extra diagnostics
        raise FileNotFoundError(
            f"vocab.json not found at: {path}\n"
            f"cwd={Path.cwd()}\n"
            f"script_dir={HERE}\n"
            f"Files in script_dir: {sorted(p.name for p in HERE.iterdir())}"
        )
    with path.open("r") as f:
        data = json.load(f)
    # Support both shapes
    if isinstance(data, dict) and "vocab" in data:
        return data["vocab"], data.get("meta", {})
    return data, {}

def tokenize(text, vocab, pad_id=0):
    return [vocab.get(tok, pad_id) for tok in text.lower().split()]

def pad_or_truncate(ids, L, pad_id=0):
    arr = np.full((L,), pad_id, dtype=np.int64)
    n = min(len(ids), L)
    arr[:n] = ids[:n]
    return arr

# --- load vocab ---
vocab, meta = load_vocab(VOCAB_PATH)
pad_id = int(meta.get("pad_id", 0))
max_len = int(meta.get("max_len", MAX_LEN))

# Test set
texts = [
    "I love this", "This is great", "Happy with the result",
    "I hate this", "This is bad", "Not satisfied"
]
expected = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)

# --- prepare batch ---
ids_batch = [pad_or_truncate(tokenize(t, vocab, pad_id), max_len, pad_id) for t in texts]
tokens = np.stack(ids_batch, axis=0)

# --- inference ---
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])  # fixed typo
outputs = sess.run(None, {"tokens": tokens})
name_to_out = {sess.get_outputs()[i].name: outputs[i] for i in range(len(outputs))}
logit = np.asarray(name_to_out.get("logit", outputs[0])).reshape(-1)
label = np.asarray(name_to_out.get("label", outputs[1])).reshape(-1).astype(np.int64)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

print(f"Loaded model: {MODEL_PATH}")
print(f"Resolved vocab: {VOCAB_PATH}")
print(f"Batch size: {tokens.shape[0]}, Seq len: {tokens.shape[1]}\n")

correct = 0
for i, (t, exp, lg, yhat) in enumerate(zip(texts, expected, logit, label)):
    prob = sigmoid(lg)
    ok = "✓" if yhat == exp else "✗"
    print(f"#{i+1}: {t}")
    print(f"   tokens: {list(tokens[i])}")
    print(f"   expected: {exp} | logit: {lg:.4f} | prob≈sigmoid(logit): {prob:.4f} | pred: {yhat}  {ok}\n")
    correct += int(yhat == exp)

acc = correct / len(expected)
print(f"Accuracy: {correct}/{len(expected)} = {acc:.2f}")