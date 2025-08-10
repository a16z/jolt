#!/usr/bin/env python3
import json, numpy as np
from pathlib import Path
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
MODEL_PATH = str(HERE / "network.onnx")
VOCAB_PATH = HERE / "vocab.json"
MAX_LEN = 5
PAD_ID = 0

with VOCAB_PATH.open() as f:
    vocab = json.load(f)

def tok(text): return [vocab.get(t, PAD_ID) for t in text.lower().split()]
def pad1(ids, L):
    a = np.full((1, L), PAD_ID, dtype=np.int64)  # (1,L) fixed batch=1
    a[0, :min(len(ids), L)] = ids[:L]
    return a

texts = [
    "I love this", "This is great", "Happy with the result",
    "I hate this", "This is bad", "Not satisfied"
]
expected = np.array([1,1,1,0,0,0], dtype=np.int64)

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

preds = []
for t in texts:
    tokens = pad1(tok(t), MAX_LEN)          # (1,L)
    out = sess.run(None, {"tokens": tokens})  # single output
    y_bool = out[0].ravel()[0]              # bool
    y = int(bool(y_bool))                   # post-process cast (outside ONNX)
    preds.append(y)

preds = np.array(preds, dtype=np.int64)
acc = (preds == expected).mean()
print("pred:", preds.tolist(), "acc:", round(float(acc), 2))