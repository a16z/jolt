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
def pad(ids, L): 
    a = np.full((L,), PAD_ID, dtype=np.int64); a[:min(len(ids), L)] = ids[:L]; return a

texts = [
    "I love this", "This is great", "Happy with the result",
    "I hate this", "This is bad", "Not satisfied"
]
expected = np.array([1,1,1,0,0,0], dtype=np.int64)

tokens = np.stack([pad(tok(t), MAX_LEN) for t in texts], axis=0)  # (N,L)
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
outs = sess.run(None, {"tokens": tokens})
names = [o.name for o in sess.get_outputs()]
name2 = {names[i]: outs[i] for i in range(len(names))}
logit = np.asarray(name2.get("logit", outs[0])).reshape(-1)
label = np.asarray(name2.get("label_bool", outs[1])).reshape(-1)  # bool array

pred = label.astype(np.int64)
acc = (pred == expected).mean()
print("logit:", logit)
print("pred:", pred.tolist(), "acc:", round(float(acc), 2))