#!/usr/bin/env python3
import json
import numpy as np
import torch
import torch.nn as nn

# ----- data -----
texts = [
    "I love this", "This is great", "Happy with the result",
    "I hate this", "This is bad", "Not satisfied"
]
labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32)

# build vocab (id 0 = pad)
vocab = {}
def tokenize(text):
    ids = []
    for tok in text.lower().split():
        if tok not in vocab:
            vocab[tok] = len(vocab) + 1
        ids.append(vocab[tok])
    return ids

max_len = 5
seqs = [tokenize(t) for t in texts]
arr = np.zeros((len(texts), max_len), dtype=np.int64)
for i, s in enumerate(seqs):
    arr[i, :min(len(s), max_len)] = s[:max_len]
X = torch.tensor(arr, dtype=torch.long)

# ----- model: only Gather, MatMul, Add, GE in ONNX -----
class TinyLinearSentiment(nn.Module):
    def __init__(self, vocab_size, L, t=0.0):
        super().__init__()
        self.S = nn.Embedding(vocab_size + 1, 1)     # -> Gather
        self.register_buffer("ones_col", torch.ones(L, 1))  # for sum via MatMul
        self.W = nn.Parameter(torch.zeros(1, 1))     # MatMul weight
        self.b = nn.Parameter(torch.zeros(1))        # Add bias
        self.register_buffer("thresh_t", torch.tensor([[float(t)]]))  # for GE

    def reset_padding_weight(self):
        with torch.no_grad():
            self.S.weight[0, 0] = 0.0  # pad contributes nothing

    def forward(self, x):
        # x: (B,L) int64
        # scores: (B,L,1) -> (B,L)
        scores = self.S(x).squeeze(-1)                     # Gather
        sum_score = torch.matmul(scores, self.ones_col)    # MatMul (B,L) x (L,1) -> (B,1)
        logit = torch.matmul(sum_score, self.W) + self.b   # MatMul + Add -> (B,1)
        label_bool = (logit >= self.thresh_t)              # GreaterOrEqual only
        return logit, label_bool

model = TinyLinearSentiment(len(vocab), L=max_len, t=0.0)
model.reset_padding_weight()

# ----- train -----
opt = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.BCEWithLogitsLoss()
for epoch in range(60):
    logit, _ = model(X)
    loss = loss_fn(logit.squeeze(1), labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 15 == 0:
        with torch.no_grad():
            _, yb = model(X)
            acc = (yb.squeeze(1).to(torch.int64) == labels.to(torch.int64)).float().mean().item()
        print(f"epoch {epoch:02d}  loss {loss.item():.4f}  acc {acc:.2f}")

# ----- save vocab (flat dict) -----
with open("vocab.json", "w") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

# ----- export ONNX -----
dummy = torch.randint(1, len(vocab)+1, (1, max_len), dtype=torch.long)
torch.onnx.export(
    model, dummy, "network.onnx",
    input_names=["tokens"],
    output_names=["logit", "label_bool"],  # bool output to avoid Cast (TODO(Forpee)): Do cast in post-processing)
    dynamic_axes={"tokens": {0: "batch"}, "logit": {0: "batch"}, "label_bool": {0: "batch"}},
    opset_version=15
)
print("Exported network.onnx and vocab.json")