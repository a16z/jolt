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

# ----- model: Gather -> ReduceSum -> (Mul + Add) -> GE -----
class TinyReduceSumSentiment(nn.Module):
    def __init__(self, vocab_size, L, t=0.0):
        super().__init__()
        self.S = nn.Embedding(vocab_size + 1, 1)   # Gather
        self.w = nn.Parameter(torch.zeros(1))      # scalar weight (Mul)
        self.b = nn.Parameter(torch.zeros(1))      # scalar bias (Add)
        self.register_buffer("thresh_t", torch.tensor([[float(t)]]))  # for GE

    def reset_padding_weight(self):
        with torch.no_grad():
            self.S.weight[0, 0] = 0.0  # pad contributes nothing

    def forward(self, x):
        # x: (B,L) int64
        scores = self.S(x).squeeze(-1)            # (B,L) from Gather
        sum_score = scores.sum(dim=1, keepdim=True)  # (B,1) ReduceSum(axis=1, keepdims=1)
        logit = sum_score * self.w + self.b       # (B,1) Mul + Add
        label_bool = (logit >= self.thresh_t)     # (B,1) GreaterOrEqual
        return logit, label_bool

model = TinyReduceSumSentiment(len(vocab), L=max_len, t=0.0)
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

# ----- export ONNX: ONE OUTPUT (label_bool), fixed batch size = 1 -----
class ExportOnlyLabel(nn.Module):
    def __init__(self, inner): 
        super().__init__(); self.inner = inner
    def forward(self, x):
        _, y = self.inner(x)
        return y  # (B,1) bool

export_model = ExportOnlyLabel(model).eval()
dummy = torch.randint(1, len(vocab)+1, (1, max_len), dtype=torch.long)

torch.onnx.export(
    export_model, dummy, "network.onnx",
    input_names=["tokens"],
    output_names=["label_bool"],
    opset_version=15,
    # fixed (1, L) I/O — no dynamic_axes
)
print("Exported network.onnx (one output, ReduceSum head) and vocab.json")