import torch
import torch.nn as nn
import numpy as np

# ----- dummy data (same as yours) -----
texts = [
    "I love this", "This is great", "Happy with the result",
    "I hate this", "This is bad", "Not satisfied"
]
labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32)

vocab = {}
def tokenize(text):
    tokens = text.lower().split()
    ids = []
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab) + 1
        ids.append(vocab[token])
    return ids

max_len = 5
input_ids = [tokenize(t) for t in texts]
padded = np.zeros((len(texts), max_len), dtype=np.int64)
for i, seq in enumerate(input_ids):
    padded[i, :len(seq)] = seq[:max_len]
X = torch.tensor(padded, dtype=torch.long)  # (batch, L)

# ----- save vocab so test.py can reuse it -----
import json

meta = {
    "pad_id": 0,
    "max_len": max_len,
}
with open("vocab.json", "w") as f:
    json.dump({"vocab": vocab, "meta": meta}, f, ensure_ascii=False, indent=2)

# ----- Model 0 -----
class Model0Sentiment(nn.Module):
    def __init__(self, vocab_size, L, C=4.0, tau=1.0, t=0.0):
        super().__init__()
        # S[V,1]: per-token scalar weight table (pad id 0 will be zeroed after init)
        self.S = nn.Embedding(vocab_size + 1, 1)  # (V+1, 1)

        # ones_col[L,1] to do sum via MatMul (B,L) x (L,1) -> (B,1)
        ones_col = torch.ones(L, 1, dtype=torch.float32)
        self.register_buffer("ones_col", ones_col)

        # tiny head: mixes [sum_score, pos_count, neg_count] -> 1 logit
        self.head = nn.Linear(3, 1)  # Gemm (MatMul + Add) in ONNX

        # constants as buffers so they export cleanly
        self.register_buffer("clip_C", torch.tensor(float(C)))
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.register_buffer("thresh_t", torch.tensor([[float(t)]]))  # shape [1,1] for GE

    def reset_padding_weight(self):
        with torch.no_grad():
            if self.S.weight.shape[0] > 0:
                self.S.weight[0, 0] = 0.0  # pad_id=0 contributes nothing

    def forward(self, x):
        # x: (B, L) int64
        B, L = x.shape

        # Gather(S): (B,L,1) -> squeeze -> (B,L)
        scores = self.S(x).squeeze(-1)

        # Clip(min=-C, max=+C)
        clipped = torch.clamp(scores, min=-self.clip_C.item(), max=self.clip_C.item())

        # Gates → bits
        pos_bits = (clipped >  self.tau).to(clipped.dtype)  # (B,L) in {0.0,1.0}
        neg_bits = (clipped < -self.tau).to(clipped.dtype)

        # Pools (counts): ReduceSum over tokens axis
        pos_count = pos_bits.sum(dim=1, keepdim=True)  # (B,1)
        neg_count = neg_bits.sum(dim=1, keepdim=True)  # (B,1)

        # Optional sum via MatMul with ones
        # (B,L) x (L,1) -> (B,1)
        sum_score = torch.matmul(clipped, self.ones_col)

        # Tiny head: concat features and project
        feats = torch.cat([sum_score, pos_count, neg_count], dim=1)  # (B,3)
        logit = self.head(feats)  # (B,1)

        # In-graph threshold to 0/1 (GreaterOrEqual)
        label = (logit >= self.thresh_t).to(torch.int64)  # (B,1)
        return logit, label

# ----- train -----
model = Model0Sentiment(len(vocab), L=max_len, C=4.0, tau=1.0, t=0.0)
model.reset_padding_weight()

optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(50):
    logit, _ = model(X)
    loss = criterion(logit.squeeze(1), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        with torch.no_grad():
            _, pred = model(X)
            acc = (pred.squeeze(1) == labels.long()).float().mean().item()
        print(f"Epoch {epoch:02d} | Loss {loss.item():.4f} | Acc {acc:.2f}")

# ----- export ONNX (two outputs: logit and label) -----
dummy_input = torch.randint(1, len(vocab)+1, (1, max_len), dtype=torch.long)
torch.onnx.export(
    model, dummy_input, "network.onnx",
    input_names=["tokens"],
    output_names=["logit", "label"],
    dynamic_axes={"tokens": {0: "batch"}, "logit": {0: "batch"}, "label": {0: "batch"}},
    opset_version=15
)
