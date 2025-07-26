import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Dummy data (unchanged)
texts = [
    "I love this", "This is great", "Happy with the result",
    "I hate this", "This is bad", "Not satisfied"
]
labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.long)  # switch to long for CE

# build vocab & inputs (same as before)
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
X = torch.tensor(padded)

class MediumTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        # (1) Gather: embedding lookup
        self.embed = nn.Embedding(vocab_size + 1, embed_dim)
        # a tiny learnable scale for Mul example
        self.scale = nn.Parameter(torch.tensor(1.0))
        # two FC layers → MatMult
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: (batch, seq)
        emb = self.embed(x)                       # Gather
        emb_t = emb.transpose(1, 2)               # Transpose
        pooled = emb_t.mean(dim=2)                # ReduceMean over seq
        # Simple LayerNorm:
        mu = pooled.mean(dim=1, keepdim=True)     # ReduceMean over features
        centered = pooled - mu                    # Sub
        var = (centered.pow(2)).mean(dim=1, keepdim=True)  # Pow + ReduceMean
        normed = centered / torch.sqrt(var + 1e-5)        # Add eps + Sqrt + Div
        normed = normed * self.scale              # Mul

        h = F.relu(self.fc1(normed))              # MatMult + Relu
        logits = self.fc2(h)                      # MatMult
        probs = F.softmax(logits, dim=1)          # Softmax
        return probs

# instantiate, train
model = MediumTextClassifier(len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    out = model(X)
    loss = criterion(out, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Export to ONNX (float32)
dummy_input = torch.randint(1, len(vocab)+1, (1, max_len))
torch.onnx.export(
    model, dummy_input, "network.onnx",
    input_names=["input"], output_names=["probs"],
    dynamic_axes={"input": {0: "batch_size"}, "probs": {0: "batch_size"}},
    opset_version=15
)
print("✅ Exported ONNX model with richer ISA ops to network.onnx")