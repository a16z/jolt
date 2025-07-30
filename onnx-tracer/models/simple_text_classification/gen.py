import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Dummy data
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
X = torch.tensor(padded)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
      emb = self.embed(x)         # (batch, seq, embed_dim)
      pooled = emb[:, 0, :]       # Use first token only
      x = F.relu(self.fc(pooled)) # Add ReLU
      return torch.sigmoid(x)     # Final activation

model = TextClassifier(len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

for epoch in range(20):
    out = model(X)
    loss = criterion(out.squeeze(), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Export float ONNX
dummy_input = torch.randint(1, len(vocab)+1, (1, max_len))
torch.onnx.export(
    model, dummy_input, "network.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=15
)
print("✅ Exported float32 ONNX model to network.onnx")
