import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import json
from collections import Counter

# =====================
# 1. Load dataset
# =====================
df = pd.read_csv("Spam_SMS.csv")  # Change to your actual CSV filename

# Map status: "ham" -> 0, "spam" -> 1
label_map = {"ham": 0, "spam": 1}
df["label"] = df["Class"].map(label_map)

# Extract data
texts = df["Message"].astype(str).tolist()
labels = torch.tensor(df["label"].values, dtype=torch.long)

# Split into train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# =====================
# 2. Tokenization & Vocab
# =====================

max_len = 100  # max number of words per SMS

def build_vocab(texts, max_vocab_size=1000):
    # Count all tokens
    counter = Counter()
    for text in texts:
        tokens = text.lower().split()
        counter.update(tokens)

    # Take top max_vocab_size tokens
    most_common = counter.most_common(max_vocab_size)
    
    # Assign IDs starting from 1 (0 reserved for padding/unknown)
    vocab = {word: idx+1 for idx, (word, _) in enumerate(most_common)}
    return vocab

def pad_sequences(texts):
    input_ids = []
    for t in texts:
        tokens = t.lower().split()
        ids = []
        for token in tokens:
            ids.append(vocab.get(token, 0))  # 0 for unknown tokens
        input_ids.append(ids)

    padded = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, seq in enumerate(input_ids):
        padded[i, :min(len(seq), max_len)] = seq[:max_len]
    return torch.tensor(padded)

vocab = build_vocab(train_texts, max_vocab_size=1000)

X_train = pad_sequences(train_texts)
X_test = pad_sequences(test_texts)
y_train = train_labels
y_test = test_labels

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

# =====================
# 4. Training
# =====================
model = MediumTextClassifier(len(vocab))
class_counts = torch.bincount(y_train)
total = len(y_train)
weights = total / (2.0 * class_counts.float())  # inverse frequency
print("Class distribution:", class_counts.tolist())
print("Loss weights:", weights.tolist())

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    model.train()
    out = model(X_train)
    loss = criterion(out, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Training accuracy
    preds = out.argmax(dim=1)
    acc = (preds == y_train).float().mean().item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Train Acc: {acc:.2f}")

# =====================
# 5. Evaluation
# =====================
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_preds = test_logits.argmax(dim=1)

# Metrics
acc = accuracy_score(y_test, test_preds)
prec = precision_score(y_test, test_preds)
rec = recall_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds)

print("\nTest Metrics:")
print(f"Accuracy:  {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall:    {rec:.2f}")
print(f"F1-score:  {f1:.2f}")

preds = model(X_test).argmax(dim=1)
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds, target_names=["ham", "spam"]))

# =====================
# Save vocabulary
# =====================
with open("vocab.json", "w") as f:
    json.dump(vocab, f)

# =====================
# 6. Export to ONNX
# =====================
dummy_input = torch.randint(1, len(vocab)+1, (1, max_len))
torch.onnx.export(
    model, dummy_input, "network.onnx",
    model, dummy_input, "network.onnx",
    input_names=["input"], output_names=["probs"],
    dynamic_axes={"input": {0: "batch_size"}, "probs": {0: "batch_size"}},
    opset_version=15
)
print("\n✅ Exported ONNX model to network.onnx")