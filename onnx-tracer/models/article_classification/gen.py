import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------------
# Config
# --------------------------
VOCAB_SIZE = 1000
SVD_COMPONENTS = 100
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
SCALE_FACTOR = 1000  # For integer scaling of TF-IDF

# --------------------------
# 1. Load Dataset
# --------------------------
data = pd.read_csv("bbc_data.csv")  # columns: "data", "labels"
texts = data["data"].astype(str).tolist()
labels = data["labels"].tolist()

# --------------------------
# 2. TF-IDF and vocab creation
# --------------------------
vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE)
tfidf_matrix = vectorizer.fit_transform(texts)

# Save vocab
vocab = vectorizer.vocabulary_  # word -> index

# Get IDF and scale
idf_values = vectorizer.idf_
idf_scaled = (idf_values * SCALE_FACTOR).astype(int)

idf_mapping = {
    word: {"index": int(idx), "idf": int(idf_scaled[idx])}
    for word, idx in vocab.items()
}

with open("vocab.json", "w") as f:
    json.dump(idf_mapping, f)

# --------------------------
# 3. Encode Labels
# --------------------------
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, labels_encoded, test_size=0.2, random_state=42
)

# --------------------------
# 4. Apply SVD (Dim Reduction)
# --------------------------
svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
reduced_features = svd.fit_transform(X_train)

# Save SVD components for ONNX
svd_components = torch.tensor(svd.components_, dtype=torch.float32)
svd_mean = torch.tensor(svd.mean_, dtype=torch.float32) if hasattr(svd, "mean_") else torch.zeros(VOCAB_SIZE)


# --------------------------
# 5. PyTorch Model (SVD layer + classifier)
# --------------------------
class SVDClassifier(nn.Module):
    def __init__(self, svd_components, svd_mean, num_classes=5):
        super().__init__()
        self.svd_components = nn.Parameter(svd_components, requires_grad=False)
        self.svd_mean = nn.Parameter(svd_mean, requires_grad=False)
        self.fc = nn.Linear(svd_components.shape[0], num_classes)

    def forward(self, x):
        # Apply SVD projection: (x - mean) @ components.T
        x_centered = x - self.svd_mean
        x_reduced = torch.matmul(x_centered, self.svd_components.T)
        return self.fc(x_reduced)

model = SVDClassifier(svd_components, svd_mean, num_classes=len(label_encoder.classes_))

# --------------------------
# 6. Train
# --------------------------
train_dataset = TensorDataset(torch.tensor(X_train.toarray(), dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")


# ---- Evaluation ----
model.eval()
with torch.no_grad():
    # Forward pass on test set
    y_pred_logits = model(torch.tensor(X_test.toarray(), dtype=torch.float32))
    y_pred = torch.argmax(y_pred_logits, dim=1).numpy()

# Compute metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# --------------------------
# 7. Export ONNX
# --------------------------
dummy_input = torch.randn(1, VOCAB_SIZE)  # integer-scaled TF-IDF vector
torch.onnx.export(
    model,
    dummy_input,
    "network.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=13
)

# Save label mapping
with open("labels.json", "w") as f:
    json.dump({i: label for i, label in enumerate(label_encoder.classes_)}, f)
    print("✅ labels.json saved.")

print("✅ Training complete, ONNX and mappings saved.")
