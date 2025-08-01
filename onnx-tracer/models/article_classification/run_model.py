import json
import numpy as np
import onnxruntime as ort
import sys
import re
import os

MAX_FEATURES = 1000  # must match training

# -------- Tokenization --------
def tokenize(text):
    """
    Split text into tokens: words and punctuation separately.
    Example: "Hello, how are you?" -> ["hello", ",", "how", "are", "you", "?"]
    """
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return tokens

# -------- Build input vector --------
def build_input_vector(text, idf_mapping):
    """
    Build input vector of length MAX_FEATURES:
    - Initialize with zeros
    - For each token: if in mapping, increment vec[index] by idf
    """
    vec = np.zeros(MAX_FEATURES, dtype=np.float32)
    tokens = tokenize(text)
    for token in tokens:
        if token in idf_mapping:
            idx = idf_mapping[token]["index"]
            idf_val = idf_mapping[token]["idf"]
            vec[idx] += idf_val
    return vec

# -------- Main inference --------
def main():
    # --- Input handling ---
    if len(sys.argv) < 2:
        # No direct text provided: try to load input.json
        if os.path.exists("input.json"):
            with open("input.json", "r") as f:
                input_data = json.load(f)
                if isinstance(input_data, dict) and "text" in input_data:
                    text = input_data["text"]
                else:
                    print("❌ input.json must contain: {\"text\": \"your text here\"}")
                    sys.exit(1)
        else:
            print("Usage:")
            print("  python run_model.py \"Your text here\"")
            print("  OR create input.json with {\"text\": \"Your text here\"}")
            sys.exit(1)
    else:
        text = sys.argv[1]

    # --- Load mapping ---
    if not os.path.exists("vocab.json"):
        print("❌ Missing vocab.json. Run training script first.")
        sys.exit(1)
    with open("vocab.json", "r") as f:
        idf_mapping = json.load(f)

    # --- Build input vector ---
    vec = build_input_vector(text, idf_mapping)

    # --- Load ONNX model ---
    if not os.path.exists("network.onnx"):
        print("❌ Missing network.onnx. Run training script first.")
        sys.exit(1)

    session = ort.InferenceSession("network.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # --- Run inference ---
    outputs = session.run(None, {input_name: vec.reshape(1, -1)})
    logits = outputs[0]

    # --- Decode prediction ---
    with open("labels.json", "r") as f:
        labels = json.load(f)

    if len(labels) == 0:
        print("❌ No labels found in labels.json. Run training script first.")
        sys.exit(1)
    

    pred_idx = int(np.argmax(logits))
    pred_label = labels['{}'.format(pred_idx)]

    print(f"Predicted label: {pred_label}")
    print(f"Logits: {logits}")

if __name__ == "__main__":
    main()
