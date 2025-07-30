import json
import sys
import os
import numpy as np
import onnxruntime as ort

# --- Config ---
MAX_LEN = 100  # must match training
VOCAB_PATH = "vocab.json"
MODEL_PATH = "network.onnx"

# --- Load vocab ---
def load_vocab(vocab_path=VOCAB_PATH):
    if not os.path.exists(vocab_path):
        print(f"❌ Vocab file '{vocab_path}' not found.")
        print("👉 Please generate it first by running: python gen.py")
        sys.exit(1)


    with open(vocab_path, "r") as f:
        return json.load(f)

# --- Tokenize ---
def tokenize_sms(text, vocab, max_len=MAX_LEN):
    tokens = text.lower().split()
    ids = [vocab.get(token, 0) for token in tokens]  # 0 = unknown
    ids = ids[:max_len]
    padded = ids + [0] * (max_len - len(ids))       # post-padding
    return padded

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_model.py '<sms text>'")
        sys.exit(1)

    # Get SMS text from CLI
    sms_text = sys.argv[1]

    # Tokenize using same vocab
    vocab = load_vocab(VOCAB_PATH)
    tokenized = tokenize_sms(sms_text, vocab, MAX_LEN)

    # Convert to numpy array for ONNX
    input_array = np.array([tokenized], dtype=np.int64)  # batch of 1

    # Load ONNX model
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(None, {input_name: input_array})
    probs = outputs[0][0]  # probabilities for [negative, positive]

    # Interpret result
    prediction = "spam" if probs[1] > probs[0] else "ham"

    print(f"Tokenized: {tokenized}")
    print(f"Probabilities: {probs}")
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
