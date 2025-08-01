import json
import sys
import os
import numpy as np
import onnxruntime as ort

# --- Config ---
MAX_LEN = 100  # must match training
VOCAB_PATH = "vocab.json"
MODEL_PATH = "network.onnx"
INPUT_JSON = "input.json"

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

# --- Load tokenized input from JSON ---
def load_input_json(path=INPUT_JSON):
    if not os.path.exists(path):
        print(f"❌ No text argument provided and '{path}' not found.")
        print("Usage options:")
        print("  1. python run_model.py '<sms text>'   # direct text input")
        print("  2. python tokenize_text.py '<sms text>' && python run_model.py   # pre-tokenized input.json")
        sys.exit(1)

    with open(path, "r") as f:
        return json.load(f)["input_data"]

def main():
    # Determine input method
    if len(sys.argv) >= 2:
        # Text input mode
        sms_text = sys.argv[1]
        vocab = load_vocab(VOCAB_PATH)
        tokenized = tokenize_sms(sms_text, vocab, MAX_LEN)
        input_array = np.array([tokenized], dtype=np.int64)  # batch=1
    else:
        # Pre-tokenized JSON mode
        tokenized_data = load_input_json(INPUT_JSON)
        input_array = np.array(tokenized_data, dtype=np.int64)

    # Load ONNX model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file '{MODEL_PATH}' not found. Train/export it first.")
        sys.exit(1)

    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(None, {input_name: input_array})
    probs = outputs[0][0]  # [ham, spam]

    # Prediction
    prediction = "spam" if probs[1] > probs[0] else "ham"

    print(f"Probabilities: {probs}")
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
