import json
import numpy as np
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
    return re.findall(r"\w+|[^\w\s]", text.lower())

# -------- Build input vector --------
def build_input_vector(text, idf_mapping):
    """
    Create vector of length MAX_FEATURES:
    - Initialize with zeros
    - For each token: if in mapping, increment vec[index] by idf
    """
    vec = np.zeros(MAX_FEATURES, dtype=np.int32)
    tokens = tokenize(text)
    for token in tokens:
        if token in idf_mapping:
            idx = idf_mapping[token]["index"]
            idf_val = idf_mapping[token]["idf"]
            vec[idx] += idf_val
    return vec

# -------- Main CLI --------
def main():
    if len(sys.argv) < 2:
        print("Usage: python tokenize_text.py '<text>' [output_json]")
        print("Default output file: input.json")
        sys.exit(1)

    text = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "input.json"

    # Load mapping
    if not os.path.exists("vocab.json"):
        print("❌ Missing vocab.json. Run training script first.")
        sys.exit(1)
    with open("vocab.json", "r") as f:
        idf_mapping = json.load(f)

    # Build vector
    vec = build_input_vector(text, idf_mapping)

    # Save in EZKL input format
    input_data = {"input_data": [vec.tolist()]}
    with open(output_path, "w") as f:
        json.dump(input_data, f)

    print(f"✅ Tokenized vector saved to {output_path}")
    print(f"Vector (non-zero entries): {[i for i,v in enumerate(vec) if v!=0]}")

if __name__ == "__main__":
    main()
