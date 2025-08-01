import json
import sys

MAX_LEN = 100  # must match value used during training

def load_vocab(vocab_path="vocab.json"):
    with open(vocab_path, "r") as f:
        return json.load(f)

def tokenize_sms(text, vocab, max_len=MAX_LEN):
    tokens = text.lower().split()
    ids = [vocab.get(token, 0) for token in tokens]  # 0 = unknown
    ids = ids[:max_len]
    padded = ids + [0] * (max_len - len(ids))       # post-padding
    return padded

def main():
    if len(sys.argv) < 2:
        print("Usage: python tokenize_sms.py '<sms text>' [output_json]")
        sys.exit(1)

    sms_text = sys.argv[1]
    # Default output file is input.json
    output_path = sys.argv[2] if len(sys.argv) > 2 else "input.json"

    # Load vocab and tokenize
    vocab = load_vocab("vocab.json")
    tokenized = tokenize_sms(sms_text, vocab, MAX_LEN)

    # Save to JSON format expected by EZKL
    input_data = {"input_data": [tokenized]}
    with open(output_path, "w") as f:
        json.dump(input_data, f)

    print(f"✅ Tokenized SMS saved to {output_path}")
    print(f"Tokenized: {tokenized}")

if __name__ == "__main__":
    main()