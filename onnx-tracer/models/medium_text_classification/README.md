# Medium Text Classification Example

Trained to recognised if given SMS text is a ham or spam.

Run `python gen.py` to create:

- `network.onnx`, the model.
- `vocab.json`, to help pre-process an input message into tokens, before feeding it to the model.

Run `python run_model.py '<input text>'` to run the model using an input text.

- Outputs model's prediction: ham or spam.

Also contains:

- `Spam_SMS.csv`, training data for the model, can be used or replaced to re-train the model.
- `vocab.json`, file holding the vocabulary issued from training the given model, can be rebuilt by running `python gen.py`.
