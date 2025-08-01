# Medium Text Classification Example

Trained to recognize if given SMS text is a ham or spam.

Run `python gen.py` to create:

- `network.onnx`, the model.
- `vocab.json`, to help pre-process an input message into tokens, before feeding it to the model.

Run `python run_model.py '<input text>'` to run the model using an input text.

- Outputs model's prediction: ham or spam.

Run `python tokenize_text.py '<input text>'` to tokenize an input text.

- Outputs `input.json`, a model-friendly input vector.

Also contains:

- `vocab.json`, file holding the vocabulary issued from training the given model,

  Built from training with Spam SMS dataset: https://www.kaggle.com/datasets/mariumfaheem666/spam-sms-classification-using-nlp.

  Can be rebuilt by running `python gen.py` after linking to the desired dataset.
