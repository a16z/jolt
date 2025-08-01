# BBC Articles Text Classification Example

Trained to recognize the topic of a given article as one of 5 classes: sport, business, politics, tech, entertainment.

Run `python gen.py` to create:

- `network.onnx`, the ONNX model.
- `vocab.json`, the integer-scaled IDF mapping, used to pre-process input articles before feeding to the model.

Run `python run_model.py '<input article text>'` to run the model on an input article.

- Outputs model's predicted topic tag.

Run `python tokenize_text.py '<input article text>'` to tokenize an input text.

- Outputs `input.json`, a model-friendly input vector.

Also contains:

- `vocab.json`, file holding the integer-scaled IDF values for the vocabulary

  Built from training with BBC articles dataset: https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification.

  Can be rebuilt by running `python gen.py` after linking to the desired dataset.
