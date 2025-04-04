---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/jondojish/NLU-Evidence-Detection

---

# Model Card for t98667jf-m21430ak-ED

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to
      detect whether two a given claim is supported by a given evidence


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a hybrid BERT + CNN architecture that was fine-tuned
      on 21K claim-evidence pairs. It integrates contextual embeddings from BERT with a CNN layer, consisting
      of multiple kernel sizes to better capture local n-gram patterns and relevance between claim and evidence.
      The outputs of the convolutional layers are passed through a max pooling layer before being passed through
      a dense layer before being passed through a final 2 neuron dense layer for classification.
      The model was inspired by the paper at https://www.sciencedirect.com/science/article/pii/S187705092300234X,
      demonstrating that integrating CNN with BERT can improve classification performance.

- **Developed by:** Jonathan Fesseha and Akram Kassim
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformer + CNN
- **Finetuned from model [optional]:** bert-base-uncased

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/google-bert/bert-base-uncased
- **Paper or documentation:** https://aclanthology.org/N19-1423.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

21K claim-evidence pairs.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 2e-05
      - batch_size: 16
      - seed: 42
      - num_epochs: 2
      - num_filters: 256
      - dense_units: 256
      - kernel_sizes: [3, 4, 5]
      - dropout_rate: 0.2

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 34 minutes
      - duration per training epoch: 17 minutes
      - model size: 1.2GB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

The full development set provided, amounting to almost 6K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy
      - Precision (Macro & Weighted)
      - Recall (Macro & Weighted)
      - F1-score (Macro & Weighted)

### Results

The model obtained:
      - Accuracy: 87.7%
      - Macro Precision: 84.4%
      - Macro Recall: 85.5%
      - Macro F1-score: 84.9%
      - Weighted Precision: 87.9%
      - Weighted Recall: 87.7%
      - Weighted F1-score: 87.8%
      

## Technical Specifications

### Hardware


      - RAM: at least 8 GB
      - Storage: at least 2GB,
      - GPU: T4

### Software


      - Transformers 4.48.3
      - tensorflow 2.18.0
      - optuna 4.2.1

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      256 subwords will be truncated by the model. The performance of the model may also reflect any biases in the training data

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Some hyperparameters (num_filters, dense_units, and dropout_rate) were tuned using Optuna, with values selected based
      on validation accuracy over two training epochs.
