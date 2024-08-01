# Comment Toxicity Model

## Introduction

This repository contains a deep learning model for detecting toxicity in comments. The model is trained using TensorFlow and leverages an LSTM-based architecture to classify comments into various toxicity categories. The aim is to identify and filter out toxic comments to foster a healthy online environment.

## Dataset

The dataset used for training the model is sourced from [Kaggle's Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). It contains comments from Wikipedia's talk page edits, labeled for various types of toxicity.

## Model Architecture

The model uses an LSTM-based neural network with the following key components:
- **Embedding Layer**: Converts text data into dense vector representations.
- **Bidirectional LSTM Layer**: Captures dependencies in both forward and backward directions.
- **Fully Connected Layers**: Extracts features from the LSTM output.
- **Output Layer**: Six sigmoid neurons corresponding to the six toxicity categories.

## Training and Evaluation

The model is trained using the binary cross-entropy loss function and the Adam optimizer. Precision, recall, and categorical accuracy metrics are used to evaluate the model's performance on the test set.

## Dependencies

The project requires the following dependencies:
- TensorFlow
- Pandas
- Numpy
- Gradio

You can install these dependencies using the following command:

```sh
pip install tensorflow pandas numpy gradio
```

## Usage

### Training

To train the model, run the training script provided in the repository. This will load the dataset, preprocess the text, and train the LSTM model.

### Evaluation

The evaluation script will load the trained model and evaluate its performance on the test set using precision, recall, and accuracy metrics.

### Inference

The model can be deployed using Gradio for an interactive interface. Users can input a comment, and the model will predict the toxicity categories.

### Deployment with Gradio

You can launch the Gradio interface by running the deployment script. This interface allows users to input comments and receive toxicity scores in real-time.

## Repository Structure

- `data/`: Directory containing the dataset files.
- `scripts/`: Directory containing training, evaluation, and deployment scripts.
- `models/`: Directory for saving trained models.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model experimentation.
- `README.md`: Project documentation.

## Author

This project was developed by [katakampranav](https://github.com/katakampranav).

## Feedback

For any feedback or queries, please reach out to me at katakampranavshankar@gmail.com.
