# Comment Toxicity Model

## Introduction

This repository contains a deep learning model for detecting toxicity in comments. The model is trained using TensorFlow and leverages an LSTM-based architecture to classify comments into various toxicity categories. The aim is to identify and filter out toxic comments to foster a healthy online environment.

## Dataset

The dataset used for training the model is sourced from [Kaggle's Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). It contains comments from Wikipedia's talk page edits, labeled for various types of toxicity. This rich dataset is designed to help build models capable of identifying toxic comments and includes the following categories:

- **Toxic**: Comments that are rude, disrespectful, or otherwise likely to make people leave a discussion.
- **Severe Toxic**: Extremely offensive comments.
- **Obscene**: Comments containing obscene language.
- **Threat**: Comments that contain threats of violence or harm.
- **Insult**: Comments that contain insulting language.
- **Identity Hate**: Comments that contain hate speech targeting individuals or groups based on identity characteristics such as race, religion, gender, etc.

### Dataset Composition

The dataset comprises:
- **Train Data**: The dataset contains 159,571 comments, each labeled with various toxicity categories. For the purpose of model development, this data is further split into training, testing, and validation subsets, as detailed in the [model training file](https://github.com/katakampranav/Comment-Toxicity-Model/blob/main/model.ipynb).

### Data Distribution

The data is imbalanced, with a higher proportion of non-toxic comments compared to toxic ones. This imbalance poses a challenge and requires careful handling during model training to ensure that the model does not become biased towards predicting non-toxicity.

### Example Entries

Here are a few example entries from the dataset:

| Comment | Toxic | Severe Toxic | Obscene | Threat | Insult | Identity Hate |
|---------|-------|--------------|---------|--------|--------|---------------|
| "You're such a stupid idiot." | 1 | 0 | 0 | 0 | 1 | 0 |
| "I love this!" | 0 | 0 | 0 | 0 | 0 | 0 |
| "This is a terrible idea, and you're an idiot for suggesting it." | 1 | 0 | 0 | 0 | 1 | 0 |

By leveraging this dataset, the model aims to accurately identify and classify various forms of toxic comments, contributing to a safer and more respectful online discourse.


## Model Architecture

The model uses an LSTM-based neural network with the following key components:
- **Embedding Layer**: Converts text data into dense vector representations.
- **Bidirectional LSTM Layer**: Captures dependencies in both forward and backward directions.
- **Fully Connected Layers**: Extracts features from the LSTM output.
- **Output Layer**: Six sigmoid neurons corresponding to the six toxicity categories.

## Training and Evaluation

The model is trained using the binary cross-entropy loss function and the Adam optimizer. Precision, recall, and categorical accuracy metrics are used to evaluate the model's performance on the test set.

### Training Time and Performance

Training this model was time-consuming due to the complexity of the LSTM architecture and the size of the dataset. It took approximately 10 hours to train the model for just 10 epochs, and the accuracy achieved was only around 70%. Although this level of accuracy is not so worst but pretty good, it highlights the challenges involved in training deep learning models on large datasets.

![Screenshot 2024-08-01 232437](https://github.com/user-attachments/assets/e4eab0f1-6424-4fd2-9e6a-872b40104239)

To achieve better accuracy, a more robust environment and the use of a pretrained model can significantly improve results. I am currently exploring such approaches to enhance the model's performance. 

### Improving Accuracy

To improve the model's accuracy, consider the following:
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and optimizer settings.
- **Data Augmentation**: Use techniques to generate more training data or balance the dataset.
- **Model Architecture**: Try more complex architectures or additional layers to capture more features.

### Why Not Google Colab?

While Google Colab offers free GPU access, I chose to run the training locally due to limitations in Colab's runtime duration and the need for a more stable environment. Training on a local machine with sufficient resources ensured uninterrupted training sessions.

## Dependencies

The project requires the following dependencies:
- TensorFlow
- Pandas
- Numpy
- Gradio
which are already mentioned in requirements.txt file

You can install these dependencies using the following command:

```sh
pip install -r requirements.txt
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

## Gradio Results

Below are some example results from the Gradio interface. These results demonstrate the model's ability to classify comments into different toxicity categories. (Images to be added later)

### *These are examples of a toxic comment.*

![Screenshot 2024-08-01 232051](https://github.com/user-attachments/assets/f284c642-5e90-4ec4-858d-d334a1368ad9)

![Screenshot 2024-08-01 232239](https://github.com/user-attachments/assets/0d48326b-ca9a-4d6b-ac43-484b5d74ddd4)

### *This is an example of a non-toxic comment.*

![Screenshot 2024-08-01 232339](https://github.com/user-attachments/assets/955937fe-46a1-475e-9426-deec29baa1a3)

## Author

This Comment Toxicity Model was developed by :
-	[@katakampranav](https://github.com/katakampranav)
-	Repository : https://github.com/katakampranav/Comment-Toxicity-Model

## Feedback

For any feedback or queries, please reach out to me at katakampranavshankar@gmail.com.
