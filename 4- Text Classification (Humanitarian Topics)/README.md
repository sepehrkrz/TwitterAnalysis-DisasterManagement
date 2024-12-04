# BERT-Based Text Classification and Cross-Validation Pipeline

This repository contains a Python implementation for fine-tuning a BERT model for multi-class text classification using PyTorch. It employs cross-validation for model evaluation and predicts sentiment labels for a given dataset.

## Features
- **Data Preprocessing**: Tokenization and input preparation using the `transformers` library.
- **Model Training**: Fine-tuning `BertForSequenceClassification` on a labeled dataset.
- **Cross-Validation**: K-Fold cross-validation to evaluate model performance.
- **Prediction Pipeline**: Predicting sentiment labels for an unseen dataset.

## Prerequisites

To run this code, you need the following:

1. Python 3.7 or later.
2. CUDA-enabled GPU (optional but recommended).
3. Installed Python libraries:
    - `pandas`
    - `numpy`
    - `torch`
    - `transformers`
    - `scikit-learn`
    - `tqdm`

Install the required packages using:

```bash
pip install pandas numpy torch transformers scikit-learn tqdm
```
## File Structure

- `Fire_trainingset_merged.csv`: Training dataset containing `Text` and `Label` columns.
- `Campfire_State_Preprocess_Sentiments.csv`: Dataset for predictions after training.
- `twitter_dataset_labeled.csv`: Output dataset with predicted labels.

## Code Overview

### 1. Initialization
- Sets up the device for computation (`CPU` or `GPU`).
- Loads and preprocesses the training data (`Fire_trainingset_merged.csv`).

### 2. Tokenization
- Tokenizes the input text using `BertTokenizer` with truncation and padding to a maximum sequence length of 128 tokens.
- Converts the tokenized text into input IDs and attention masks for the BERT model.

### 3. Model Preparation
- Initializes a `BertForSequenceClassification` model with a number of output labels matching the dataset's classes.
- Optimizes using Adam optimizer with a learning rate of `2e-5`.

### 4. Cross-Validation
- Implements 5-fold cross-validation using `KFold` from `sklearn`.
- Trains the model on training folds and evaluates on validation folds.
- Tracks loss and accuracy for each fold and computes mean accuracy and standard deviation across folds.

### 5. Predictions
- After training, applies the fine-tuned model to classify unseen text data from `Campfire_State_Preprocess_Sentiments.csv`.
- Saves the predictions in `twitter_dataset_labeled.csv`.

## Usage

1. Place the training dataset (`Fire_trainingset_merged.csv`) in the project directory.
2. Run the script to train the model and validate it using 5-fold cross-validation.
3. After training, provide the dataset (`Campfire_State_Preprocess_Sentiments.csv`) for prediction.
4. The predictions will be saved in `twitter_dataset_labeled.csv`.

## Results
- The script prints the average accuracy and standard deviation of the model across all folds during cross-validation.
- The labeled dataset is saved for further analysis.

## Notes
- Ensure that the training and prediction datasets are in the correct format with necessary columns (`Text` and `Label` for training, and `Text_Stop` for prediction).
- Training requires a GPU for optimal performance.
- Adjust the hyperparameters, such as learning rate or batch size, if needed.

