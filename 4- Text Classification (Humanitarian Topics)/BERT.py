import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping
from transformers import AdamW
from joblib import Parallel, delayed
import tensorflow as tf

# Load data
df = pd.read_csv('Fire_trainingset_merged.csv')

# Print label value counts
print(df.Label.value_counts())

# Convert 'Text' column to string
df["Text"] = df["Text"].map(str)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize text
def tokenize_text(text):
    return tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors='tf')

X = list(df['Text'].values)
X_tokenized = [tokenize_text(text) for text in X]

# Convert labels to categorical
Y = pd.get_dummies(df['Label']).values

# Define BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# Compile model
optimizer = AdamW(learning_rate=2e-5, epsilon=1e-8)
model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Setup early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
fold = 1

# 5-fold cross-validation
for train_index, val_index in kf.split(X_tokenized):
    print(f"Training fold {fold}...")
    
    X_train = [X_tokenized[i] for i in train_index]
    X_val = [X_tokenized[i] for i in val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]
    
    # Train model
    history = model.fit(
        [x['input_ids'] for x in X_train], Y_train, 
        validation_data=([x['input_ids'] for x in X_val], Y_val),
        epochs=6, 
        batch_size=32, 
        callbacks=[early_stopping]
    )
    
    # Evaluate model on validation set
    loss, accuracy = model.evaluate([x['input_ids'] for x in X_val], Y_val)
    print(f"Fold {fold} - Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")
    accuracies.append(accuracy)
    fold += 1

# Final results
mean_accuracy = np.mean(accuracies)
std_dev = np.std(accuracies)
print(f"Mean Accuracy: {mean_accuracy:.3f} (SD: {std_dev:.3f})")

# Apply model to full dataset
twitter_dataset = pd.read_csv('Campfire_State_Preprocess.csv')

# Function to predict label for each tweet
def predict_label(text):
    inputs = tokenizer(text, return_tensors="tf", padding="max_length", max_length=128, truncation=True)
    logits = model(inputs)
    pred = tf.argmax(logits.logits, axis=1).numpy()
    labels = ['caution_and_advice', 'displaced_people_and_evacuations', 'help_donation_recovery', 'infrastructure_and_utility_damage', 'injured_or_dead_people', 'other_relevant_information', 'sympathy_and_support']
    return labels[pred[0]]

# Parallelize predictions using joblib
labels = Parallel(n_jobs=-1)(delayed(predict_label)(row['Text_Stop']) for index, row in twitter_dataset.iterrows() if isinstance(row['Text_Stop'], str))

# Add predicted labels to dataframe
twitter_dataset['Label'] = labels

# Save the labeled dataset
twitter_dataset.to_csv('twitter_dataset_labeled.csv', index=False)
