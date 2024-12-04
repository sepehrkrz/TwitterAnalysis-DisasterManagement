import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(f"Using device: {device}")

# Load data
df = pd.read_csv('Fire_trainingset_merged.csv')

# Convert 'Text' column to string
df["Text"] = df["Text"].map(str)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize text
encoding = tokenizer(df['Text'].tolist(), padding='max_length', max_length=128, truncation=True, return_tensors='pt')

# Extract input IDs and attention masks
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# Convert labels to categorical
Y = pd.get_dummies(df['Label']).values
y_tensor = torch.tensor(Y, dtype=torch.float32)

num_labels = Y.shape[1]
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model.to(device)

# Compile model optimizer
optimizer = Adam(model.parameters(), lr=2e-5)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
losses = []
fold = 1
num_epochs = 20

for train_index, val_index in kf.split(input_ids):
    print(f"Training fold {fold}...")
    
    X_train_ids = input_ids[train_index]
    X_val_ids = input_ids[val_index]
    X_train_mask = attention_mask[train_index]
    X_val_mask = attention_mask[val_index]
    Y_train = y_tensor[train_index]
    Y_val = y_tensor[val_index]
    
    train_dataset = TensorDataset(X_train_ids, X_train_mask, Y_train)
    val_dataset = TensorDataset(X_val_ids, X_val_mask, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train model
    model.train()
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader, desc="Training", leave=False):
            b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]

            model.zero_grad()
            
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss

            loss.backward()

            optimizer.step()

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
            
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            val_loss += CrossEntropyLoss()(logits, b_labels).item()
            
            predictions = torch.argmax(logits, dim=-1)
            labels = torch.argmax(b_labels, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    
    accuracy = correct_predictions / total_predictions
    print(f"Fold {fold} - Loss: {val_loss:.3f}, Accuracy: {accuracy:.3f}")
    accuracies.append(accuracy)
    losses.append(val_loss)
    fold += 1

print(f"Average Accuracy: {np.mean(accuracies):.3f}")
torch.save(model.state_dict(), f'model_fold_{fold}.pt')

# Final results
mean_accuracy = np.mean(accuracies)
std_dev = np.std(accuracies)
print(f"Mean Accuracy: {mean_accuracy:.3f} (SD: {std_dev:.3f})")

# Apply model to full dataset
twitter_dataset = pd.read_csv('Campfire_State_Preprocess_Sentiments.csv')

# Function to predict label for each tweet
def predict_label(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get logits from model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().item()

    labels = ['caution_and_advice', 'displaced_people_and_evacuations', 'help_donation_recovery', 
              'infrastructure_and_utility_damage', 'injured_or_dead_people', 'other_relevant_information', 
              'sympathy_and_support']
    
    return labels[pred]

# Predict labels for each row in the dataset
labels = []
for index, row in tqdm(twitter_dataset.iterrows(), total=len(twitter_dataset)):
    if isinstance(row['Text_Stop'], str):
        labels.append(predict_label(row['Text_Stop']))
    else:
        labels.append(None)


# Add predicted labels to dataframe
twitter_dataset['Label'] = labels

# Save the labeled dataset
twitter_dataset.to_csv('twitter_dataset_labeled.csv', index=False)
