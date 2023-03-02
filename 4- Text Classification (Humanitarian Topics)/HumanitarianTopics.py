import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

# Load data
df = pd.read_csv('Fire_trainingset_merged.csv')

# Print label value counts
print (df.Label.value_counts())

# Convert 'Text' column to string
df["Text"]= df["Text"].map(str)

# Tokenize text
tokenizer = Tokenizer(num_words=500000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Set model parameters
MAX_NB_WORDS = len(word_index) + 1
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 32
print(len(word_index))

# Convert text to sequences and pad
X = tokenizer.texts_to_sequences(df['Text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

# Convert labels to categorical variables
Y = pd.get_dummies(df['Label']).values

# Print label tensor shape
print('Shape of label tensor:', Y.shape)


# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10)

# Print shapes of training and test sets
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
print(X.shape[1])

# Build model
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train model
epochs = 40
batch_size = 32
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# Evaluate model on test set
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

# Define a function to predict the label for a given text
   
def predict_label(text):
    if isinstance(text, str):
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(padded)
        labels = ['caution_and_advice', 'displaced_people_and_evacuations', 'help_donation_recovery', 'infrastructure_and_utility_damage', 'injured_or_dead_people', 'other_relevant_information', 'sympathy_and_support']
        return labels[np.argmax(pred)]
        
    else:
        return ''

# Load the twitter dataset
twitter_dataset = pd.read_csv('Campfire_State_Preprocess.csv')

# Parallelize the for loop using joblib
labels = Parallel(n_jobs=-1)(delayed(predict_label)(row['Text_Stop']) for index, row in twitter_dataset.iterrows() if isinstance(row['Text_Stop'], str))

# Add the predicted labels to the twitter_dataset dataframe
twitter_dataset['Label'] = labels

# Save the updated dataset
twitter_dataset.to_csv('twitter_dataset_labeled.csv', index=False)