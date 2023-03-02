## Text Classification with LSTM Model
This code trains a Long Short-Term Memory (LSTM) model for text classification using the Keras library. It uses a dataset of text data containing tweets and their corresponding labels, and uses this data to train the model to classify new text data into one of seven categories of humanitarian topics.

### Required Libraries
- ***NumPy***: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- ***Pandas***: A library for data manipulation and analysis.
- ***Tensorflow***: A library for building and training machine learning models.
- ***Scikit-learn***: A library for machine learning algorithms and tools for data analysis and mining.
- ***Joblib***: A set of tools to provide lightweight pipelining in Python.

## Loading and Preprocessing Data
The code reads in a dataset stored in a CSV file (`Fire_trainingset_merged.csv`) using pandas. The label value counts are printed using the Pandas function `value_counts()`. The text data in the `Text` column is converted to a string format using the Pandas function `map()`.

```python
df["Text"]= df["Text"].map(str)
```

The text data is tokenized using the Keras Tokenizer class, which converts each word in the text to an integer. The maximum number of words to keep is set to 500000. The data is then converted into sequences of integers and padded to ensure all sequences have the same length.

```python
tokenizer = Tokenizer(num_words=500000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Text'].values)
word_index = tokenizer.word_index
```

The labels are converted to categorical variables using the `get_dummies()` function in Pandas. The data is then split into training and testing sets using the Scikit-learn function `train_test_split()`.
```python
MAX_SEQUENCE_LENGTH = 50
X = tokenizer.texts_to_sequences(df['Text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(df['Label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10)
```
## Building the LSTM Model
The LSTM model is built using the Keras Sequential model. The embedding layer takes the integer-encoded vocabulary and looks up the embedding vector for each word index. These vectors are learned as the model trains. The SpatialDropout1D layer randomly drops out (sets to zero) entire 1D feature maps in the input.

```python
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

The LSTM layer has 100 units and a dropout rate of 0.2 to prevent overfitting. The output layer has a softmax activation function and the loss function used is categorical crossentropy. The Adam optimizer is used to optimize the model parameters.
## Training and Evaluating the Model
The model is trained using the `fit()` function in Keras, with early stopping to prevent overfitting. The number of epochs is set to 40 and the batch size is set to 32.
```python
epochs = 40
batch_size = 32
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
```
The model is then evaluated on the test set using the `evaluate()` function in Keras, which returns the loss and accuracy of the model on the test set.
```python
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
```
## Predicting New Labels for Twitter Dataset
The trained model is used to predict the labels of a new Twitter dataset stored in a CSV file (`Campfire_State_Preprocess.csv`). The `predict_label()` function takes a text input, tokenizes it, and passes it to the trained model for prediction.

The `iterrows()` function in Pandas is used to loop through each row in the Twitter dataset and the `predict_label()` function is called for each text input. The `Parallel()` function from the Joblib library is used to parallelize the loop and speed up the processing.

The predicted labels are added to the Twitter dataset using the Pandas function `assign()`, and the updated dataset is saved to a new CSV file (`twitter_dataset_labeled.csv`) using the Pandas function `to_csv()`.





