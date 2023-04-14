"""
Alright, this is the file where I'm training the model with the data set.
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
import numpy as np

# Load the data from CSV files
train_df = pd.read_csv('Datasets/train.csv')
test_df = pd.read_csv('Datasets/test.csv')
valid_df = pd.read_csv('Datasets/valid.csv')

# Combine train, test, and valid data for text and intent

all_text = train_df['text'].append(test_df['text']).append(valid_df['text']).str.lower().tolist()
all_intent = train_df['intent'].append(test_df['intent']).append(valid_df['intent']).tolist()

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)


# Convert text data to sequences of numerical representations
def text_to_sequences(text):
    try:
        sequences = tokenizer.texts_to_sequences(text)
    except KeyError:
        # Handle unknown words by setting their index to 0 (reserved for padding)
        sequences = [[tokenizer.word_index.get(w, 0) for w in t.split()] for t in text]
    return sequences


train_sequences = text_to_sequences(train_df['text'].str.lower().tolist())
test_sequences = text_to_sequences(test_df['text'].str.lower().tolist())
valid_sequences = text_to_sequences(valid_df['text'].str.lower().tolist())

# Padding to ensure all input sequences have the same length
max_length = max(len(seq) for seq in train_sequences)
train_padded_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post')
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')
valid_padded_sequences = pad_sequences(valid_sequences, maxlen=max_length, padding='post')

vocab_size = len(tokenizer.word_index)

print(vocab_size)

# Encode intent labels as integers
intent_mapping = {intent: i for i, intent in enumerate(set(all_intent))}
y_train = np.array([intent_mapping[intent] for intent in train_df['intent']])
y_test = np.array([intent_mapping[intent] for intent in test_df['intent']])
y_valid = np.array([intent_mapping[intent] for intent in valid_df['intent']])

# Define the model
model = Sequential()
model.add(Embedding(input_dim=13084, output_dim=300, input_length=max_length))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=len(intent_mapping), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_padded_sequences, y_train, validation_data=(valid_padded_sequences, y_valid), epochs=5, batch_size=64)

# Evaluate the model
loss, accuracy = model.evaluate(test_padded_sequences, y_test, batch_size=64)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)


# Use the trained model to predict intents for user input


def predict_intent(model_, tokenizer_, input_text):
    input_text = input_text.lower()  # Convert input text to lowercase
    input_sequence = tokenizer_.texts_to_sequences([input_text])  # Convert input text to numerical representation
    input_padded_sequence = pad_sequences(input_sequence, maxlen=max_length, padding='post')  # Pad input sequence
    predicted_probs = model_.predict(input_padded_sequence)[0]  # Predict probabilities for each intent
    predicted_intent_index = tf.argmax(
        predicted_probs).numpy()  # Get index of predicted intent with the highest probability
    predicted_intent = list(intent_mapping.keys())[
        list(intent_mapping.values()).index(predicted_intent_index)]  # Get the corresponding intent label
    return predicted_intent


intent = predict_intent(model_=model, tokenizer_=tokenizer, input_text="Play Matsuri by Fujii Kaze")

print(intent)

model.save('my_model.h5')
