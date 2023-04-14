import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('NLP.h5')

train_df = pd.read_csv('Datasets/train.csv')
test_df = pd.read_csv('Datasets/test.csv')
valid_df = pd.read_csv('Datasets/valid.csv')

all_text = train_df['text'].append(test_df['text']).append(valid_df['text']).str.lower().tolist()
all_intent = train_df['intent'].append(test_df['intent']).append(valid_df['intent']).tolist()


def text_to_sequences(text):
    try:
        sequences = Tokenizer().texts_to_sequences(texts=text)
    except KeyError:
        # Handle unknown words by setting their index to 0 (reserved for padding)
        sequences = [[Tokenizer().word_index.get(w, 0) for w in t.split()] for t in text]

    return sequences


train_sequences = text_to_sequences(train_df['text'].str.lower().tolist())
test_sequences = text_to_sequences(test_df['text'].str.lower().tolist())
valid_sequences = text_to_sequences(valid_df['text'].str.lower().tolist())

# Padding to ensure all input sequences have the same length
max_length = max(len(seq) for seq in train_sequences)

intent_mapping = {intent: i for i, intent in enumerate(set(all_intent))}


class IntentRecogniser:
    def __init__(self, model_=model, tokeniser_=Tokenizer()):
        self.model_ = model_
        self.tokenizer_ = tokeniser_

    def predict_intent(self, input_text):
        input_text = input_text.lower()  # Convert input text to lowercase
        input_sequence = self.tokenizer_.texts_to_sequences(
            [input_text])  # Convert input text to numerical representation
        input_padded_sequence = pad_sequences(input_sequence, maxlen=max_length, padding='post')  # Pad input sequence
        predicted_probs = self.model_.predict(input_padded_sequence)[0]  # Predict probabilities for each intent
        predicted_intent_index = tf.argmax(
            predicted_probs).numpy()  # Get index of predicted intent with the highest probability
        predicted_intent = list(intent_mapping.keys())[
            list(intent_mapping.values()).index(predicted_intent_index)]  # Get the corresponding intent label
        return predicted_intent


intent_recogniser = IntentRecogniser()

intent_recogniser.predict_intent("Play G4L by Giga.")