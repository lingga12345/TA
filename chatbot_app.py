import streamlit as st
import numpy as np
import pandas as pd
import json
import string
import nltk
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import LabelEncoder
import random
import time

nltk.download('punkt')

# Load the trained model
model = load_model('model/trained_model95.h5')
tokenizer = pickle.load(open('model/tokenizers95.pickle', 'rb'))
le = pickle.load(open('model/le95.pickle', 'rb'))

# Load dataset
with open('dataset1.json') as content:
    data1 = json.load(content)

# Process dataset
tags = []
inputs = []
responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])

data = pd.DataFrame({"patterns": inputs, "tag": tags})
data = data.sample(frac=1)

train = tokenizer.texts_to_sequences(data['patterns'])
x_train = pad_sequences(train)

# Define input shape
input_shape = x_train.shape[1]

# Main Streamlit app
def main():
    st.title("Layanan Informasi TA")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.write("")  # Spacing

    # Accept user input
    user_input = st.chat_input("Masukan pertanyaan")
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    if  user_input:
        texts_p = []
        prediction_input = user_input

        prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
        prediction_input = ''.join(prediction_input)
        texts_p.append(prediction_input)

        prediction_input = tokenizer.texts_to_sequences(texts_p)
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input], input_shape)

        output = model.predict(prediction_input)
        output = output.argmax()

        response_tag = le.inverse_transform([output])[0]
        bot_response = random.choice(responses[response_tag])
        
        # simulasi mengetik
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in bot_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

if __name__ == "__main__":
    main()
