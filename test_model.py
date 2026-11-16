import tensorflow as tf
import pickle
import os

MODEL_PATH = "models/arabshield_lstm_v2.keras"
TOKENIZER_PATH = "models/arabshield_lstm_tokenizer_v2.pkl"

print("TensorFlow:", tf.__version__)

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

print("Model loaded successfully!")
print("Tokenizer loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)