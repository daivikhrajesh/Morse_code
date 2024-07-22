import numpy as np
import tensorflow as tf
from transformers import pipeline

model = tf.keras.models.load_model('../models/morse_model.h5')
nlp = pipeline("fill-mask", model="bert-base-uncased")

def post_process_text(sequence):
    sequence = " ".join(sequence)
    corrected_text = nlp(sequence)
    return corrected_text

if __name__ == '__main__':
    sequence = ["H", "E", "L", "L", "O"]
    corrected_text = post_process_text(sequence)
    print("Corrected Text:", corrected_text)
