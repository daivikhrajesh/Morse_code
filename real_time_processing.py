import numpy as np
import tensorflow as tf
from signal_processing import preprocess_signal, segment_signal
from feature_extraction import extract_features

model = tf.keras.models.load_model('../models/morse_model.h5')

def real_time_processing(signal):
    preprocessed_signal = preprocess_signal(signal)
    segments = segment_signal(preprocessed_signal)
    features = extract_features(segments)
    input_sequence = [1 if f == '.' else 2 if f == '-' else 5 for f in features]
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences([input_sequence], maxlen=10)
    input_sequence = np.expand_dims(input_sequence, -1)
    prediction = model.predict(input_sequence)
    letter = chr(np.argmax(prediction) + ord('A'))
    return letter

if __name__ == '__main__':
    real_time_signal = np.array([1, 1, 5, 2, 1, 1, 5, 2, 1])  # Example for 'B'
    letter = real_time_processing(real_time_signal)
    print("Predicted Letter:", letter)
