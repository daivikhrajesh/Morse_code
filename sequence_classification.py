import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


# Load the generated signals and labels
signals = np.load('../data/signals.npy', allow_pickle=True)
labels = np.load('../data/labels.npy')

# Convert labels to indices
label_indices = [ord(label) - ord('A') for label in labels]

# Update signals: convert 1 to dot, 2 to dash, and 5 to space
def convert_signals(signals):
    converted = []
    for signal in signals:
        converted_signal = []
        for value in signal:
            if value == 1:
                converted_signal.append(0.1)  # Dot
            elif value == 2:
                converted_signal.append(0.3)  # Dash
            elif value == 5:
                converted_signal.append(0.5)  # Space
        converted.append(converted_signal)
    return converted

signals = convert_signals(signals)

# Pad sequences to a maximum length of 10
X = tf.keras.preprocessing.sequence.pad_sequences(signals, maxlen=10, value=0.5)  # Use 0.5 for padding
y = np.array(label_indices)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(10, 1)))
model.add(Dense(26, activation='softmax'))  # 26 classes for each letter
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('../models/morse_model.h5')
