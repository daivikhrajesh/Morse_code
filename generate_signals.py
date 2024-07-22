import numpy as np

morse_code = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..'
}

def generate_signal(letter):
    code = morse_code[letter]
    signal = []
    for symbol in code:
        if symbol == '.':
            signal.append(1)  # Dot
        elif symbol == '-':
            signal.append(2)  # Dash
        signal.append(1)  # Pause between parts of the same letter
    signal.append(5)  # Pause between letters
    return signal

signals = []
labels = []

for letter in morse_code.keys():
    signal = generate_signal(letter)
    signals.append(signal)
    labels.append(letter)

# Save signals and labels
np.save('../data/signals.npy', np.array(signals, dtype=object))
np.save('../data/labels.npy', np.array(labels))

if __name__ == '__main__':
    print("Signals and labels saved.")
