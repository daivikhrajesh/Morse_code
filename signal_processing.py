import numpy as np
import scipy.signal as sg  # Renaming to avoid conflict

def preprocess_signal(raw_signal):
    # Apply noise reduction
    filtered_signal = sg.medfilt(raw_signal)
    # Normalize signal
    normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
    return normalized_signal

def segment_signal(signal, dot_threshold=0.15, dash_threshold=0.35, space_threshold=0.55):
    segments = []
    current_segment = []
    
    for duration in signal:
        if duration <= dot_threshold:
            # Dot
            if current_segment and current_segment[-1] == '.':
                segments.append(current_segment)
                current_segment = []
            current_segment.append('.')
        elif duration <= dash_threshold:
            # Dash
            if current_segment and current_segment[-1] == '-':
                segments.append(current_segment)
                current_segment = []
            current_segment.append('-')
        elif duration >= space_threshold:
            # Space
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            segments.append([' '])  # Space between letters or words
    
    if current_segment:
        segments.append(current_segment)
    
    return segments

if __name__ == '__main__':
    raw_signal = np.load('../data/signals.npy', allow_pickle=True)
    # Flatten the array to process one signal at a time
    for signal in raw_signal:
        preprocessed_signal = preprocess_signal(signal)
        print("Preprocessed Signal:", preprocessed_signal)
        segments = segment_signal(preprocessed_signal)
        print("Segments:", segments)
