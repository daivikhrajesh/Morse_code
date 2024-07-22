import numpy as np

def extract_features(segments):
    features = []
    for segment in segments:
        if len(segment) == 1:
            if segment[0] == 5:
                features.append(' ')  # Space between letters
            else:
                features.append('.')  # Dot
        else:
            avg_duration = np.mean(segment)
            if avg_duration == 2:
                features.append('-')  # Dash
            else:
                features.append('.')  # Dot
    return features

if __name__ == '__main__':
    segments = [[1], [2], [1, 1, 2, 1]]
    features = extract_features(segments)
    print("Features:", features)
