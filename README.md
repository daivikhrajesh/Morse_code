
# Morse Code Classification Model

This repository contains a machine learning model for classifying Morse code signals into their corresponding characters. The model is implemented using TensorFlow and trained on a dataset of Morse code signals.

## Overview

The model takes Morse code signals as input, which are sequences of integers representing dots, dashes, and spaces. It then predicts the corresponding character (A-Z) based on the trained model.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Flask

You can install the required packages using:

```bash
pip install -r requirements.txt
```

### Repository Structure

- `data/`: Directory containing the Morse code dataset.
- `model/`: Directory containing the saved TensorFlow model.
- `scr/`:
  - `files for training data and running the application`
- `README.md`: This file.
- `templates/`:
  - `index.html`
- `requirements.txt`

### Training the Model

To train the model, run the following command:

```bash
python scr/reinforcement_learning.py
python src/sequence_classification.py
```

Ensure you have your dataset placed in the `data/` directory or update the script to point to the correct location.


### Results

The model is evaluated based on its accuracy in classifying Morse code signals. The results are printed in the webpage when runnning the flask.py file.

## Usage

You can use the trained model to classify new Morse code signals by loading the model and passing the signals through it. The model expects signals in the form of a list of integers.
