# NeuroDigits — Drawing-Only Handwritten Digit Recognition

A beginner-friendly machine-learning practical using a TensorFlow/Keras feedforward neural network and Streamlit.

## Features
- Draw a handwritten digit on the canvas (the only input method).
- Display the predicted digit and confidence percentage.
- Show a 28 × 28 processed input preview; no graphs and no image-upload feature.
- Train from the included Kaggle Digit Recognizer dataset.

## Requirements
- Windows 64-bit and Python 3.14.
- Setup installs TensorFlow 2.22.0rc0 (a pre-release build used here for Python 3.14 compatibility) and the remaining dependencies.
- Internet connection required for the initial dependency installation.

## Run for the first time
1. Extract the ZIP to a short path such as `C:\NeuroDigits`.
2. Double-click `setup.bat` and wait for it to finish.
3. Double-click `train.bat` and wait for training to create `digit_recognizer.keras`.
4. Double-click `run.bat`.
5. Open `http://localhost:8501` if the browser does not open automatically.

## Later use
After setup and training have completed successfully, double-click only `run.bat`. Do not delete `.venv314` or `digit_recognizer.keras`.

## Model
- Input: 784 normalized pixel values (28 × 28).
- Hidden layers: Dense(128, ReLU), Dropout(0.15), Dense(64, ReLU).
- Output: 10 softmax scores for digits 0–9.
- Training: Adam, sparse categorical cross-entropy, stratified 80/20 train/validation split, up to 10 epochs with early stopping.

The confidence is the model's softmax score, not a guarantee of correctness. The trained model is generated locally by `train.bat` and is not bundled in this source ZIP.
