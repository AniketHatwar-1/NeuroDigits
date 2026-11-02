from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = Path(__file__).resolve().parent
DATA_ZIP = ROOT / "data" / "digit-recognizer.zip"
DATA_DIR = ROOT / "data" / "extracted"
MODEL_PATH = ROOT / "digit_recognizer.keras"
REPORT_PATH = ROOT / "training_report.txt"

def main():
    if not DATA_ZIP.exists():
        raise FileNotFoundError(
            f"Dataset ZIP not found: {DATA_ZIP}\n"
            "Place digit-recognizer.zip inside the data folder."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_csv = DATA_DIR / "train.csv"
    if not train_csv.exists():
        print("Extracting the supplied dataset...")
        with zipfile.ZipFile(DATA_ZIP, "r") as zf:
            member = next((n for n in zf.namelist() if n.endswith("train.csv")), None)
            if member is None:
                raise FileNotFoundError("train.csv was not found inside the dataset ZIP.")
            with zf.open(member) as source, open(train_csv, "wb") as target:
                target.write(source.read())

    print("Loading the provided Kaggle training CSV...")
    df = pd.read_csv(train_csv)
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in train.csv.")
    y = df["label"].to_numpy(dtype=np.int64)
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32, copy=True)

    if X.shape[1] != 784:
        raise ValueError(f"Expected 784 pixel columns, found {X.shape[1]}.")
    if not np.isfinite(X).all():
        raise ValueError("The dataset contains missing or non-finite pixel values.")

    X = X / 255.0  # Create a writable normalized array
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nTraining feedforward neural network on your dataset only...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )

    probs = model.predict(X_val, batch_size=256, verbose=0)
    pred = np.argmax(probs, axis=1)
    accuracy = accuracy_score(y_val, pred)
    report = classification_report(y_val, pred, digits=4)
    matrix = confusion_matrix(y_val, pred)

    model.save(MODEL_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("NeuroDigits training report\n")
        f.write("===========================\n")
        f.write(f"Validation records: {len(y_val)}\n")
        f.write(f"Validation accuracy: {accuracy:.4%}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(matrix))
        f.write("\n")

    print(f"\nValidation accuracy: {accuracy:.2%}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved evaluation report: {REPORT_PATH}")
    print("Training complete. Now run run.bat.")

if __name__ == "__main__":
    main()
