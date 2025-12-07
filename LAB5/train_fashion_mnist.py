#!/usr/bin/env python

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#etykiety klas z fashion_mnist, kolejność wg dokumnetacji
CLASS_NAMES = [
    "T-shirt/top",  # 0
    "Trouser",      # 1
    "Pullover",     # 2
    "Dress",        # 3
    "Coat",         # 4
    "Sandal",       # 5
    "Shirt",        # 6
    "Sneaker",      # 7
    "Bag",          # 8
    "Ankle boot",   # 9
]


def load_data():
    """
    Wczytuje dane Fashion-MNIST z tf.keras.datasets
    Zwraca: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    #dodanie wymiaru kanału: (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    #etykiety jako int32
    y_train = y_train.astype("int32")
    y_test = y_test.astype("int32")

    return (x_train, y_train), (x_test, y_test)


def build_model(arch: str = "dense", use_augmentation: bool = False) -> keras.Model:
    """
    Buduje i kompiluje model Keras zgodnie z wybraną architekturą.
    Przygotowane tak, żeby dało się łatwo podpiąć Keras Tuner
    (np. przez podanie hp i sterowanie liczbą neuronów/warstw).

    arch: "dense" lub "cnn"
    use_augmentation: czy dodać warstwę augmentacji
    """
    input_shape = (28, 28, 1)

    #warstwa skalująca – dane 0-255 -> 0-1
    rescale = layers.Rescaling(1.0 / 255.0, name="rescale")

    #warstwa augmentacji
    data_augmentation = None
    if use_augmentation:
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal", name="aug_flip"),
                layers.RandomRotation(0.1, name="aug_rot"),
                layers.RandomZoom(0.1, name="aug_zoom"),
            ],
            name="data_augmentation",
        )

    inputs = keras.Input(shape=input_shape, name="input")
    x = rescale(inputs)
    if data_augmentation is not None:
        x = data_augmentation(x)

    if arch == "dense":
        #prosta sieć w pełni połączona
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(256, activation="relu", name="dense_1")(x)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        x = layers.Dense(128, activation="relu", name="dense_2")(x)
        x = layers.Dropout(0.3, name="dropout_2")(x)
    elif arch == "cnn":
        #sieć splotowa
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(x)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)

        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)

        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3")(x)
        x = layers.MaxPooling2D((2, 2), name="pool3")(x)

        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(128, activation="relu", name="dense_cnn")(x)
        x = layers.Dropout(0.4, name="dropout_cnn")(x)
    else:
        raise ValueError(f"Nieznana architektura: {arch}. Użyj 'dense' lub 'cnn'.")

    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"fashion_mnist_{arch}")

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def compute_confusion_matrix(y_true: np.ndarray, y_pred_probs: np.ndarray) -> np.ndarray:
    """
    Liczy macierz pomyłek na podstawie prawdziwych etykiet i predykcji.
    """
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=10)
    return cm.numpy()


def save_metrics(
    output_dir: Path,
    history: keras.callbacks.History,
    test_loss: float,
    test_acc: float,
    confusion_matrix: np.ndarray,
) -> None:
    """
    Zapisuje metryki do pliku JSON.
    """
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "history": history.history,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": CLASS_NAMES,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Zapisano metryki do: {metrics_path}")


def train_and_save(
    arch: str,
    epochs: int,
    batch_size: int,
    use_augmentation: bool,
    output_dir: Path,
) -> None:
    """
    Cała logika treningu + zapis modelu i metryk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Wczytywanie danych Fashion-MNIST...")
    (x_train, y_train), (x_test, y_test) = load_data()

    print(f"[INFO] Dane treningowe: {x_train.shape}, etykiety: {y_train.shape}")
    print(f"[INFO] Dane testowe: {x_test.shape}, etykiety: {y_test.shape}")

    print(f"[INFO] Budowanie modelu: arch={arch}, augmentacja={use_augmentation}...")
    model = build_model(arch=arch, use_augmentation=use_augmentation)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    print("[INFO] Start treningu...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    print("[INFO] Ewaluacja na zbiorze testowym...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[RESULT] Test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")

    print("[INFO] Liczenie macierzy pomyłek...")
    y_test_pred_probs = model.predict(x_test, verbose=0)
    cm = compute_confusion_matrix(y_test, y_test_pred_probs)

    #zapis metryk
    save_metrics(output_dir, history, test_loss, test_acc, cm)

    #zapis modelu
    model_path = output_dir / f"fashion_mnist_{arch}.keras"
    model.save(model_path)
    print(f"[INFO] Zapisano model do: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trening modelu klasyfikującego obrazki fashion_mnist."
    )
    parser.add_argument(
        "--arch",
        choices=["dense", "cnn"],
        default="dense",
        help="Architektura modelu: 'dense' (MLP) lub 'cnn' (splotowa).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Liczba epok treningu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Rozmiar batcha.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Jeśli podane, użyje warstwy augmentacji danych.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models_fashion_mnist",
        help="Katalog wyjściowy na model i metryki.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    #dla powtarzalności
    tf.random.set_seed(42)
    np.random.seed(42)

    train_and_save(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_augmentation=args.augment,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
