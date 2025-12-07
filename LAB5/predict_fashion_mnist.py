#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

#etykiety klas z fashion_mnist, kolejność wg dokumnetacji - tak samo jak przy treningu
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


def load_and_preprocess_image(image_path: Path) -> np.ndarray:
    """
    Wczytuje obrazek, konwertuje do skali szarości, skaluje do 28x28,
    normalizuje do [0,1] i robi negatyw.
    Zwraca tablicę o kształcie (1, 28, 28, 1).
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")

    #wczytanie
    img = Image.open(image_path)

    #grayscale
    img = img.convert("L")  # 8-bit grayscale

    #skalowanie do 28x28 (jak w fashion_mnist)
    img = img.resize((28, 28))

    #do tablicy numpy
    img_array = np.array(img).astype("float32") / 255.0

    #negatyw: dataset ma białe na czarnym, więc odwracmy
    img_array = 1.0 - img_array

    #dodaj wymiar kanału i batcha: (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def parse_args():
    parser = argparse.ArgumentParser(
        description="Klasyfikacja obrazka przy użyciu wytrenowanego modelu fashion_mnist."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Ścieżka do pliku modelu (.keras).",
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Ścieżka do obrazka do klasyfikacji.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

    print(f"[INFO] Ładowanie modelu z: {model_path}")
    model = keras.models.load_model(model_path)

    print(f"[INFO] Wczytywanie i przetwarzanie obrazka: {image_path}")
    img_batch = load_and_preprocess_image(image_path)

    print("[INFO] Predykcja...")
    preds = model.predict(img_batch, verbose=0)[0]  # (10,)
    predicted_index = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(preds[predicted_index])

    print("=======================================")
    print(f"Plik: {image_path}")
    print(f"Klasa: {predicted_class} (indeks: {predicted_index})")
    print(f"Pewność: {confidence * 100:.2f}%")
    print("=======================================")

if __name__ == "__main__":
    main()
