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
    Wczytuje obrazek, konwertuje do skali szarości, skaluje do 28x28
    i (opcjonalnie) robi negatyw w taki sposób, żeby pasowało do treningu.

    Zwraca tablicę o kształcie (1, 28, 28, 1) w skali 0–255,
    bo w modelu jest jeszcze warstwa Rescaling(1/255).
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")

    #wczytanie, grayscale, zmiana rozdzielczości
    img = Image.open(image_path)
    img = img.convert("L")          # 8-bit grayscale
    img = img.resize((28, 28))      # jak w fashion_mnist

    #do tablicy numpy
    img_array = np.array(img).astype("float32")  # zakres 0–255

    #negatyw w skali 0–255 (tylko jeśli jasne tło)
    mean_val = img_array.mean()
    if mean_val > 128:
        img_array = 255.0 - img_array

    #dodanie wymiarów: (28,28) -> (28,28,1) -> (1,28,28,1)
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
