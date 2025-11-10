import argparse
import numpy as np
import tensorflow as tf

def parse_arguments():
    parser = argparse.ArgumentParser(description="klasyfikator winiaczy")
    parser.add_argument("--alcohol", type=float, required=True)
    parser.add_argument("--malic_acid", type=float, required=True)
    parser.add_argument("--ash", type=float, required=True)
    parser.add_argument("--alcalinity", type=float, required=True)
    parser.add_argument("--magnesium", type=float, required=True)
    parser.add_argument("--total_phenols", type=float, required=True)
    parser.add_argument("--flavanoids", type=float, required=True)
    parser.add_argument("--nonflavanoid_phenols", type=float, required=True)
    parser.add_argument("--proanthocyanins", type=float, required=True)
    parser.add_argument("--color_intensity", type=float, required=True)
    parser.add_argument("--hue", type=float, required=True)
    parser.add_argument("--od280_od315", type=float, required=True)
    parser.add_argument("--proline", type=float, required=True)

    return parser.parse_args()

def main():
    args = parse_arguments()

    # wczytanie modelu
    model = tf.keras.models.load_model("models/model_B.keras")

    # wczytanie skalera
    mean = np.load("models/scaler_mean.npy")
    scale = np.load("models/scaler_scale.npy")

    # wczytanie wektora cech
    x = np.array([[args.alcohol,
                  args.malic_acid,
                  args.ash,
                  args.alcalinity,
                  args.magnesium,
                  args.total_phenols,
                  args.flavanoids,
                  args.nonflavanoid_phenols,
                  args.proanthocyanins,
                  args.color_intensity,
                  args.hue,
                  args.od280_od315,
                  args.proline]])

    # normalizacja
    x = (x - mean) / scale

    #predykcja
    pred = model.predict(x)
    cls = int(np.argmax(pred)) + 1 # powrót do rzeczywistych etykiet

    print(f"predykcja klasy: {cls}")
    print(f"prawdopodobieństwo: {pred}")

if __name__ == "__main__":
    main()