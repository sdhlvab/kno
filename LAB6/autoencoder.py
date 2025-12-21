import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.src import layers, losses
from keras.src.models import Model

# =========================
# USTAWIENIA
# =========================
DATA_DIR = "data"
OUT_DIR = "out"
IMG_H, IMG_W = 128, 128
BATCH_SIZE = 8
EPOCHS = 50
LATENT_DIM = 2

os.makedirs(OUT_DIR, exist_ok=True)
AUTOTUNE = tf.data.AUTOTUNE


# =========================
# SAVE GRID
# =========================



def save_grid(images, path, cols=8):
    """images: [N,H,W,C] float [0..1]"""
    images = np.clip(images, 0.0, 1.0)
    n, h, w, c = images.shape
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * h, cols * w, c), dtype=np.float32)

    for i in range(n):
        r = i // cols
        cc = i % cols
        canvas[r*h:(r+1)*h, cc*w:(cc+1)*w, :] = images[i]

    tf.keras.utils.save_img(path, canvas)


# =========================
# DATASET Z KATALOGU
# =========================
def make_dataset():
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        labels=None,
        label_mode=None,
        image_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )

    #normalizacja do [0,1]
    ds = ds.map(lambda x: tf.cast(x, tf.float32) / 255.0, num_parallel_calls=AUTOTUNE)
    ds = ds.cache().prefetch(AUTOTUNE)
    return ds


# =========================
# AUGMENTACJA (tylko na wejściu)
# =========================
augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="augment",
)


# =========================
# MODEL
# =========================
class AutoencoderCNN(Model):
    """
    - self.encoder = Sequential(...)
    - self.decoder = Sequential(...)
    - call() zwraca decoded
    """

    def __init__(self, latent_dim, encoder0=None, decoder0=None):
        super().__init__()
        self.latent_dim = latent_dim

        #Encoder: 128x128x3 -> latent_dim
        if encoder0 is None:
            self.encoder = tf.keras.Sequential(
                [
                    layers.Input(shape=(IMG_H, IMG_W, 3)),

                    layers.Conv2D(32, 3, strides=2, padding="same"),
                    layers.BatchNormalization(),
                    layers.Activation("relu"),

                    layers.Conv2D(64, 3, strides=2, padding="same"),
                    layers.BatchNormalization(),
                    layers.Activation("relu"),

                    layers.Conv2D(128, 3, strides=2, padding="same"),
                    layers.BatchNormalization(),
                    layers.Activation("relu"),

                    layers.Conv2D(256, 3, strides=2, padding="same"),  # -> 8x8x256
                    layers.BatchNormalization(),
                    layers.Activation("relu"),

                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(latent_dim, name="z"),
                ],
                name="encoder",
            )
        else:
            self.encoder = encoder0

        if decoder0 is None:
            self.decoder = tf.keras.Sequential(
                [
                    layers.Input(shape=(latent_dim,)),

                    layers.Dense(8 * 8 * 256, activation="relu"),
                    layers.Reshape((8, 8, 256)),

                    layers.UpSampling2D(size=2),
                    layers.Conv2D(256, 3, padding="same"),
                    layers.BatchNormalization(),
                    layers.Activation("relu"),

                    layers.UpSampling2D(size=2),
                    layers.Conv2D(128, 3, padding="same"),
                    layers.BatchNormalization(),
                    layers.Activation("relu"),

                    layers.UpSampling2D(size=2),
                    layers.Conv2D(64, 3, padding="same"),
                    layers.BatchNormalization(),
                    layers.Activation("relu"),

                    layers.UpSampling2D(size=2),
                    layers.Conv2D(2, 3, padding="same"),
                    layers.BatchNormalization(),
                    layers.Activation("relu"),

                    # final: 3 kanały, [0..1]
                    #layers.Conv2D(3, 3, padding="same", activation="sigmoid"),
                    
                    #do uruchomienia na vm z GPU i zweryfikowania czy jest różnica względem linijki wyżej
                    layers.Flatten(),
                    layers.Dense(128*128*3, activation="sigmoid"),
                    layers.Reshape((128,128,3))
                ],
                name="decoder",
            )
        else:
            self.decoder = decoder0

    def call(self, x, training=False):
        z = self.encoder(x, training=training)
        x_hat = self.decoder(z, training=training)
        return x_hat


# =========================
# MAIN
# =========================
def main():
    ds = make_dataset()

    #wczytanie modeli z treningu, jeżeli są
    try:
        encoder = tf.keras.models.load_model(os.path.join(OUT_DIR, "model_encoder.keras"))
        decoder = tf.keras.models.load_model(os.path.join(OUT_DIR, "model_decoder.keras"))
        do_not_fit = True
        print("Loaded saved encoder/decoder -> skip fit")
    except Exception as e:
        encoder = None
        decoder = None
        do_not_fit = False
        print("No saved models yet -> will fit. Reason:", str(e))

    autoencoder = AutoencoderCNN(LATENT_DIM, encoder0=encoder, decoder0=decoder)
    autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

    #dataset dla autoenkodera, dodajemy augmentacje: (augmented_x, original_x)
    ds_ae = ds.map(lambda x: (augment(x, training=True), x), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    if not do_not_fit:
        autoencoder.fit(ds_ae, epochs=EPOCHS, shuffle=True)

    #podgląd: weź 1 batch, zapisz oryginał i rekonstrukcję
    for batch in ds.take(1):
        x = batch.numpy()  # [B,128,128,3]
        x_hat = autoencoder.predict(x, verbose=0)

        save_grid(x[:16], os.path.join(OUT_DIR, "input_grid.png"), cols=8)
        save_grid(x_hat[:16], os.path.join(OUT_DIR, "recon_grid.png"), cols=8)

        #wyświetlenie latent dla pierwszych pięciiu obrazków
        z = autoencoder.encoder.predict(x, verbose=0)
        print("Latents (first 5):")
        for i in range(min(5, len(z))):
            print(i, " -> ", z[i])

        #generowanie: siatka po latent 2D
        #zakres dobieramy z tego batcha (żeby nie startować w [0,0] bez sensu)
        mu = z.mean(axis=0)
        sd = z.std(axis=0) + 1e-6
        span = 2.0

        grid_n = 10
        xs = np.linspace(mu[0] - span * sd[0], mu[0] + span * sd[0], grid_n)
        ys = np.linspace(mu[1] - span * sd[1], mu[1] + span * sd[1], grid_n)

        Zgrid = np.array([[xx, yy] for yy in ys for xx in xs], dtype=np.float32)
        gen = autoencoder.decoder.predict(Zgrid, verbose=0)
        save_grid(gen, os.path.join(OUT_DIR, "generated_grid.png"), cols=grid_n)

        break

    #zapis encoder/decoder
    autoencoder.encoder.save(os.path.join(OUT_DIR, "model_encoder.keras"))
    autoencoder.decoder.save(os.path.join(OUT_DIR, "model_decoder.keras"))
    print("Saved models to:", OUT_DIR)
    print("Saved images: input_grid.png, recon_grid.png, generated_grid.png")


if __name__ == "__main__":
    main()
