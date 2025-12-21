import os
import numpy as np
import tensorflow as tf

ENCODER_PATH = "out/model_encoder.keras"
DECODER_PATH = "out/model_decoder.keras"
DATA_DIR = "data/"
OUT_DIR = "out/generated_like"

IMG_SIZE = (128,128)
BATCH_SIZE = 8
N_GEN = 20
TEMPERATURE = 0.8   # 0.5–1.2

os.makedirs(OUT_DIR, exist_ok=True)

encoder = tf.keras.models.load_model(ENCODER_PATH)
decoder = tf.keras.models.load_model(DECODER_PATH)

#wczytywanie danych jak przy treningu
ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels=None,
    label_mode=None,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
ds = ds.map(lambda x: tf.cast(x, tf.float32) / 255.0)

#liczenie latent mean/std
Z = []
for batch in ds:
    z = encoder.predict(batch, verbose=0)
    Z.append(z)
Z = np.concatenate(Z, axis=0)

mu = Z.mean(axis=0)
sd = Z.std(axis=0) + 1e-6

print("latent mean:", mu)
print("latent std :", sd)

#losowanie sensownych latentów
Z_gen = np.random.normal(mu, sd * TEMPERATURE, size=(N_GEN, 2)).astype(np.float32)
imgs = decoder.predict(Z_gen, verbose=0)
imgs = np.clip(imgs, 0.0, 1.0)

for i, img in enumerate(imgs):
    tf.keras.utils.save_img(os.path.join(OUT_DIR, f"gen_{i:03d}.png"), img)

print("Saved to:", OUT_DIR)
