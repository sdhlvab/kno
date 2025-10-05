import numpy as np
import tensorflow as tf
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description="Podaj sciezke do pliku!")
parser.add_argument("plik", help="sciezka do pliku")
args = parser.parse_args()


model = tf.keras.models.load_model("model.keras")

# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),   tf.keras.layers.Dense(128, activation='relu'),   tf.keras.layers.Dropout(0.2),   tf.keras.layers.Dense(10, activation='softmax') ])
# model.compile(optimizer='adam',               loss='sparse_categorical_crossentropy',               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5) # użyj verbose=0 jeśli jest problem z konsolą
# model.evaluate(x_test, y_test)
# model.save("model.keras")
print(f"plik wejsciowy: {args.plik}")

img = Image.open(args.plik).convert("L")
img = img.resize((28, 28), Image.Resampling.LANCZOS)
img_array = np.array(img).astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)
if img_array.mean() > 0.5:
    img_array = 1.0 - img_array

pred = model.predict(img_array, verbose=0)
digit = np.argmax(pred[0])

print(f"przewidywana cyfra: {digit}")
print(f"prawdopodobienstwo: {np.round(pred[0], 3)}")
