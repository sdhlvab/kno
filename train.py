import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.utils.np_utils import to_categorical

learning_rate = 0.001
batch_size = 64
epochs = 100

# wczytanie danych z pliku csv
wine = pd.read_csv('wine.data', header=None)

# tasowanie
df = wine.sample(frac=1, random_state=42)

# pierwsza kolumna to kategoria win (1, 2, 3)
y = df.iloc[:, 0] - 1 # zamiana 1, 2, 3 na 0, 1, 2
X = df.iloc[:, 1:]

# one hot encoding
y = to_categorical(y, num_classes=3)
#y = pd.get_dummies(y, prefix="kategoria", dtype='int')

# podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# normalizacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# pierwszy model A
# MLP - multi-layer perceptron, aktywacja oparta na relu
model_A = tf.keras.Sequential(name="Model_A")
model_A.add(tf.keras.layers.Input(shape=(13,), name="wejscie"))
model_A.add(tf.keras.layers.Dense(32, activation="relu", name="warstwa1"))
model_A.add(tf.keras.layers.Dense(16, activation="relu", name="warstwa2"))
model_A.add(tf.keras.layers.Dense(3, activation="softmax", name="wyjscie"))

model_A.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_A.summary()

# trenowanie modelu pierwszego
hist_A = model_A.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, y_test)
)

# drugi model B
# głębszy MLP, aktywacja oparta na gelu, inicjalizacja na glorot
l2 = tf.keras.regularizers.L2(1e-4)
model_B = tf.keras.Sequential(name="Model_B")
model_B.add(tf.keras.layers.Input(shape=(13,), name="wejscie"))
model_B.add(tf.keras.layers.Dense(64, kernel_initializer="glorot_uniform", kernel_regularizer=l2, name="warstwa1"))
model_B.add(tf.keras.layers.Activation(tf.keras.activations.gelu, name="gelu1"))
model_B.add(tf.keras.layers.Dropout(0.15, name="drop1"))
model_B.add(tf.keras.layers.Dense(32, kernel_initializer="glorot_uniform", kernel_regularizer=l2, name="warstwa2"))
model_B.add(tf.keras.layers.Activation(tf.keras.activations.gelu, name="gelu2"))
model_B.add(tf.keras.layers.Dropout(0.1, name="drop2"))
model_B.add(tf.keras.layers.Dense(16, kernel_initializer="glorot_uniform", kernel_regularizer=l2, name="warstwa3"))
model_B.add(tf.keras.layers.Activation(tf.keras.activations.gelu, name="gelu3"))
model_B.add(tf.keras.layers.Dense(3, activation="softmax", name="wyjscie"))

model_B.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_B.summary()

# trenowanie modelu drugiego
hist_B = model_B.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, y_test)
)

# zapis obu modeli
model_A.save("models/model_A.keras")
model_B.save("models/model_B.keras")

# wykres pierwszego modelu
plt.figure()
plt.plot(hist_A.history["accuracy"], label="train_accuracy")
plt.plot(hist_A.history["val_accuracy"], label="val_accuracy")
plt.title("model_A – accuracy")
plt.xlabel("epoka")
plt.ylabel("dokładność")
plt.legend()
plt.savefig(f"results/model_A_acc_ep{epochs}_bs{batch_size}_lr{learning_rate}.png", dpi=200)
plt.close()

plt.figure()
plt.plot(hist_A.history["loss"], label="train_loss")
plt.plot(hist_A.history["val_loss"], label="val_loss")
plt.title("model A – loss")
plt.xlabel("epoka")
plt.ylabel("strata")
plt.legend()
plt.savefig("results/model_A_loss.png", dpi=200)
plt.close()

# wykres drugiego modelu
plt.figure()
plt.plot(hist_B.history["accuracy"], label="train_accuracy")
plt.plot(hist_B.history["val_accuracy"], label="val_accuracy")
plt.title("model_B – accuracy")
plt.xlabel("epoka")
plt.ylabel("dokładność")
plt.legend()
plt.savefig(f"results/model_B_acc_ep{epochs}_bs{batch_size}_lr{learning_rate}.png", dpi=200)
plt.close()

plt.figure()
plt.plot(hist_B.history["loss"], label="train_loss")
plt.plot(hist_B.history["val_loss"], label="val_loss")
plt.title("model B – loss")
plt.xlabel("epoka")
plt.ylabel("strata")
plt.legend()
plt.savefig("results/model_B_loss.png", dpi=200)
plt.close()

# zapis skalera
np.save("models/scaler_mean.npy", scaler.mean_.astype(np.float32))
np.save("models/scaler_scale.npy", scaler.scale_.astype(np.float32))

# wskazanie lepszego modelu
test_loss_A, test_acc_A = model_A.evaluate(X_test, y_test, verbose=0)
test_loss_B, test_acc_B = model_B.evaluate(X_test, y_test, verbose=0)

print(f"[A] test_acc={test_acc_A:.4f}  test_loss={test_loss_A:.4f}")
print(f"[B] test_acc={test_acc_B:.4f}  test_loss={test_loss_B:.4f}")

if test_acc_A >= test_acc_B:
    best_name = "model_A"
else:
    best_name = "model_B"

print(f"Lepszy model: {best_name}")