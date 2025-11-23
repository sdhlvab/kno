import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.python.keras.utils.np_utils import to_categorical

learning_rate = 0.001
batch_size = 64
epochs = 100

# wczytanie danych z pliku csv
wine = pd.read_csv('wine.data', header=None)

# tasowanie
df = wine.sample(frac=1, random_state=42)

# pierwsza kolumna to kategoria win (1, 2, 3)
y = df.iloc[:, 0] - 1  # zamiana 1, 2, 3 na 0, 1, 2
X = df.iloc[:, 1:]

# one-hot encoding
y = to_categorical(y, num_classes=3)

# podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# konwersja na float32 - zalecane przez tf
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
input_dim = X_train.shape[1]

# ------------ normalizacja
# uczymy normalizator na zbiorze treningowym
base_normalizer = tf.keras.layers.Normalization(axis=-1, name="normalizacja_wejscia")
base_normalizer.adapt(X_train)
# zapamiętujemy wagi, żeby każdemu modelowi dać własną kopię
norm_weights = base_normalizer.get_weights()


# funkcja tworząca warstwę Normalization z wagami wyliczonymi z X_train
def make_normalizer(input_dim: int) -> tf.keras.layers.Normalization:
    layer = tf.keras.layers.Normalization(axis=-1, name="normalizacja_wejscia")
    # inicjalizacja kształtu, żeby można było wstrzyknąć wagi
    layer.build((None, input_dim))
    layer.set_weights(norm_weights)
    return layer


# ------------------ funkcja do tworzenia modeli - parametry
def build_model_from_params(
    input_dim: int,
    learning_rate: float = 0.001,
    units1: int = 32,
    units2: int = 16,
    dropout_rate: float = 0.0,
    activation="relu",
    l2_reg: float = 0.0,
    name: str = "Model",
) -> tf.keras.Model:

    inputs = tf.keras.Input(shape=(input_dim,), name="wejscie")
    x = make_normalizer(input_dim)(inputs)

    reg = tf.keras.regularizers.L2(l2_reg) if l2_reg > 0 else None

    x = tf.keras.layers.Dense(
        units1,
        activation=activation,
        kernel_regularizer=reg,
        name="warstwa1",
    )(x)

    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate, name="drop1")(x)

    x = tf.keras.layers.Dense(
        units2,
        activation=activation,
        kernel_regularizer=reg,
        name="warstwa2",
    )(x)

    outputs = tf.keras.layers.Dense(3, activation="softmax", name="wyjscie")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------

# ----------- odwzorowanie modeli z poprzedniego zjazdu za pomocą build_model_from_params
# pierwszy model A
# MLP - multi-layer perceptron, aktywacja oparta na relu
model_A = build_model_from_params(
    input_dim=input_dim,
    learning_rate=learning_rate,
    units1=32,
    units2=16,
    dropout_rate=0.0,
    activation="relu",
    l2_reg=0.0,
    name="Model_A",
)

model_A.summary()

hist_A = model_A.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, y_test),
)

# drugi model B
# głębszy MLP, aktywacja oparta na gelu, inicjalizacja na glorot
model_B = build_model_from_params(
    input_dim=input_dim,
    learning_rate=learning_rate,
    units1=64,
    units2=32,
    dropout_rate=0.15,
    activation=tf.keras.activations.gelu,
    l2_reg=0.0001,
    name="Model_B",
)

model_B.summary()

hist_B = model_B.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, y_test),
)

#---------------------------------------------

# zapis modeli
model_A.save("models/model_A.keras")
model_B.save("models/model_B.keras")

# wykres pierwszego modelu
plt.figure()
plt.plot(hist_A.history["accuracy"], label="train_accuracy")
plt.plot(hist_A.history["val_accuracy"], label="val_accuracy")
plt.title(
    f"model_A – accuracy (lr={learning_rate}, u1=32, u2=16, do=0.0)"
)
plt.xlabel("epoka")
plt.ylabel("dokładność")
plt.legend()
plt.savefig(
    f"results/model_A_acc_ep{epochs}_bs{batch_size}_lr{learning_rate}.png", dpi=200
)
plt.close()

plt.figure()
plt.plot(hist_A.history["loss"], label="train_loss")
plt.plot(hist_A.history["val_loss"], label="val_loss")
plt.title(
    f"model_A – loss (lr={learning_rate}, u1=32, u2=16, do=0.0)"
)
plt.xlabel("epoka")
plt.ylabel("strata")
plt.legend()
plt.savefig("results/model_A_loss.png", dpi=200)
plt.close()

# wykres drugiego modelu
plt.figure()
plt.plot(hist_B.history["accuracy"], label="train_accuracy")
plt.plot(hist_B.history["val_accuracy"], label="val_accuracy")
plt.title(
    f"model_B – accuracy (lr={learning_rate}, u1=64, u2=32, do=0.15, l2=1e-4)"
)
plt.xlabel("epoka")
plt.ylabel("dokładność")
plt.legend()
plt.savefig(
    f"results/model_B_acc_ep{epochs}_bs{batch_size}_lr{learning_rate}.png", dpi=200
)
plt.close()

plt.figure()
plt.plot(hist_B.history["loss"], label="train_loss")
plt.plot(hist_B.history["val_loss"], label="val_loss")
plt.title(
    f"model_B – loss (lr={learning_rate}, u1=64, u2=32, do=0.15, l2=1e-4)"
)
plt.xlabel("epoka")
plt.ylabel("strata")
plt.legend()
plt.savefig("results/model_B_loss.png", dpi=200)
plt.close()

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

# ------------------ funkcja do tworzenia modeli - hiperparametry z KT
# 3 różne parametry:
# - learning_rate
# - units1 (liczba neuronów w pierwszej warstwie)
# - dropout_rate

def build_model_hp(hp: kt.HyperParameters) -> tf.keras.Model:
    # pierwszy parametr - learning_rate
    lr = hp.Float(
        "learning_rate",
        min_value=0.0001,
        max_value=0.01,
        sampling="log",
    )

    # drugi parametr - liczba neuronów w pierwszej warstwie
    units1 = hp.Int(
        "units1",
        min_value=16,
        max_value=128,
        step=16,
    )

    # trzeci parametr - dropout_rate
    dropout_rate = hp.Choice(
        "dropout_rate",
        values=[0.0, 0.1, 0.2, 0.3],
    )

    # druga warstwa ma połowę neuronów z pierwszej warstwy
    units2 = units1 // 2

    # wykorzystanie struktury modelu drugiego, czyli gelu + L2
    model = build_model_from_params(
        input_dim=input_dim,
        learning_rate=lr,
        units1=units1,
        units2=units2,
        dropout_rate=dropout_rate,
        activation=tf.keras.activations.gelu,
        l2_reg=0.0001,
        name="model_tuned",
    )
    return model


# konfiguracja tunera
tuner = kt.RandomSearch(
    hypermodel=build_model_hp,
    # celem tunera jest minimalizacja błędu na zbiorze walidacyjnym
    objective="val_loss",
    # ilość kombinacji hiperparametrów
    max_trials=10,
    # ile razy powtarzać ten sam zestaw hiperparametrów
    executions_per_trial=2,
    # katalog z logami tunera
    directory="kt_dir",
    project_name="LAB4",
    overwrite=True,
)

tuner.search_space_summary()

# uruchomienie tuningu
tuner.search(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    # fragment X_train idzie na walidację dla tunera
    validation_split=0.2,
    verbose=1,
)

tuner.results_summary()

# najlepsze hiperparametry z tunera
best_hp = tuner.get_best_hyperparameters(1)[0]
hp_lr = best_hp.get("learning_rate")
hp_units1 = best_hp.get("units1")
hp_dropout = best_hp.get("dropout_rate")
hp_units2 = hp_units1 // 2

print("Najlepsze hiperparametry z tunera:")
print(f"  learning_rate = {hp_lr}")
print(f"  units1        = {hp_units1}")
print(f"  dropout_rate  = {hp_dropout}")

# budujemy najlepszy model na podstawie best_hp i trenujemy, żeby mieć hist_T
best_model = build_model_from_params(
    input_dim=input_dim,
    learning_rate=hp_lr,
    units1=hp_units1,
    units2=hp_units2,
    dropout_rate=hp_dropout,
    activation=tf.keras.activations.gelu,
    l2_reg=0.0001,
    name="model_tuned_best",
)

best_model.summary()

hist_T = best_model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_test, y_test),
)

# ocena na zbiorze testowym
test_loss_T, test_acc_T = best_model.evaluate(X_test, y_test, verbose=1)

# podusmowanie z wszytskich modeli
print(f"[model pierwszy] test_acc={test_acc_A:.4f}  test_loss={test_loss_A:.4f}")
print(f"[model drugi] test_acc={test_acc_B:.4f}  test_loss={test_loss_B:.4f}")
print(f"[tuned] test_acc={test_acc_T:.4f}  test_loss={test_loss_T:.4f}")

# zapis najlepszego modelu z tunera
best_model.save("models/model_tuned_best.keras")

# --------- wykresy dla modelu z hiperparametrami (tuned)
plt.figure()
plt.plot(hist_T.history["accuracy"], label="train_accuracy")
plt.plot(hist_T.history["val_accuracy"], label="val_accuracy")
plt.title(
    f"model_tuned – accuracy (lr={hp_lr:.5f}, u1={hp_units1}, u2={hp_units2}, do={hp_dropout})"
)
plt.xlabel("epoka")
plt.ylabel("dokładność")
plt.legend()
plt.savefig(
    f"results/model_Tuned_acc_lr{hp_lr:.5f}_u1{hp_units1}_do{hp_dropout}.png",
    dpi=200,
)
plt.close()

plt.figure()
plt.plot(hist_T.history["loss"], label="train_loss")
plt.plot(hist_T.history["val_loss"], label="val_loss")
plt.title(
    f"model_tuned – loss (lr={hp_lr:.5f}, u1={hp_units1}, u2={hp_units2}, do={hp_dropout})"
)
plt.xlabel("epoka")
plt.ylabel("strata")
plt.legend()
plt.savefig(
    f"results/model_Tuned_loss_lr{hp_lr:.5f}_u1{hp_units1}_do{hp_dropout}.png",
    dpi=200,
)
plt.close()

#--------macierze pomyłek

# y_true w postaci indeksów klas
y_true = np.argmax(y_test, axis=1)

# model pierwszy
y_pred_A = np.argmax(model_A.predict(X_test), axis=1)
cm_A = confusion_matrix(y_true, y_pred_A)
print("Confusion matrix – model pierwszy:")
print(cm_A)

fig, ax = plt.subplots()
disp_A = ConfusionMatrixDisplay(confusion_matrix=cm_A)
disp_A.plot(ax=ax)
ax.set_title("model pierwszy – confusion matrix")
plt.savefig("results/model_A_confusion_matrix.png", dpi=200)
plt.close()

# model drugi
y_pred_B = np.argmax(model_B.predict(X_test), axis=1)
cm_B = confusion_matrix(y_true, y_pred_B)
print("Confusion matrix – model drugi:")
print(cm_B)

fig, ax = plt.subplots()
disp_B = ConfusionMatrixDisplay(confusion_matrix=cm_B)
disp_B.plot(ax=ax)
ax.set_title("model drugi – confusion matrix")
plt.savefig("results/model_B_confusion_matrix.png", dpi=200)
plt.close()

# model z tunera
y_pred_T = np.argmax(best_model.predict(X_test), axis=1)
cm_T = confusion_matrix(y_true, y_pred_T)
print("Confusion matrix – model_tuned:")
print(cm_T)

fig, ax = plt.subplots()
disp_T = ConfusionMatrixDisplay(confusion_matrix=cm_T)
disp_T.plot(ax=ax)
ax.set_title("model_tuned – confusion matrix")
plt.savefig("results/model_Tuned_confusion_matrix.png", dpi=200)
plt.close()