import sys
import tensorflow as tf
import numpy as np
import math
import argparse


# --------zadanie1---------
def rotation_matrix(angle_rad: float):
    """
    wzór na obrót macierzy

    R(a) = [[cos(a), -sin(a)],
            [sin(a),  cos(a)]]
    """
    a = tf.convert_to_tensor(angle_rad, tf.float64)
    c = tf.math.cos(a)
    s = tf.math.sin(a)
    return tf.stack([tf.stack([c, -s]), tf.stack([s, c])])


# --------zadanie2---------
@tf.function
def tf_rotation_matrix(angle_rad: float):
    a = tf.convert_to_tensor(angle_rad, tf.float64)
    c = tf.math.cos(a)
    s = tf.math.sin(a)
    return tf.stack([tf.stack([c, -s]), tf.stack([s, c])])


# --------zadanie3---------
@tf.function
def solve_linear(A, b):
    A_tf = tf.convert_to_tensor(A, tf.float64)
    b_tf = tf.convert_to_tensor(b, tf.float64)
    # tf.linalg.solve wymaga (n, m), jeżeli b jest 1D to zmienia na (n, 1)
    b_was_vector = tf.rank(b_tf) == 1
    if b_was_vector:
        b_tf = tf.reshape(b_tf, (-1, 1))

    # rozwiązanie układu równań (n, m)
    x = tf.linalg.solve(A_tf, b_tf)

    # jeśli wejście było jendowymiarowe to spłaszczanie do (n,)
    if b_was_vector:
        x = tf.reshape(x, (-1,))

    return x


# --------zadanie4---------
# parsowanie
def parse_vector(s):
    if not s:
        raise ValueError("nie podano wektora b")
    s = s.replace(" ", "")
    czesci = s.split(",")
    liczby = []
    for element in czesci:
        if element == "":
            continue
        liczby.append(float(element))

    return liczby


def parse_matrix(s):
    if not s:
        raise ValueError("nie podano macierzy A")
    s = s.replace(" ", "")
    wiersze = s.split(";")
    macierz = []
    for w in wiersze:
        if w == "":
            continue
        liczby = [float(x) for x in w.split(",") if x != ""]
        macierz.append(liczby)

    return macierz


def solve_linear_arg(A, b):
    A_tf = tf.convert_to_tensor(A, tf.float64)
    b_tf = tf.convert_to_tensor(b, tf.float64)

    # sprawdzenie kształtu macierzy
    n, m = A_tf.shape
    if n != m:
        raise ValueError("macierz A musi być kwadratowa")

    if len(b) != n:
        raise ValueError("długość wektora b nie pasuje do rozmiaru macierzy A")

    # sprawdzenie czy macierz ma pełny rząd
    rank = int(tf.linalg.matrix_rank(A_tf).numpy())
    if rank != n:
        raise ValueError(
            "macierz A nie ma pełnego rzędu (nie ma jednoznacznego rozwiązania)"
        )

    # rozwiązywanie A * x = b
    b_tf = tf.reshape(b_tf, (n, 1))
    x = tf.linalg.solve(A_tf, b_tf)
    x = tf.reshape(x, (n,))

    return x


def main() -> int:

    # zmiana stopni na radiany
    angle_deg = 90
    angle_rad = math.radians(angle_deg)

    R = rotation_matrix(angle_rad)
    tf_R = tf_rotation_matrix(angle_rad)

    print("zadanie 1")
    print("macierz obrotu dla", angle_deg, "stopni:")
    print(R.numpy())

    print("zadanie 2")
    print("tf.function macierz obrotu dla", angle_deg, "stopni:")
    print(tf_R.numpy())

    A = [[3.0, 2.0], [1.0, 2.0]]
    b = [5.0, 5.0]

    x = solve_linear(A, b).numpy()
    print("zadanie 3")
    print("A =", np.array(A))
    print("b =", np.array(b))
    print("x =", np.array2string(x, precision=6, suppress_small=True))

    print("zadanie4")
    parser = argparse.ArgumentParser(
        description="rozwiazywanie ukladu rownan z podanymi argumentami"
    )
    parser.add_argument("--A", type=str, required=True, help="macierz np. '3,2;1,2'")
    parser.add_argument("--b", type=str, required=True, help="wektor np. '5,5'")
    args = parser.parse_args()

    try:
        A = parse_matrix(args.A)
        b = parse_vector(args.b)
        wynik = solve_linear_arg(A, b)
        print("rozwiązanie układu równań liniowych:")
        print(np.array2string(wynik.numpy(), precision=6, suppress_small=True))
        return 0
    except ValueError as e:
        print("błąd danych:", e, file=sys.stderr)
        return 2
    except Exception as e:
        print("nieoczekiwany błąd:", e, file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
