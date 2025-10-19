import tensorflow as tf
import numpy as np
import math

# zadanie1
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

# zadanie2    
@tf.function
def tf_rotation_matrix(angle_rad: float):
    a = tf.convert_to_tensor(angle_rad, tf.float64)
    c = tf.math.cos(a)
    s = tf.math.sin(a)
    return tf.stack([tf.stack([c, -s]), tf.stack([s, c])])
    
# zadanie3
def solve_linear(A, b):
    A_tf = tf.convert_to_tensor(A, tf.float64)
    b_tf = tf.convert_to_tensor(b, tf.float64)
    # tf.linalg.solve wymaga (n, m), jeżeli b jest 1D to zmienia na (n, 1)
    if tf.rank(b_tf) == 1:
        b_tf = tf.reshape(b_tf, (-1, 1))
        
    # rozwiązanie układu równań (n, m)
    x = tf.linalg.solve(A_tf, b_tf)
    
    # jeśli wejście było jendowymiarowe to spłaszczanie do (n,)
    if tf.rank(b_tf) == 1:
        x.tf.reshape(x, (-1,))
        
    return x
    
def main():
    
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
    
    A = [[3.0, 2.0],
         [1.0, 2.0]]
    b = [5.0, 5.0]

    x = solve_linear(A, b).numpy()
    print("zadanie 3")
    print("A =", np.array(A))
    print("b =", np.array(b))
    print("x =", np.array2string(x, precision=6, suppress_small=True))    
    
    
    return 0
    
if __name__ == "__main__":
    main()