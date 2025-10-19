import tensorflow as tf
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
    
    
    return 0
    
if __name__ == "__main__":
    main()