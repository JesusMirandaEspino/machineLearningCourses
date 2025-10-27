# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 13:01:22 2025

@author: jesus
"""

import tensorflow as tf

tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(tf.constant(15))

t = tf.constant([[1., 2.],[3., 4.]])
print(t.shape)
print(t.dtype)


t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t.shape)


print(t[0,:])
print(t[:, 1])


t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t.shape)


print(t)
print(t + 10)


print(t + t)
print(tf.square(t))

t1 = tf.constant(1.0)
t2 = tf.constant(2)

print("Tipo t1:", t1.dtype)
print("Tipo t2:", t2.dtype)


try:
    t3 = t1 + t2
except Exception as e:
    print("Exception:", e)


try:
    tf.constant([1, 2, 3]) + tf.constant([1.0, 2.0, 3.0])
except Exception as e:
    print("Exception:", e)



t = tf.constant([1.0, 2.0, 3.0])
t[0] = 4.0



t = tf.Variable([1.0, 2.0, 3.0])
print("Tensor original:", t.value())

t[0].assign(2.0)
print("Tensor modificado:", t.value()) 



tf.constant(b"hola mundo")


tf.constant("españa")


tf.constant(["hola", "mundo", "españa"])
























































