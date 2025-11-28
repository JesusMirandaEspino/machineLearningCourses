# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 13:42:25 2025

@author: jesus
"""

import tensorflow as tf
import numpy as np

t = tf.constant( [ [1.,2.,3.], [4.,5.,6.] ] )
print(t)
print(t.shape)
print(t.dtype)

print(t[:,1:])
print(t[:,2:])

print(t[...,1,tf.newaxis])

print(t + 10)
print(tf.square(t))

print( t @ tf.transpose(t) )
print(tf.constant(42))


a = np.array( [2.,4.,5.] )
print(tf.constant(a))


print(t.numpy())
print(tf.square(a))
print(np.square(t))

# print(tf.constant(2.) + tf.constant(40))

print(tf.constant(2.) + tf.constant(40.))

t2 = tf.constant(40., dtype=tf.float64)

print(tf.constant(2.0) + tf.cast(t2, tf.float32))

v = tf.Variable( [ [ 1.,2.,3. ], [4.,5.,6.] ] )
print(v)


print( v.assign(2 * v) )
print(v[0,1])
print(v[:,2])
print( v[0,1].assign(42) )
print( v[:,2].assign([0., 1.]) )
print(v.scatter_nd_update(indices=[ [0,0],[1,2] ], updates=[100., 200.]))

#v[1] = [7.,8.,9.]

























