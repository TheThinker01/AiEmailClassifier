import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers.experimental.preprocessing import TextVectorization
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Reshape,RepeatVector,LSTM,Dense,Flatten,Bidirectional,Embedding,Input,Layer,GRU,Multiply,Activation,Lambda,Dot,TimeDistributed,Dropout,Embedding
from keras.models import Model
from keras.activations import softmax,selu,sigmoid
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras.constraints import min_max_norm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
import tensorflow as tf
import os
import gensim
from tqdm import tqdm

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    """

    def __init__(self, **kwargs):
        super(AttentionWithContext, self).__init__(**kwargs)

    def __call__(self, x, mask=None):
        self.shapes = int(x.shape[-1])

        self.W = self.add_weight(shape = (self.shapes, self.shapes,),
                                 initializer='he_uniform',name='W')
        self.b = self.add_weight(shape = (self.shapes,),
                                     initializer='zero',name = 'B')

        self.u = self.add_weight(shape = (self.shapes,),
                                 initializer='he_uniform',name = 'U')

        uit = dot_product(x, self.W)

        uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
