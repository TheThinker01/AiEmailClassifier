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


class sentence_encoder(Layer):
  def __init__(self,lstm_units,dense_units,document,sentences,units,name='Sentence_Encoder',**kwargs):
    super(sentence_encoder,self).__init__(name=name,**kwargs)
    self.l1 = Lambda(lambda x : tf.reshape(x,([document, sentences, units])))
    self.x1 = Bidirectional(LSTM(lstm_units,activation='selu',kernel_initializer='he_uniform',return_sequences=True))
    self.x2 = TimeDistributed(Dense(dense_units,activation='selu',kernel_initializer='he_uniform'))

  def call(self,inputs):
    x = self.l1(inputs)
    lstm = self.x1(x)
    output = self.x2(lstm)
    return output