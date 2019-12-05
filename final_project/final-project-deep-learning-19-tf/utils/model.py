import tensorflow as tf
import matplotlib.pyplot as plt
import os, nltk
from miscc.config import cfg
import numpy as np


#################################################
# DO NOT CHANGE 
class GENERATOR:
#################################################
    def __init__(self, input_z, input_rnn, is_training=False, reuse=False):
        '''
        input_z: latent vector (noise)
        input_rnn: text feature (caption feature)
        '''
        self.input_z = input_z
        self.input_rnn = input_rnn
        self.is_training = is_training
        self.reuse = reuse
        self.t_dim = 128
        self.gf_dim = 128
        self.image_size = 256
        self.c_dim = 3
        self._build_model()

    def _build_model(self):
        '''
        self.outputs: final output of generator = synthesis image of size 256
        '''
        s = self.image_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
            
        gf_dim = self.gf_dim
        t_dim = self.t_dim
        c_dim = self.c_dim
            
        with tf.variable_scope("generator", reuse=self.reuse):
            '''
            '''
            
            
            self.outputs = #

#################################################
# DO NOT CHANGE 
class DISCRIMINATOR:
#################################################
    def __init__(self, input_image, is_training=False, reuse=False):
        '''
        input_image: generate or real image
        '''
        self._build_model()

    def _build_model(self):
        '''
        self.outputs: final output of discriminator (real or fake)
        '''
        with tf.variable_scope("discriminator", reuse=self.reuse):
            '''
            '''
            self.outputs = #

#################################################
# DO NOT CHANGE 
class RNN_ENCODER:
#################################################
    '''
        caption --> t_dim (in latent space)
    '''
    def __init__(self, input_seqs, is_training=False, reuse=False):
        '''
        input_seqs: [cfg.BATCH_SIZE X cfg.TEXT.WORDS_NUM] (captions)
        '''
        self._build_model()

    def _build_model(self):
        '''
        self.outputs: final output of text encoder (setence embedding)
        '''
        with tf.variable_scope("rnnencoder", reuse=self.reuse):
            '''
            '''
            self.outputs = #

#################################################
# DO NOT CHANGE 
class CNN_ENCODER:
#################################################
    def __init__(self, inputs, is_training=False, reuse=False):
        '''
        inputs: [cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3] (images)
        '''
        self._build_model()

    def _build_model(self):
        '''
        self.outputs: final output of image encoder (image embedding)
        '''
        with tf.variable_scope('cnnencoder', reuse=self.reuse):
            '''
            '''
            self.outputs = #