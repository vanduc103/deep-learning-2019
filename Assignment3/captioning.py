# This library is used for Assignment3_Part2_ImageCaptioning

# Write your own image captiong code
# You can modify the class structure
# and add additional function needed for image captionging

import tensorflow as tf
import numpy as np


class Captioning():
    
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128, hidden_dim=128):
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']
        self._end = word_to_idx['<END>']
        self.vocab_size = len(word_to_idx)
        self.input_dim = input_dim
        self.wordvec_dim = wordvec_dim
        self.hidden_dim = hidden_dim
        
    def build_model(self, maxlen):
        self.loss = 0
        self.img_features = tf.placeholder(tf.float32, [None, self.input_dim], name="features")
        self.captions = tf.placeholder(tf.int64, [None, maxlen], name="captions")
        batch_size = tf.shape(self.img_features)[0] # get batch size as a tensor
        
        # caption input = captions but the last word (<END>)
        X = self.captions[:, :-1]
        # caption output = captions except the first word (<START>)
        y = self.captions[:, 1:]
        # mask out the NULL word in caption output
        null_words = tf.zeros_like(y)
        caption_mask = tf.math.not_equal(y, null_words)
        
        # compute the initial state from image features => use an NN layer
        h0 = tf.layers.dense(self.img_features, self.hidden_dim) # [N, hidden_dim]
        
        # compute word embedding from words in caption to embedding vector => [N, T, wordvec_dim]
        embed = tf.contrib.layers.embed_sequence(ids=X, vocab_size=self.vocab_size, embed_dim=self.wordvec_dim)
        
        # use RNN to compute output from initial state h0 => outputs [N, T, hidden_dim]
        rnn_cell = tf.contrib.rnn.BasicRNNCell(self.hidden_dim)
        #rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, embed, initial_state=h0, dtype=tf.float32)
        #rnn_outputs, rnn_states = tf.nn.static_rnn(rnn_cell, embed, initial_state=h0, dtype=tf.float32)
        
        # convert rnn output to vocab size
        rnn_outputs = tf.reshape(rnn_outputs, [batch_size * (maxlen-1), self.hidden_dim]) # [N, T, H] => [N*T, H]
        logits = tf.layers.dense(rnn_outputs, self.vocab_size) # [N*T, Vocab_Size]
        self.outputs = tf.reshape(logits, [batch_size, maxlen-1, self.vocab_size], name="outputs") # [N, T, V]
        
        # compute loss, using caption mask
        labels = tf.reshape(y, [batch_size * (maxlen-1)])
        mask_reshape = tf.reshape(caption_mask, [batch_size * (maxlen-1)])
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        self.loss = tf.reduce_mean(tf.boolean_mask(self.loss, mask_reshape))
        
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        
    def predict(self):
        captions = None


