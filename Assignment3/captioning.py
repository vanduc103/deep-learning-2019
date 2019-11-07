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
        print(self._null)
        self._end = word_to_idx['<END>']
        self.vocab_size = len(word_to_idx)
        self.input_dim = input_dim
        self.wordvec_dim = wordvec_dim
        self.hidden_dim = hidden_dim
        
    def build_model(self, img_features, captions):
        self.loss = 0
        self.img_features = tf.placeholder(tf.float32, [None, self.input_dim])
        self.captions = tf.placeholder(tf.float32, [None, captions.shape[1]])
        
        # caption input = captions but the last word (<END>)
        X = tf.slice(self.captions, [0, 0], [self.captions.get_shape()[0], self.captions.get_shape()[1]-1])
        # caption output = captions except the first word (<START>)
        y = tf.slice(self.captions, [0, 1], [self.captions.get_shape()[0], self.captions.get_shape()[1]-1])
        # mask out the NULL word in caption output
        null_words = tf.zeros_like(y)
        caption_mask = tf.math.not_equal(y, null_words)
        
        # compute the initial state from image features => use an NN layer
        h0 = tf.layers.dense(self.img_features, self.hidden_dim) # [N, 128]
        
        # compute word embedding from words in caption in to embedding vector => [N, T, 128]
        embed = tf.contrib.layers.embed_sequence(ids=self.X, vocab_size=self.vocab_size, embed_dim=self.wordvec_dim)
        
        # use RNN to compute output from initial state h0 => outputs [N, T, hidden_dim]
        rnn_cell = tf.contrib.rnn.BasicRNNCell(self.hidden_dim)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, embed, initial_state=h0, dtype=tf.float32)
        
        # convert rnn output to vocab size
        rnn_outputs = tf.unstack(rnn_outputs, axis=0) # [N, T, H] => [N*T, H]
        logits = tf.layers.dense(rnn_outputs, self.vocab_size) # [N*T, Vocab_Size]
        outputs = tf.reshape(outputs, [logits.shape[0], self.X.shape[1], self.vocab_size])
        
        # compute loss, using caption mask
        labels = tf.reshape(self.y, [self.y.shape[0] * self.y.shape[1]])
        mask_reshape = tf.reshape(caption_mask, [caption_mask.shape[0] * caption_mask.shape[1]])
        labels = tf.boolean_mask(labels, mask_reshape)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

    def predict(self):
        captions = None
        


