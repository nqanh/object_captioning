# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg

class vgg16(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._scope = 'vgg_16'

  def _image_to_head(self, is_training, reuse=None):
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                          trainable=False, scope='conv1')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=False, scope='conv2')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training, scope='conv3')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv4')
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')

    self._act_summaries.append(net)
    self._layers['head'] = net
    
    return net

  def _head_to_tail(self, pool5, is_training, reuse=None):
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      pool5_flat = slim.flatten(pool5, scope='flatten')
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True,
                            scope='dropout6')
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True,
                            scope='dropout7')

    return fc7

# #   # Anh them vo - alwasy return fc7, when both testing and training for rnn
#   def _head_to_rnn(self, pool5, reuse=None):
#       with tf.variable_scope(self._scope, self._scope, reuse=reuse):
#         pool5_flat = slim.flatten(pool5, scope='flatten_rnn')
#         fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6_rnn')
#         fc7_for_rnn = slim.fully_connected(fc6, 4096, scope='fc7_rnn')
#        
#       return fc7_for_rnn

  
#   # Anh them vo - rnn part for caption
#   def _head_to_rnn(self, pool5, is_training, reuse=None):
#     with tf.variable_scope(self._scope, self._scope, reuse=reuse):
#         pool5_flat = slim.flatten(pool5, scope='flatten_rnn')
#       
#         ## NEED TO REMOVE THESE PLACEHOLDER
# #         fc7_features = tf.placeholder('float32',[self.batch_size, self.dim_img_feature], name = 'fc7')
# #         sentence = tf.placeholder('int32', [self.batch_size, self.num_lstm_steps], name = "sentence") # int32, [batch_size, num_of_lstm_step
# #         answer   = tf.placeholder('int32', [self.batch_size, self.num_lstm_steps], name = "answer")
# #         
# #         # mask the loss ## only apply for output mask
# #         answer_mask   = tf.placeholder('float32', [self.batch_size, self.num_lstm_steps])
#         
#         # tempo state
#         # use cell object to init 
#         state1 = self.lstm1.zero_state(self.batch_size, tf.float32)  # --> return a LSTMStateTuple
#         state2 = self.lstm2.zero_state(self.batch_size, tf.float32)
#         padding = tf.zeros([self.batch_size, self.num_lstm_hidden_units])
#         
#         probs = []
#         loss = 0.0
#         
#         with tf.variable_scope(tf.get_variable_scope()) as scope:   #TO FIX: https://github.com/tensorflow/tensorflow/issues/6220
#      
#             for i in range(self.num_lstm_steps): 
#                 if i > 0:   # if i=0 --> create new variables then reuse
#                     tf.get_variable_scope().reuse_variables()
#                 
#                 ## LOOKUP FOR input sentence
#                 word_emb_in = tf.nn.embedding_lookup(self.Wemb, sentence[:,i])  ## LOOK UP FOR Sentence --> input phrases 
#                 with tf.variable_scope("LSTM1"):
#                     output1, state1 = self.lstm1(word_emb_in, state1)  # don't care about the input image ???
#                 
#                 # HANDLE img_featue
#                 image_embedding = tf.nn.xw_plus_b(fc7_features, self.Wimg, self.bimg, name='img_embedding') # output_size = (batch_size, dim_hidden)
#                 image_embedding = tf.nn.tanh(image_embedding) ## make it non-linear
#                 
#                 # combine img feature with output1 (sentence featue)
#                 sen_img_embedding = tf.concat([output1, image_embedding], 1) ## concat axis = 1; 0 axis keeps batch_size; output will be (batch_size, dim_hiden * 2)
#                 #sen_img_embedding = tf.concat(([3, 3, 3], [4, 4, 4]), 1) ## concat axis = 1; 0 axis keeps batch_size; output will be (batch_size, dim_hiden * 2)
#                 
#                 # learn sen_img_embeddign again to make it become (batch_size, dim)    
#                 output_with_sen_and_img = tf.nn.xw_plus_b(sen_img_embedding, self.Wcombine_img_sen, self.bcombine_img_sen) # output shape = (batch_size, dim_hidden)
#                 output_with_sen_and_img = tf.nn.tanh(output_with_sen_and_img)
#                 
#                             
#                 ## LOOKUP FOR output sentence  
#                 if i == 0:
#                     word_embed_out = tf.zeros([self.batch_size, self.num_lstm_hidden_units])  ## i=0: no previous word --> start with zero
#                 else:                                                                                                           
#                     with tf.device("/cpu:0"):
#                         word_embed_out = tf.nn.embedding_lookup(self.Wemb, answer[:, i-1]) # Function: Find ROWS of Wemb for a list of WORD in caption[:,i-1]  
#                                                                                            # caption[:, i-1] returns all values (WORDS) at COLUMN i-1
#                                                                                            # caption shape (batch_size, pink_box)
#                  
#                 with tf.variable_scope("LSTM2"):
#                     #output2, state2 = self.lstm2( tf.concat([word_embed_out, output1], 1), state2 )  ### feed captions
#                     output2, state2 = self.lstm2(tf.concat([word_embed_out, output_with_sen_and_img], 1), state2)  ### USE output_with_sen_and_img, NOT output1
#      
#                 labels = tf.expand_dims(answer[:, i], 1)    # get currrent caption???     
#         
#                 indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)   # tf.range() creates a sequence of number  # indice is the index
#         
#                 concated = tf.concat([indices, labels], 1)
#                  
#                 #onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)  ### use tf.one_hot() ???
#                 onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.vocab_size]), 1.0, 0.0)  # tf.pack --> tf.stack in TF1
#                 
#                 logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)   ### lost for words
#                    
#                 #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)   ### logit and onehot must have the same shape (batch_size, num_class)
#                 cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words)
#                  
#                 cross_entropy = cross_entropy * answer_mask[:, i]   # caption_mask - only count the lost of real data (image/word), ignore the padding
#                  
#                 probs.append(logit_words)
#      
#                 current_loss = tf.reduce_sum(cross_entropy)  # sum all cross_entropy loss of all batch (100)
#                 loss += current_loss
#             
#         loss = loss/tf.reduce_sum(answer_mask) # average loss over all words
#         
#           
#     return loss

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      if v.name == (self._scope + '/fc6/weights:0') or \
         v.name == (self._scope + '/fc7/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv,
                                      self._scope + "/fc7/weights": fc7_conv,
                                      self._scope + "/conv1/conv1_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv,
                            self._variables_to_fix[self._scope + '/fc6/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv,
                            self._variables_to_fix[self._scope + '/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'],
                            tf.reverse(conv1_rgb, [2])))
