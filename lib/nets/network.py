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
import math

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer, proposal_layer_test_caption, proposal_layer_test_caption_compact
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from model.config import cfg

class Network(object):
  def __init__(self):
      self._predictions = {}
      self._losses = {}
      self._anchor_targets = {}
      self._proposal_targets = {}
      self._layers = {}
      self._gt_image = None
      self._act_summaries = []
      self._score_summaries = {}
      self._train_summaries = []
      self._event_summaries = {}
      self._variables_to_fix = {}
         
      # Anh them vo
      ## LSTM stuff
      self.vocab_size = cfg.vocab_size
      self.num_lstm_time_steps = cfg.num_lstm_time_steps
      self.num_lstm_hidden_units = cfg.num_lstm_hidden_units
      self.img_feature_dim = cfg.img_feature_dim
      self.bias_init_vector = cfg.bias_init_vector

      # Tempo state - keep words to feed to LSTM
      #self.Wemb = tf.Variable(tf.random_uniform([self.vocab_size + 1, self.num_lstm_hidden_units], -0.1, 0.1), name='Wemb_in')   ## should + 1 here???
      self.Wemb = tf.Variable(tf.random_uniform([self.vocab_size, self.num_lstm_hidden_units], -0.1, 0.1), name='Wemb_in') ## not + 1, already did in self.vocab_size
     
      # define lstm
      self.lstm1 = tf.contrib.rnn.LSTMCell(self.num_lstm_hidden_units) # handle input sentence

      # for the loss
      self.embed_word_W = tf.Variable(tf.random_uniform([self.num_lstm_hidden_units, self.vocab_size], -0.1,0.1), name='embed_word_W')
      if self.bias_init_vector is not None:
          self.embed_word_b = tf.Variable(self.bias_init_vector.astype(np.float32), name='embed_word_b')
      else:
          self.embed_word_b = tf.Variable(tf.zeros(self.vocab_size), name='embed_word_b')
          
    
  def _add_gt_image(self):
    # add back mean
    image = self._image + cfg.PIXEL_MEANS 
    # BGR to RGB (opencv uses BGR)
    resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
    self._gt_image = tf.reverse(resized, axis=[-1])

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    if self._gt_image is None:
      self._add_gt_image()
    image = tf.py_func(draw_bounding_boxes, 
                      [self._gt_image, self._gt_boxes, self._im_info],
                      tf.float32, name="gt_boxes")
    
    return tf.summary.image('GROUND_TRUTH', image)

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_top_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                     self._feat_stride, self._anchors, self._num_anchors],
                                    [tf.float32, tf.float32], name="proposal_top")
      rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
      rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors],
                                    [tf.float32, tf.float32], name="proposal")
      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores

  # Anh them vo
  ## call real function to get real rois, rpn_scores and input sentences (based on rois shape)  ->> return 300 rois, pad if missing
  def _proposal_layer_test_caption(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores, sentences = tf.py_func(proposal_layer_test_caption,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors, self._input_test_phrase],
                                    [tf.float32, tf.float32, tf.int32],
                                    name="proposal_layer_test_caption")
      
      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])
      sentences.set_shape([None, cfg.MAX_PHRASE_LENGTH])

    return rois, rpn_scores, sentences

  # Anh them vo --> only return real rois
  
  def _proposal_layer_test_caption_compact(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_layer_test_caption_compact,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors,],
                                    [tf.float32, tf.float32, tf.int32],
                                    name="proposal_layer_test_caption_compact")
      
      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores

  # Only use it if you have roi_pooling op written in tf.image
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.POOLING_SIZE,
                                  pooled_width=cfg.POOLING_SIZE,
                                  spatial_scale=1. / 16.)[0]

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bounding boxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be back-propagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      pre_pool_size = cfg.POOLING_SIZE * 2
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

    return slim.max_pool2d(crops, [2, 2], padding='SAME')

  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  def _anchor_target_layer(self, rpn_cls_score, name):
    with tf.variable_scope(name) as scope:
      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
        anchor_target_layer,
        [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
        [tf.float32, tf.float32, tf.float32, tf.float32],
        name="anchor_target")

      rpn_labels.set_shape([1, 1, None, None])
      rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets['rpn_labels'] = rpn_labels
      self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
      self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
      self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

      self._score_summaries.update(self._anchor_targets)

    return rpn_labels

  # Anh them vo
  def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
          rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois_phrase_obj_indexes, sentences, answers, answers_masks, positive_rois = tf.py_func(
            proposal_target_layer,
            [rois, roi_scores, self._gt_boxes, self._num_classes, self._all_phrases, self._image], # Anh them all_phrases
            [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64, tf.int32, tf.int32, tf.float32, tf.float32],
            name="proposal_target")
          
          
          if cfg.DEBUG_VERBOSE == 1:
              print('network.py ************** rois shape                {:s}'.format(rois.shape))              ## unknown -- not has shape yet     
              print('network.py ************** roi_scores shape          {:s}'.format(roi_scores.shape))
              print('network.py ************** labels shape              {:s}'.format(labels.shape))             
              print('network.py ************** bbox_targets shape        {:s}'.format(bbox_targets.shape))           
              print('network.py ************** bbox_inside_weights shape {:s}'.format(bbox_inside_weights.shape))
              
              print('network.py ************** sentences shape           {:s}'.format(sentences.shape))
              print('network.py ************** answers   shape           {:s}'.format(answers.shape))
              
          rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
          roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
          labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
          bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
          bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
          bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
          
#           ### need to change to dynamic shape      
#           sentences.set_shape([cfg.TRAIN.BATCH_SIZE, cfg.MAX_PHRASE_LENGTH]) # check lai cfg.TRAIN.BATCH_SIZE vs. cfg.TRAIN.RPN_BATCHSIZE
#           answers.set_shape([cfg.TRAIN.BATCH_SIZE, cfg.MAX_PHRASE_LENGTH]) 
        
          ### dynamic shape      
          num_positive_rois = positive_rois.shape[0] 
          sentences.set_shape([num_positive_rois, cfg.MAX_PHRASE_LENGTH]) # check lai cfg.TRAIN.BATCH_SIZE vs. cfg.TRAIN.RPN_BATCHSIZE
          answers.set_shape([num_positive_rois, cfg.MAX_PHRASE_LENGTH]) 

          if cfg.DEBUG_VERBOSE == 1:
              print('network.py ############## rois shape                {:s}'.format(rois.shape))              ## (256, 5) ## WHY cfg.TRAIN.BATCH_SIZE = 256?
              print('network.py ############## roi_scores shape          {:s}'.format(roi_scores.shape))
              print('network.py ############## labels shape              {:s}'.format(labels.shape))             
              print('network.py ############## bbox_targets shape        {:s}'.format(bbox_targets.shape))           
              print('network.py ############## bbox_inside_weights shape {:s}'.format(bbox_inside_weights.shape))
              
              print('network.py ############## sentences shape           {:s}'.format(sentences.shape))
              print('network.py ############## answers   shape           {:s}'.format(answers.shape))           # (256, 10)
          
          
          self._proposal_targets['rois'] = rois
          self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
          self._proposal_targets['bbox_targets'] = bbox_targets
          self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
          self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
                
          self._proposal_targets['sentences'] = sentences # Anh them vo
          self._proposal_targets['answers'] = answers
          
          self._score_summaries.update(self._proposal_targets)
          
          return rois, roi_scores, sentences, answers, answers_masks, positive_rois  # Anh add sentences, answers
  
  
#   def _get_test_input_sentence(self, name):
#       with tf.variable_scope(name) as scope:
          
  #   # Anh them vo - alwasy return fc7, when both testing and training for rnn
  def _head_to_rnn(self, pool5, reuse=None):
      with tf.variable_scope(self._scope, self._scope, reuse=reuse):
        pool5_flat = slim.flatten(pool5, scope='flatten_rnn')
        fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6_rnn')
        fc7 = slim.fully_connected(fc6, 2048, scope='fc7_rnn')
        fc7_for_rnn = slim.fully_connected(fc7, cfg.num_lstm_hidden_units, scope='fc7_for_rnn')  ## compress to #num_lstm_hidden_unit to use directly in LSTM
       
      return fc7_for_rnn

  
  # Anh them vo - use when training
  def _caption_trainer(self, fc7_features, sentences, sentence_masks):  
    
    with tf.variable_scope(self._scope, self._scope, reuse=None):
        ## CREATE DYNAMIC SHAPE base onf fc7_featues
        self.train_lstm_batch_size = tf.shape(fc7_features)[0]
        if cfg.DEBUG_VERBOSE == 1:
            print('network.py ---------- CAPTION TRAINER ---------- fc7_features shape                 {:s}'.format(fc7_features.shape))         # (256, 4096)
              
        # tempo zero state
        zero_state = self.lstm1.zero_state(self.train_lstm_batch_size, tf.float32)  # --> return a LSTMStateTuple
        loss = 0.0
        
        for i in range(self.num_lstm_time_steps): 
            if i == 0:
                word_embed_out = tf.zeros([self.train_lstm_batch_size, self.num_lstm_hidden_units])  ## i=0: no previous word --> start with zero
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(tf.concat([word_embed_out, fc7_features], 1), zero_state) ## 1st state --> feed image feature
                    #print 'state1 shape  : ', state1.get_shape()
            else:                            
                tf.get_variable_scope().reuse_variables()
                                                                                                   
                word_embed_out = tf.nn.embedding_lookup(self.Wemb, sentences[:, i-1]) # Function: Find ROWS of Wemb for a list of WORD in caption[:,i-1]  
                                                                                      # caption[:, i-1] returns all values (WORDS) at COLUMN i-1
                                                                                      # caption shape (batch_size, pink_box)
                
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(tf.concat([word_embed_out, output1], 1), state1)  ### USE output_with_sen_and_img, NOT output1
                                                                                       
 
            labels = tf.expand_dims(sentences[:,i], 1)    # get currrent caption
            indices = tf.expand_dims(tf.range(0, self.train_lstm_batch_size, 1), 1)   # tf.range() creates a sequence of number  # indice is the index
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.train_lstm_batch_size, self.vocab_size]), 1.0, 0.0)  # tf.pack --> tf.stack in TF1
            
            logit_words = tf.nn.xw_plus_b(output1, self.embed_word_W, self.embed_word_b)   ### lost for words
               
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words)
            cross_entropy = cross_entropy * sentence_masks[:, i]   # caption_mask - only count the lost of real data (image/word), ignore the padding
             
            current_loss = tf.reduce_sum(cross_entropy)  # sum all cross_entropy loss of all batch (100)
            loss += current_loss
            
            
        loss = loss/tf.reduce_sum(sentence_masks) # average loss over all words
        
    return loss
  
  
  ## Anh them vo - use when testing
  def _caption_generator(self, fc7_features):
      
    with tf.variable_scope(self._scope, self._scope, reuse=None):
        
        self.test_lstm_batch_size = tf.shape(fc7_features)[0]
          
        zero_state = self.lstm1.zero_state(self.test_lstm_batch_size, tf.float32)  # --> return a LSTMStateTuple
        
        generated_words = []  ## keep generated words as propability --> AS A LIST
        
        for i in range(self.num_lstm_time_steps):
            if i == 0:
                word_embed_out = tf.zeros([self.test_lstm_batch_size, self.num_lstm_hidden_units])  ## i=0: no previous word --> start with zero
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(tf.concat([word_embed_out, fc7_features], 1), zero_state) ## 1st state --> feed image feature
            else:                            
                tf.get_variable_scope().reuse_variables()
    
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(tf.concat([word_embed_out, output1], 1), state1) ## word_embed_out from last time step
    
            
            # find logit word
            logit_words = tf.nn.xw_plus_b(output1, self.embed_word_W, self.embed_word_b)
            #max_prob_index = tf.argmax(logit_words, 1)[0]
            max_prob_index = tf.argmax(logit_words, 1) ## has many input --> advoid using [0] --> get only 1 value
            
            generated_words.append(max_prob_index)
            
            current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)  ## find current embed for next word
    
    return generated_words    
  
  
  def _anchor_component(self):
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      # just to get the shape right
      height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
      width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
      anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                          [height, width,
                                           self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                          [tf.float32, tf.int32], name="generate_anchors")
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

    ## original code
#   def _build_network(self, is_training=True):
#         # select initializers
#         if cfg.TRAIN.TRUNCATED:
#           initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
#           initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
#         else:
#           initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
#           initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
#           
#         net_conv = self._image_to_head(is_training)
#         with tf.variable_scope(self._scope, self._scope):
#           # build the anchors for the image
#           self._anchor_component()
#           # region proposal network
#           rois = self._region_proposal(net_conv, is_training, initializer)
#           # region of interest pooling
#           if cfg.POOLING_MODE == 'crop':
#             pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
#           else:
#             raise NotImplementedError
#           
#         fc7 = self._head_to_tail(pool5, is_training)
#         with tf.variable_scope(self._scope, self._scope):
#           # region classification
#           cls_prob, bbox_pred = self._region_classification(fc7, is_training, 
#                                                             initializer, initializer_bbox)
#           
#         self._score_summaries.update(self._predictions)
#           
#         return rois, cls_prob, bbox_pred

  
#   # Anh them vo
  def _build_network(self, is_training=True):
      # select initializers
      if cfg.TRAIN.TRUNCATED:
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
      else:
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
      
      net_conv = self._image_to_head(is_training)
      with tf.variable_scope(self._scope, self._scope):
        # build the anchors for the image
        self._anchor_component()
        # region proposal network
        #rois = self._region_proposal(net_conv, is_training, initializer)
        if is_training:
            rois, sentences, answers, answers_masks, positive_rois = self._region_proposal(net_conv, is_training, initializer)  ## tra ra nhieu rois
            positive_pool5 = self._crop_pool_layer(net_conv, positive_rois, "positive_pool5") ## only pool positive_rois 
            if cfg.DEBUG_VERBOSE == 1:
                print('network.py @@@@@@@@@@@@@@@@@ positive_rois shape               {:s}'.format(positive_rois.shape))         
                print('network.py @@@@@@@@@@@@@@@@@ positive_pool5 shape              {:s}'.format(positive_pool5.shape))  
        else:
            rois = self._region_proposal(net_conv, is_training, initializer)  ## tra ra nhieu rois, + query sentence
         
        # region of interest pooling
        if cfg.POOLING_MODE == 'crop':
            pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
        else:
            raise NotImplementedError
      

      ## original code
      fc7 = self._head_to_tail(pool5, is_training) ## check scope
      
      if is_training:
          fc7_for_rnn = self._head_to_rnn(positive_pool5) ## get fc7 feature for rnn
      else:
          fc7_for_rnn = self._head_to_rnn(pool5) ## get fc7 feature for rnn - from all rois
        
      
      with tf.variable_scope(self._scope, self._scope):
        
        #fc7 = self._head_to_tail(pool5, is_training) ## when is_training = True --> use dropout --> when testing no dropout
        
        
        # region classification
        cls_prob, bbox_pred = self._region_classification(fc7, is_training, 
                                                          initializer, initializer_bbox)    ## here will feed all batch_size fc7 

        ## debug info
        if cfg.DEBUG_VERBOSE == 1:
            print('network.py ++++++++++++++ fc7 shape                 {:s}'.format(fc7.shape))         # (256, 4096)
            print('network.py ++++++++++++++ pool5 shape               {:s}'.format(pool5.shape))       # (256, 7, 7, 512)
        
        if is_training:
            caption_loss = self._caption_trainer(fc7_for_rnn, answers, answers_masks) # sentences is (positive_rois, num_lstm_steps): each POSITIVE roi has 1 sentence --> map to 1 answer
        else:
            if cfg.DEBUG_VERBOSE == 1:  
                print('network.py ~~~~~~~~~~~~~ TESTING ... fc7 shape                 {:s}'.format(fc7.shape))         # (?, 4096)
                print('network.py ~~~~~~~~~~~~~ TESTING ... sentences shape           {:s}'.format(sentences.shape))   
                print('network.py ~~~~~~~~~~~~~ TESTING ... sentences                 {:s}'.format(sentences))         
                
                            
            generated_sentences = self._caption_generator(fc7_for_rnn)   ## testing --> get output words from fc7 feature + input sentences
            
            self._predictions["generated_sentences"] = generated_sentences                                                                                
                                                                                                        
      self._score_summaries.update(self._predictions)
      
      
      if is_training:
          #return rois, cls_prob, bbox_pred, caption_loss
          return rois, cls_prob, bbox_pred, caption_loss, positive_rois
      else:
          return rois, cls_prob, bbox_pred, generated_sentences


  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box

#   ## original code
#   def _add_losses(self, sigma_rpn=3.0):
#         with tf.variable_scope('LOSS_' + self._tag) as scope:
#           # RPN, class loss
#           rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
#           rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
#           rpn_select = tf.where(tf.not_equal(rpn_label, -1))
#           rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
#           rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
#           rpn_cross_entropy = tf.reduce_mean(
#             tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
#           
#           # RPN, bbox loss
#           rpn_bbox_pred = self._predictions['rpn_bbox_pred']
#           rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
#           rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
#           rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
#           rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
#                                               rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
#           
#           # RCNN, class loss
#           cls_score = self._predictions["cls_score"]
#           label = tf.reshape(self._proposal_targets["labels"], [-1])
#           cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
#           
#           # RCNN, bbox loss
#           bbox_pred = self._predictions['bbox_pred']
#           bbox_targets = self._proposal_targets['bbox_targets']
#           bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
#           bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
#           loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
#           
#           self._losses['cross_entropy'] = cross_entropy
#           self._losses['loss_box'] = loss_box
#           self._losses['rpn_cross_entropy'] = rpn_cross_entropy
#           self._losses['rpn_loss_box'] = rpn_loss_box
#           
#           loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
#           regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
#           self._losses['total_loss'] = loss + regularization_loss
#           
#           self._event_summaries.update(self._losses)
#           
#         return loss


  ## Anh them vo
  def _add_losses(self, caption_loss, sigma_rpn=3.0):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
          # RPN, class loss
          rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
          rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
          rpn_select = tf.where(tf.not_equal(rpn_label, -1))
          rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
          rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
          rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
          
          # RPN, bbox loss
          rpn_bbox_pred = self._predictions['rpn_bbox_pred']
          rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
          rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
          rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
          rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                              rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
          
          # RCNN, class loss
          cls_score = self._predictions["cls_score"]
          label = tf.reshape(self._proposal_targets["labels"], [-1])
          cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
          
          # RCNN, bbox loss
          bbox_pred = self._predictions['bbox_pred']
          bbox_targets = self._proposal_targets['bbox_targets']
          bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
          bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
          loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
          
          self._losses['cross_entropy'] = cross_entropy
          self._losses['loss_box'] = loss_box
          self._losses['rpn_cross_entropy'] = rpn_cross_entropy
          self._losses['rpn_loss_box'] = rpn_loss_box
             
          self._losses['caption_loss'] = caption_loss
          
          # Anh them vo
          loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + caption_loss
          
          if cfg.TRAIN.USE_REGULARIZATION == False:    
              self._losses['total_loss'] = loss
          else:
              regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
              self._losses['total_loss'] = loss + regularization_loss
          
          
          self._event_summaries.update(self._losses)
          
        return loss


## original code
#   def _region_proposal(self, net_conv, is_training, initializer):
#       rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
#                           scope="rpn_conv/3x3")
#       self._act_summaries.append(rpn)
#       rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
#                                   weights_initializer=initializer,
#                                   padding='VALID', activation_fn=None, scope='rpn_cls_score')
#       # change it so that the score has 2 as its channel size
#       rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
#       rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
#       rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
#       rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
#       rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
#                                   weights_initializer=initializer,
#                                   padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
#       if is_training:
#         rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois") ## NO sentence or answers here
#         rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
#         # Try to have a deterministic order for the computing graph, for reproducibility
#         with tf.control_dependencies([rpn_labels]):
#           rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois") ## return "sentences" and "answers" for lstm
#       else:
#         if cfg.TEST.MODE == 'nms':
#           rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
#         elif cfg.TEST.MODE == 'top':
#           rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
#         else:
#           raise NotImplementedError
#      
#       self._predictions["rpn_cls_score"] = rpn_cls_score
#       self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
#       self._predictions["rpn_cls_prob"] = rpn_cls_prob
#       self._predictions["rpn_cls_pred"] = rpn_cls_pred
#       self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
#       self._predictions["rois"] = rois
#          
#       return rois


      # Anh them vo
  def _region_proposal(self, net_conv, is_training, initializer):
        rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                            scope="rpn_conv/3x3")
        self._act_summaries.append(rpn)
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_cls_score')
        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        if is_training:
          rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois") ## NO sentence or answers here
          rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
          # Try to have a deterministic order for the computing graph, for reproducibility
          with tf.control_dependencies([rpn_labels]):
            rois, _, sentences, answers, answers_masks, positive_rois = self._proposal_target_layer(rois, roi_scores, "rpn_rois") ## return "sentences" and "answers" for lstm
        else:
          
          if cfg.TEST.MODE == 'nms':
            #rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois") ## only has rois here
            # Anh them vo - use _proposal_test_caption to get both rois in input sentences with the same shape
            #rois, _, sentences = self._proposal_layer_test_caption(rpn_cls_prob, rpn_bbox_pred, "rois") ## only has rois here
            #rois, _= self._proposal_layer_test_caption_compact(rpn_cls_prob, rpn_bbox_pred, "rois") ## only has rois here
            rois, _= self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois") ## only has rois here
            if cfg.DEBUG_VERBOSE == 1:   
                print('network.py ~~~~~~~~~~~~~ TESTING ~~~~~~~~~~~~~ rois shape                 {:s}'.format(rois.shape))         # (?, 4096)
                print('network.py ~~~~~~~~~~~~~ TESTING ~~~~~~~~~~~~~ sentences shape            {:s}'.format(sentences.shape))         # (?, 4096)
                
            ## tempo disable
#           elif cfg.TEST.MODE == 'top':  
#             rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
          else:
            raise NotImplementedError
          
        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["rois"] = rois
              
#         self._predictions["sentences"] = sentences  ## WRONG HERE
#         self._predictions["answers"] = answers
        
        if is_training:  
            return rois, sentences, answers, answers_masks, positive_rois
        else:
            return rois ## return rois
            



  def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
    cls_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
    cls_prob = self._softmax_layer(cls_score, "cls_prob")
    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
    bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                     weights_initializer=initializer_bbox,
                                     trainable=is_training,
                                     activation_fn=None, scope='bbox_pred')

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred

    return cls_prob, bbox_pred

  def _image_to_head(self, is_training, reuse=None):
    raise NotImplementedError

  def _head_to_tail(self, pool5, is_training, reuse=None):
    raise NotImplementedError

  
#   ## original code
#   def create_architecture(self, mode, num_classes, tag=None,
#                               anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
#         self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
#         self._im_info = tf.placeholder(tf.float32, shape=[3])
#         self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
#         self._tag = tag
#           
#         self._num_classes = num_classes
#         self._mode = mode
#         self._anchor_scales = anchor_scales
#         self._num_scales = len(anchor_scales)
#           
#         self._anchor_ratios = anchor_ratios
#         self._num_ratios = len(anchor_ratios)
#           
#         self._num_anchors = self._num_scales * self._num_ratios
#           
#         # Anh them vo
#         self._all_phrases = tf.placeholder(tf.float32, shape=[None, cfg.MAX_PHRASE_LENGTH + 1])
#               
#         training = mode == 'TRAIN'
#         testing = mode == 'TEST'
#           
#         assert tag != None
#           
#         # handle most of the regularizers here
#         weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
#         if cfg.TRAIN.BIAS_DECAY:
#           biases_regularizer = weights_regularizer
#         else:
#           biases_regularizer = tf.no_regularizer
#           
#         # list as many types of layers as possible, even if they are not used now
#         with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
#                         slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
#                         weights_regularizer=weights_regularizer,
#                         biases_regularizer=biases_regularizer, 
#                         biases_initializer=tf.constant_initializer(0.0)): 
#           rois, cls_prob, bbox_pred = self._build_network(training)
#           
#         layers_to_output = {'rois': rois}
#           
#         for var in tf.trainable_variables():
#           self._train_summaries.append(var)
#           
#         if testing:
#           stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
#           means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
#           self._predictions["bbox_pred"] *= stds
#           self._predictions["bbox_pred"] += means
#         else:
#           self._add_losses()
#           layers_to_output.update(self._losses)
#           
#           val_summaries = []
#           with tf.device("/cpu:0"):
#             val_summaries.append(self._add_gt_image_summary())
#             for key, var in self._event_summaries.items():
#               val_summaries.append(tf.summary.scalar(key, var))
#             for key, var in self._score_summaries.items():
#               self._add_score_summary(key, var)
#             for var in self._act_summaries:
#               self._add_act_summary(var)
#             for var in self._train_summaries:
#               self._add_train_summary(var)
#           
#           self._summary_op = tf.summary.merge_all()
#           self._summary_op_val = tf.summary.merge(val_summaries)
#           
#         layers_to_output.update(self._predictions)
#           
#         return layers_to_output


      # Anh them vo
  def create_architecture(self, mode, num_classes, tag=None,
                              anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._tag = tag
          
        self._num_classes = num_classes
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)
          
        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)
          
        self._num_anchors = self._num_scales * self._num_ratios
          
        # Anh them vo  ## use when training
        self._all_phrases = tf.placeholder(tf.float32, shape=[None, cfg.MAX_PHRASE_LENGTH + 1])
        
        # use when testing
        #self._input_test_phrase = tf.placeholder(tf.int32, shape=[None, cfg.MAX_PHRASE_LENGTH])   ## test + 1 later ---> assume know bbox index
              
        training = mode == 'TRAIN'
        testing = mode == 'TEST'
          
        assert tag != None
          
        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
          biases_regularizer = weights_regularizer
        else:
          biases_regularizer = tf.no_regularizer
          
        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                        weights_regularizer=weights_regularizer,
                        biases_regularizer=biases_regularizer, 
                        biases_initializer=tf.constant_initializer(0.0)): 
          
          ## Anh them vo
          if training:  
              rois, cls_prob, bbox_pred, caption_loss, positive_rois = self._build_network(training)
          else:
              rois, cls_prob, bbox_pred, generated_sentences = self._build_network(training)       ## build net with testing flag
          
        layers_to_output = {'rois': rois}
          
        for var in tf.trainable_variables():
          self._train_summaries.append(var)
          
        if testing:
          stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
          means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
          self._predictions["bbox_pred"] *= stds
          self._predictions["bbox_pred"] += means
        else:
          self._add_losses(caption_loss)
          layers_to_output.update(self._losses)
          
          val_summaries = []
          with tf.device("/cpu:0"):
            val_summaries.append(self._add_gt_image_summary())
            for key, var in self._event_summaries.items():
              val_summaries.append(tf.summary.scalar(key, var))
            for key, var in self._score_summaries.items():
              self._add_score_summary(key, var)
            for var in self._act_summaries:
              self._add_act_summary(var)
            for var in self._train_summaries:
              self._add_train_summary(var)
          
          self._summary_op = tf.summary.merge_all()
          self._summary_op_val = tf.summary.merge(val_summaries)
          
        layers_to_output.update(self._predictions)
          
        
          
        print('network.py >>>>>>>>>>>>>>>>>>>>>>> CREATE ARCHITECTURE DONE! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        
        # Anh them vo  
        if training:  
            #return layers_to_output
            
            return layers_to_output
        else:
            return layers_to_output, generated_sentences



  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError

  def fix_variables(self, sess, pretrained_model):
    raise NotImplementedError

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, sess, image):
    feed_dict = {self._image: image}
    feat = sess.run(self._layers["head"], feed_dict=feed_dict)
    return feat

  # original test code
  # only useful during testing mode
  def test_image(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info}

    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois

  # test code for caption
  # Anh them vo
  def test_image_caption(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info}

    ### original 1 feed_dict     
    cls_score, cls_prob, bbox_pred, generated_sentences, rois = sess.run([self._predictions["cls_score"],
                                                                          self._predictions['cls_prob'],
                                                                          self._predictions['bbox_pred'],
                                                                          self._predictions['generated_sentences'],
                                                                          self._predictions['rois']],
                                                                          feed_dict=feed_dict)
    
    generated_sentences = np.column_stack(generated_sentences) ## reshape the answer results (stack by column
    
    return cls_score, cls_prob, bbox_pred, generated_sentences, rois   ## rois here is not fc7 features




## Anh them vo - version 1 - train with 1 sess --> need fix batch_size for lstm 
  def get_summary(self, sess, blobs):
     feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                  self._gt_boxes: blobs['gt_boxes'],
                  self._all_phrases: blobs['all_phrases']}
      
     summary = sess.run(self._summary_op_val, feed_dict=feed_dict)
  
     return summary
     
  def train_step(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes'],
                 self._all_phrases: blobs['all_phrases']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, caption_loss, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                        self._losses['rpn_loss_box'],
                                                                        self._losses['cross_entropy'],
                                                                        self._losses['loss_box'],
                                                                        self._losses['caption_loss'],
                                                                        self._losses['total_loss'],
                                                                        train_op],
                                                                       feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, caption_loss, loss
 
  def train_step_with_summary(self, sess, blobs, train_op):  # blobs come from minibatch.py
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes'],
                 self._all_phrases: blobs['all_phrases']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, caption_loss, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                 self._losses['rpn_loss_box'],
                                                                                 self._losses['cross_entropy'],
                                                                                 self._losses['loss_box'],
                                                                                 self._losses['caption_loss'],
                                                                                 self._losses['total_loss'],
                                                                                 self._summary_op,
                                                                                 train_op],
                                                                                feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, caption_loss, loss, summary
 
 
  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes'],
                 self._all_phrases: blobs['all_phrases']}
    sess.run([train_op], feed_dict=feed_dict)
    
    
    
# ###################################################################################################    
# ## run with 2 sess.run
# #   def get_summary(self, sess, blobs):
# #      feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
# #                   self._gt_boxes: blobs['gt_boxes'],
# #                   self._all_phrases: blobs['all_phrases']}
# #      
# #      summary = sess.run(self._summary_op_val, feed_dict=feed_dict)
# #  
# #      return summary
#     
#   def train_step(self, sess, blobs, train_op):
#     feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
#                  self._gt_boxes: blobs['gt_boxes'],
#                  self._all_phrases: blobs['all_phrases']}
#     
#     ## 1st round --> return also positive_rois --> get shape from this one
# #     rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, positive_rois, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
# #                                                                         self._losses['rpn_loss_box'],
# #                                                                         self._losses['cross_entropy'],
# #                                                                         self._losses['loss_box'],
# #                                                                         self._positive_rois['positive_rois'],
# #                                                                         self._losses['total_loss'],
# #                                                                         train_op],
# #                                                                        feed_dict=feed_dict)
#     
#     
#     # get positive roi only
#     positive_rois = sess.run([self._positive_rois['positive_rois']],
#                              feed_dict=feed_dict)
#     
#     
#     feed_dict_2 = {self._image: blobs['data'], self._im_info: blobs['im_info'],
#                    self._gt_boxes: blobs['gt_boxes'],
#                    self._all_phrases: blobs['all_phrases'],
#                    self.batch_size: np.asarray(positive_rois).shape[0]}
#     ## 2nd round
#     rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, caption_loss, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
#                                                                                       self._losses['rpn_loss_box'],
#                                                                                       self._losses['cross_entropy'],
#                                                                                       self._losses['loss_box'],
#                                                                                       self._losses['caption_loss'],
#                                                                                       self._losses['total_loss'],
#                                                                                       train_op],
#                                                                                       feed_dict=feed_dict)
#      
#     
#     
#     
#     return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, caption_loss, loss
# 
#   def train_step_with_summary(self, sess, blobs, train_op):  # blobs come from minibatch.py
#     feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
#                  self._gt_boxes: blobs['gt_boxes'],
#                  self._all_phrases: blobs['all_phrases']}
#     
# #     rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, caption_loss, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
# #                                                                                  self._losses['rpn_loss_box'],
# #                                                                                  self._losses['cross_entropy'],
# #                                                                                  self._losses['loss_box'],
# #                                                                                  self._losses['caption_loss'],
# #                                                                                  self._losses['total_loss'],
# #                                                                                  self._summary_op,
# #                                                                                  train_op],
# #                                                                                 feed_dict=feed_dict)
#     
#     
#     # get positive roi only
#     positive_rois = sess.run([self._positive_rois['positive_rois']],
#                              feed_dict=feed_dict)
#     
#     
#     feed_dict_2 = {self._image: blobs['data'], self._im_info: blobs['im_info'],
#                    self._gt_boxes: blobs['gt_boxes'],
#                    self._all_phrases: blobs['all_phrases'],
#                    self.batch_size: np.asarray(positive_rois).shape[0]}
#     ## 2nd round
#     rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, caption_loss, loss, summary,  _ = sess.run([self._losses["rpn_cross_entropy"],
#                                                                                                   self._losses['rpn_loss_box'],
#                                                                                                   self._losses['cross_entropy'],
#                                                                                                   self._losses['loss_box'],
#                                                                                                   self._losses['caption_loss'],
#                                                                                                   self._losses['total_loss'],
#                                                                                                   self._summary_op,
#                                                                                                   train_op],
#                                                                                                   feed_dict=feed_dict)
#     
#     return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, caption_loss, loss, summary
# 
# 
# #   def train_step_no_return(self, sess, blobs, train_op):
# #     feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
# #                  self._gt_boxes: blobs['gt_boxes'],
# #                  self._all_phrases: blobs['all_phrases']}
# #     sess.run([train_op], feed_dict=feed_dict)    

