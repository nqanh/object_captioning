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
from layer_utils.proposal_layer import proposal_layer
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

#   # Anh them vo
#   def __init__(self):
#     self._predictions = {}
#     self._losses = {}
#     self._anchor_targets = {}
#     self._proposal_targets = {}
#     self._layers = {}
#     self._gt_image = None
#     self._act_summaries = []
#     self._score_summaries = {}
#     self._train_summaries = []
#     self._event_summaries = {}
#     self._variables_to_fix = {}
#     
#     # Anh them vo
#     ## LSTM stuff
#     self.dim_img_feature = cfg.dim_img_feature
#     self.num_lstm_hidden_units = cfg.num_lstm_hidden_units
#     self.vocab_size = cfg.vocab_size
#     self.batch_size = cfg.batch_size
#     self.num_lstm_steps = cfg.num_lstm_steps
#     self.bias_init_vector = cfg.bias_init_vector
#      
#      
#     with tf.device('/cpu:0'):
#         #self.Wemb = tf.Variable(tf.random_uniform([options['q_vocab_size'] + 1, options['embedding_size']], -0.1, 0.1), name='Wemb_in')   # TEMPO state - keep words to feed to LSTM2
#         self.Wemb = tf.Variable(tf.random_uniform([self.vocab_size, self.num_lstm_hidden_units], -0.1, 0.1), name='Wemb_in')   # TEMPO state - keep words to feed to LSTM2
#          
#     # define lstm
#     self.lstm1 = tf.contrib.rnn.LSTMCell(self.num_lstm_hidden_units) # handle input sentence
#     self.lstm2 = tf.contrib.rnn.LSTMCell(self.num_lstm_hidden_units) # handle output sentence
#          
#     # for output sentence    
#     self.embed_word_W = tf.Variable(tf.random_uniform([self.num_lstm_hidden_units, self.vocab_size], -0.1,0.1), name='embed_word_W')
#     if self.bias_init_vector is not None:
#         self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
#     else:
#         self.embed_word_b = tf.Variable(tf.zeros(self.vocab_size), name='embed_word_b')
#                  
#     ## weight for image --> transform img from (batch_size, 4096) * (batch_size, dim_hidden) + (batch_size, dim_hidden) --> (batch_size, dim_hidden)
#     self.Wimg = self.init_weight(self.dim_img_feature, self.num_lstm_hidden_units, name = 'Wimg')
#     self.bimg = self.init_bias(self.num_lstm_hidden_units, name = 'bimg')
#          
#     # weight for combining image + input sentence
#     self.Wcombine_img_sen = self.init_weight(self.num_lstm_hidden_units * 2, self.num_lstm_hidden_units, name='Wcombine_img_sen') ## shape = (dim_hidden * 2, dim_hidden)
#     self.bcombine_img_sen = self.init_bias(self.num_lstm_hidden_units, name='bcombine_img_sen')
     

  # Anh them vo
  def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
    return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

  def init_bias(self, dim_out, name=None):
    return tf.Variable(tf.zeros([dim_out]), name=name)

    
    
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

      # check here
  def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
          rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois_phrase_obj_indexes, sentences, answers = tf.py_func(
            proposal_target_layer,
            [rois, roi_scores, self._gt_boxes, self._num_classes, self._all_phrases, self._image], # Anh them all_phrases
            [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64, tf.int32, tf.int32],
            name="proposal_target")
          
          rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
          roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
          labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
          bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
          bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
          bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
          
          self._proposal_targets['rois'] = rois
          self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
          self._proposal_targets['bbox_targets'] = bbox_targets
          self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
          self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
          
          self._score_summaries.update(self._proposal_targets)
          
          return rois, roi_scores

    # Anh them vo
#   def _proposal_target_layer(self, rois, roi_scores, name):
#       with tf.variable_scope(name) as scope:
#         rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois_phrase_obj_indexes, sentences, answers = tf.py_func(
#           proposal_target_layer,
#           [rois, roi_scores, self._gt_boxes, self._num_classes, self._all_phrases, self._image], # Anh them all_phrases
#           [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int64, tf.int32, tf.int32],
#           name="proposal_target")
#      
#         rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
#         roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
#         labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
#         bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
#         bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
#         bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
#            
#         sentences.set_shape([cfg.TRAIN.BATCH_SIZE, cfg.MAX_PHRASE_LENGTH]) # check lai cfg.TRAIN.BATCH_SIZE vs. cfg.TRAIN.RPN_BATCHSIZE
#         answers.set_shape([cfg.TRAIN.BATCH_SIZE, cfg.MAX_PHRASE_LENGTH]) 
#            
#         self._proposal_targets['rois'] = rois
#         self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
#         self._proposal_targets['bbox_targets'] = bbox_targets
#         self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
#         self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
#            
#         self._proposal_targets['sentences'] = sentences # Anh them vo
#         self._proposal_targets['answers'] = answers
#      
#         self._score_summaries.update(self._proposal_targets)
#      
#         return rois, roi_scores, sentences, answers  # Anh add sentences, answers
  
  
  # Anh them vo
  def _caption_generation(self, pool5, sentence, answer):  # must be ONE sentence ONLY
    #with tf.variable_scope(self._scope, self._scope, reuse=reuse):  # no scope here
    pool5_flat = slim.flatten(pool5, scope='flatten_rnn')
   
    ## NEED TO REMOVE THESE PLACEHOLDER
#         fc7_features = tf.placeholder('float32',[self.batch_size, self.dim_img_feature], name = 'fc7')
#         sentence = tf.placeholder('int32', [self.batch_size, self.num_lstm_steps], name = "sentence") # int32, [batch_size, num_of_lstm_step
#         answer   = tf.placeholder('int32', [self.batch_size, self.num_lstm_steps], name = "answer")
#         
#         # mask the loss ## only apply for output mask
#         answer_mask   = tf.placeholder('float32', [self.batch_size, self.num_lstm_steps])
     
    # tempo state
    # use cell object to init 
    state1 = self.lstm1.zero_state(self.batch_size, tf.float32)  # --> return a LSTMStateTuple
    state2 = self.lstm2.zero_state(self.batch_size, tf.float32)
    padding = tf.zeros([self.batch_size, self.num_lstm_hidden_units])
     
    probs = []
    loss = 0.0
     
    with tf.variable_scope(tf.get_variable_scope()) as scope:   #TO FIX: https://github.com/tensorflow/tensorflow/issues/6220
  
        for i in range(self.num_lstm_steps): 
            if i > 0:   # if i=0 --> create new variables then reuse
                tf.get_variable_scope().reuse_variables()
             
            ## LOOKUP FOR input sentence
            word_emb_in = tf.nn.embedding_lookup(self.Wemb, sentence[:,i])  ## LOOK UP FOR Sentence --> input phrases 
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(word_emb_in, state1)  # don't care about the input image ???
             
            # HANDLE img_featue
            image_embedding = tf.nn.xw_plus_b(fc7_features, self.Wimg, self.bimg, name='img_embedding') # output_size = (batch_size, dim_hidden)
            image_embedding = tf.nn.tanh(image_embedding) ## make it non-linear
             
            # combine img feature with output1 (sentence featue)
            sen_img_embedding = tf.concat([output1, image_embedding], 1) ## concat axis = 1; 0 axis keeps batch_size; output will be (batch_size, dim_hiden * 2)
            #sen_img_embedding = tf.concat(([3, 3, 3], [4, 4, 4]), 1) ## concat axis = 1; 0 axis keeps batch_size; output will be (batch_size, dim_hiden * 2)
             
            # learn sen_img_embeddign again to make it become (batch_size, dim)    
            output_with_sen_and_img = tf.nn.xw_plus_b(sen_img_embedding, self.Wcombine_img_sen, self.bcombine_img_sen) # output shape = (batch_size, dim_hidden)
            output_with_sen_and_img = tf.nn.tanh(output_with_sen_and_img)
             
                         
            ## LOOKUP FOR output sentence  
            if i == 0:
                word_embed_out = tf.zeros([self.batch_size, self.num_lstm_hidden_units])  ## i=0: no previous word --> start with zero
            else:                                                                                                           
                with tf.device("/cpu:0"):
                    word_embed_out = tf.nn.embedding_lookup(self.Wemb, answer[:, i-1]) # Function: Find ROWS of Wemb for a list of WORD in caption[:,i-1]  
                                                                                       # caption[:, i-1] returns all values (WORDS) at COLUMN i-1
                                                                                       # caption shape (batch_size, pink_box)
              
            with tf.variable_scope("LSTM2"):
                #output2, state2 = self.lstm2( tf.concat([word_embed_out, output1], 1), state2 )  ### feed captions
                output2, state2 = self.lstm2(tf.concat([word_embed_out, output_with_sen_and_img], 1), state2)  ### USE output_with_sen_and_img, NOT output1
  
            labels = tf.expand_dims(answer[:, i], 1)    # get currrent caption???     
     
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)   # tf.range() creates a sequence of number  # indice is the index
     
            concated = tf.concat([indices, labels], 1)
              
            #onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)  ### use tf.one_hot() ???
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.vocab_size]), 1.0, 0.0)  # tf.pack --> tf.stack in TF1
             
            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)   ### lost for words
                
            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)   ### logit and onehot must have the same shape (batch_size, num_class)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words)
              
            cross_entropy = cross_entropy * answer_mask[:, i]   # caption_mask - only count the lost of real data (image/word), ignore the padding
              
            probs.append(logit_words)
  
            current_loss = tf.reduce_sum(cross_entropy)  # sum all cross_entropy loss of all batch (100)
            loss += current_loss
         
    loss = loss/tf.reduce_sum(answer_mask) # average loss over all words
        
    return loss
  
  
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
          rois = self._region_proposal(net_conv, is_training, initializer)
          # region of interest pooling
          if cfg.POOLING_MODE == 'crop':
            pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
          else:
            raise NotImplementedError
          
        fc7 = self._head_to_tail(pool5, is_training)
        with tf.variable_scope(self._scope, self._scope):
          # region classification
          cls_prob, bbox_pred = self._region_classification(fc7, is_training, 
                                                            initializer, initializer_bbox)
          
        self._score_summaries.update(self._predictions)
          
        return rois, cls_prob, bbox_pred

  
# #   # Anh them vo
#   def _build_network(self, is_training=True):
#       # select initializers
#       if cfg.TRAIN.TRUNCATED:
#         initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
#         initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
#       else:
#         initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
#         initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
#      
#       net_conv = self._image_to_head(is_training)
#       with tf.variable_scope(self._scope, self._scope):
#         # build the anchors for the image
#         self._anchor_component()
#         # region proposal network
#         #rois = self._region_proposal(net_conv, is_training, initializer)
#         rois, sentences, answers = self._region_proposal(net_conv, is_training, initializer)  ## tra ra nhieu rois?
#         
#         # region of interest pooling
#         if cfg.POOLING_MODE == 'crop':
#           pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
#         else:
#           raise NotImplementedError
#      
#       fc7 = self._head_to_tail(pool5, is_training)
#       with tf.variable_scope(self._scope, self._scope):
#         # region classification
#         cls_prob, bbox_pred = self._region_classification(fc7, is_training, 
#                                                           initializer, initializer_bbox)
#      
#         # Anh them vo
#         caption_loss = self._caption_generation(pool5, sentences, answers) # sentences is (roi_batch_size --> 256, 10): each roi has 1 sentence --> map to 1 answer
#                                                                                         # use pool5 features for now (should be fc7?) 
#            
#       self._score_summaries.update(self._predictions)
#      
#       return rois, cls_prob, bbox_pred, caption_loss


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


  def _add_losses(self, sigma_rpn=3.0):
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
          
          loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
          regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
          self._losses['total_loss'] = loss + regularization_loss
          
          self._event_summaries.update(self._losses)
          
        return loss


#   ## Anh them vo
#   def _add_losses(self, caption_loss, sigma_rpn=3.0):
#       with tf.variable_scope('LOSS_' + self._tag) as scope:
#         # RPN, class loss
#         rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
#         rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
#         rpn_select = tf.where(tf.not_equal(rpn_label, -1))
#         rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
#         rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
#         rpn_cross_entropy = tf.reduce_mean(
#           tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
#      
#         # RPN, bbox loss
#         rpn_bbox_pred = self._predictions['rpn_bbox_pred']
#         rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
#         rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
#         rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
#         rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
#                                             rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
#      
#         # RCNN, class loss
#         cls_score = self._predictions["cls_score"]
#         label = tf.reshape(self._proposal_targets["labels"], [-1])
#         cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
#      
#         # RCNN, bbox loss
#         bbox_pred = self._predictions['bbox_pred']
#         bbox_targets = self._proposal_targets['bbox_targets']
#         bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
#         bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
#         loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
#      
#         self._losses['cross_entropy'] = cross_entropy
#         self._losses['loss_box'] = loss_box
#         self._losses['rpn_cross_entropy'] = rpn_cross_entropy
#         self._losses['rpn_loss_box'] = rpn_loss_box
#         
#         self._losses['caption_loss'] = caption_loss
#      
#         # Anh them vo
#         #loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
#         loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + caption_loss
#            
#         regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
#         self._losses['total_loss'] = loss + regularization_loss
#      
#         self._event_summaries.update(self._losses)
#      
#       return loss



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
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois") ## return "sentences" and "answers" for lstm
      else:
        if cfg.TEST.MODE == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.TEST.MODE == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError
     
      self._predictions["rpn_cls_score"] = rpn_cls_score
      self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
      self._predictions["rpn_cls_prob"] = rpn_cls_prob
      self._predictions["rpn_cls_pred"] = rpn_cls_pred
      self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
      self._predictions["rois"] = rois
         
      return rois

#     # Anh them vo
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
#           rois, _, sentences, answers = self._proposal_target_layer(rois, roi_scores, "rpn_rois") ## return "sentences" and "answers" for lstm
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
#       self._predictions["sentences"] = sentences  ## WRONG HERE
#       self._predictions["answers"] = answers
#      
#       return rois, sentences, answers



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
          
        # Anh them vo
        self._all_phrases = tf.placeholder(tf.float32, shape=[None, cfg.MAX_PHRASE_LENGTH + 1])
              
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
          rois, cls_prob, bbox_pred = self._build_network(training)
          
        layers_to_output = {'rois': rois}
          
        for var in tf.trainable_variables():
          self._train_summaries.append(var)
          
        if testing:
          stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
          means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
          self._predictions["bbox_pred"] *= stds
          self._predictions["bbox_pred"] += means
        else:
          self._add_losses()
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
          
        return layers_to_output


#     # Anh them vo
#   def create_architecture(self, mode, num_classes, tag=None,
#                             anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
#       self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
#       self._im_info = tf.placeholder(tf.float32, shape=[3])
#       self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
#       self._tag = tag
#      
#       self._num_classes = num_classes
#       self._mode = mode
#       self._anchor_scales = anchor_scales
#       self._num_scales = len(anchor_scales)
#      
#       self._anchor_ratios = anchor_ratios
#       self._num_ratios = len(anchor_ratios)
#      
#       self._num_anchors = self._num_scales * self._num_ratios
#      
#       # Anh them vo
#       self._all_phrases = tf.placeholder(tf.float32, shape=[None, cfg.MAX_PHRASE_LENGTH + 1])
#          
#       training = mode == 'TRAIN'
#       testing = mode == 'TEST'
#      
#       assert tag != None
#      
#       # handle most of the regularizers here
#       weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
#       if cfg.TRAIN.BIAS_DECAY:
#         biases_regularizer = weights_regularizer
#       else:
#         biases_regularizer = tf.no_regularizer
#      
#       # list as many types of layers as possible, even if they are not used now
#       with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
#                       slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
#                       weights_regularizer=weights_regularizer,
#                       biases_regularizer=biases_regularizer, 
#                       biases_initializer=tf.constant_initializer(0.0)): 
#         rois, cls_prob, bbox_pred, caption_loss = self._build_network(training)
#      
#       layers_to_output = {'rois': rois}
#      
#       for var in tf.trainable_variables():
#         self._train_summaries.append(var)
#      
#       if testing:
#         stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
#         means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
#         self._predictions["bbox_pred"] *= stds
#         self._predictions["bbox_pred"] += means
#       else:
#         self._add_losses(caption_loss)
#         layers_to_output.update(self._losses)
#      
#         val_summaries = []
#         with tf.device("/cpu:0"):
#           val_summaries.append(self._add_gt_image_summary())
#           for key, var in self._event_summaries.items():
#             val_summaries.append(tf.summary.scalar(key, var))
#           for key, var in self._score_summaries.items():
#             self._add_score_summary(key, var)
#           for var in self._act_summaries:
#             self._add_act_summary(var)
#           for var in self._train_summaries:
#             self._add_train_summary(var)
#      
#         self._summary_op = tf.summary.merge_all()
#         self._summary_op_val = tf.summary.merge(val_summaries)
#      
#       layers_to_output.update(self._predictions)
#      
#       return layers_to_output



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

#   def get_summary(self, sess, blobs):
#     feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
#                  self._gt_boxes: blobs['gt_boxes']}
#     summary = sess.run(self._summary_op_val, feed_dict=feed_dict)
# 
#     return summary

#   def train_step(self, sess, blobs, train_op):
#     feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
#                  self._gt_boxes: blobs['gt_boxes']}
#     rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
#                                                                         self._losses['rpn_loss_box'],
#                                                                         self._losses['cross_entropy'],
#                                                                         self._losses['loss_box'],
#                                                                         self._losses['total_loss'],
#                                                                         train_op],
#                                                                        feed_dict=feed_dict)
#     return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

#   def train_step_with_summary(self, sess, blobs, train_op):  # blobs come from minibatch.py
#     feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
#                  self._gt_boxes: blobs['gt_boxes']}
#     rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
#                                                                                  self._losses['rpn_loss_box'],
#                                                                                  self._losses['cross_entropy'],
#                                                                                  self._losses['loss_box'],
#                                                                                  self._losses['total_loss'],
#                                                                                  self._summary_op,
#                                                                                  train_op],
#                                                                                 feed_dict=feed_dict)
#     return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

#   def train_step_no_return(self, sess, blobs, train_op):
#     feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
#                  self._gt_boxes: blobs['gt_boxes']}
#     sess.run([train_op], feed_dict=feed_dict)
    

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
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                        self._losses['rpn_loss_box'],
                                                                        self._losses['cross_entropy'],
                                                                        self._losses['loss_box'],
                                                                        self._losses['total_loss'],
                                                                        train_op],
                                                                       feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

  def train_step_with_summary(self, sess, blobs, train_op):  # blobs come from minibatch.py
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes'],
                 self._all_phrases: blobs['all_phrases']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                 self._losses['rpn_loss_box'],
                                                                                 self._losses['cross_entropy'],
                                                                                 self._losses['loss_box'],
                                                                                 self._losses['total_loss'],
                                                                                 self._summary_op,
                                                                                 train_op],
                                                                                feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary


  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes'],
                 self._all_phrases: blobs['all_phrases']}
    sess.run([train_op], feed_dict=feed_dict)

