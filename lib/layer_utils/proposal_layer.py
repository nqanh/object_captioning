# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob, scores
  

## Anh them vo
def proposal_layer_test_caption(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors, input_test_phrase):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

##  GOOD to create sentences = rois shape --> need to change to cfg.max_test_batch_size
#   # Anh them vo --> copy input sentence for all rois --> each roi will have 1 sentence
#   sentences = np.full((blob.shape[0], cfg.MAX_PHRASE_LENGTH), 0, dtype='int32') ## (300, 10) --> during testing, keep 300 roi
#   sentences[:,] = input_test_phrase ## broadcast in_phrase to all rows of sentences 
     
  sentences = np.full((cfg.max_test_batch_size, cfg.MAX_PHRASE_LENGTH), 0, dtype='int32') ## (300, 10) --> during testing, keep 300 roi
  sentences[:,] = input_test_phrase ## broadcast in_phrase to all rows of sentences 
        
  num_of_rois = blob.shape[0]
  
  # create a new blob with shape = (max_test_batch_size, feature_dim)
  new_blob = np.full((cfg.max_test_batch_size, blob.shape[1]), 0, dtype='float32')
  new_scores = np.full((cfg.max_test_batch_size, scores.shape[1]), 0, dtype='float32') ## new score with shapes 
  
  
  if num_of_rois > cfg.max_test_batch_size:
      print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... WARNING --- number of output rois > {:d}'.format(cfg.max_test_batch_size))       
      new_blob = blob[0:cfg.max_test_batch_size, ] # TO FIX: now get only first (max_test_batch_size) rois
      new_scores = scores[0:cfg.max_test_batch_size, ]         
  else:
      # pad empty roi with the first roi ~~ similar to score
      first_roi = blob[0]
      first_score = scores[0]
      
      new_blob[0:num_of_rois, ] = blob
      new_scores[0:num_of_rois, ] = scores
      
      for idx in range(num_of_rois, cfg.max_test_batch_size):
          new_blob[idx] = first_roi  ## pad empty roi with the first roi --> will not affect the result
          new_scores[idx] = first_score
      
  if cfg.DEBUG_VERBOSE == 1:   
      print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... blob shape                {:s}'.format(blob.shape))        # e.g. (210, 5)
      print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... sentences shape           {:s}'.format(sentences.shape))   # e.g. (210, 10)      
      print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... sentences                 {:s}'.format(sentences))
      print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... score shape               {:s}'.format(scores.shape))                  
      print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... num_of_rois               {:d}'.format(num_of_rois))       # e.g. (210) 
      print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... new blob shape            {:s}'.format(new_blob.shape))                

  #return blob, scores, sentences
  return new_blob, new_scores, sentences
  
  
## Anh them vo
## only return real rois, not padding
def proposal_layer_test_caption_compact(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))


     
#   if cfg.DEBUG_VERBOSE == 1:   
#       print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... blob shape                {:s}'.format(blob.shape))        # e.g. (210, 5)
#       print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... sentences shape           {:s}'.format(sentences.shape))   # e.g. (210, 10)      
#       print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... sentences                 {:s}'.format(sentences))
#       print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... score shape               {:s}'.format(scores.shape))                  
#       print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... blob (rois)               {:s}'.format(blob))  
#       print('_region_proposal.py ~~~~~~~~~~~~~ TESTING ... score                     {:s}'.format(scores))
      

  return blob, scores
  #return new_blob, new_scores, sentences
