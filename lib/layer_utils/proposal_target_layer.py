# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import random
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
from utils.draw_debug_info import draw_reg_text

random.seed(123)

# def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
#   """
#   Assign object detection proposals to ground-truth targets. Produces proposal
#   classification labels and bounding-box regression targets.
#   """
# 
#   # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
#   # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
#   all_rois = rpn_rois
#   all_scores = rpn_scores
# 
#   # Include ground-truth boxes in the set of candidate rois
#   if cfg.TRAIN.USE_GT:
#     zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
#     all_rois = np.vstack(
#       (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
#     )
#     # not sure if it a wise appending, but anyway i am not using it
#     all_scores = np.vstack((all_scores, zeros))
# 
#   num_images = 1
#   rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
#   fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)
# 
#   # Sample rois with classification labels and bounding box regression
#   # targets
#   labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
#     all_rois, all_scores, gt_boxes, fg_rois_per_image,
#     rois_per_image, _num_classes)
# 
#   ## ADD HERE !!!
#   
#   rois = rois.reshape(-1, 5)
#   roi_scores = roi_scores.reshape(-1)
#   labels = labels.reshape(-1, 1)
#   bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
#   bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
#   bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
# 
#   return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights



def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes, all_phrases, img):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  all_scores = rpn_scores

  # Include ground-truth boxes in the set of candidate rois
  if cfg.TRAIN.USE_GT:
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack(
      (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    )
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = np.vstack((all_scores, zeros))

  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)  # 0.25 * 128 = 32

  # Sample rois with classification labels and bounding box regression
  # targets
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights, rois_object_indexes, sentences, answers, answers_masks, positive_rois = _sample_rois(
    all_rois, all_scores, gt_boxes, fg_rois_per_image,
    rois_per_image, _num_classes, all_phrases, img)

  
  rois = rois.reshape(-1, 5)
  roi_scores = roi_scores.reshape(-1)
  labels = labels.reshape(-1, 1)
  bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
  bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)


  if cfg.DEBUG_VERBOSE == 1:
      print('proposal_target_layer.py ========== rois shape                {:s}'.format(rois.shape))                   
      print('proposal_target_layer.py ========== roi_scores shape          {:s}'.format(roi_scores.shape))
      print('proposal_target_layer.py ========== labels shape              {:s}'.format(labels.shape))             
      print('proposal_target_layer.py ========== bbox_targets shape        {:s}'.format(bbox_targets.shape))           
      print('proposal_target_layer.py ========== bbox_inside_weights shape {:s}'.format(bbox_inside_weights.shape))
      
      print('proposal_target_layer.py ========== sentences shape           {:s}'.format(sentences.shape))
      print('proposal_target_layer.py ========== answers   shape           {:s}'.format(answers.shape))
          
      
        
  #return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois_object_indexes, sentences, answers, answers_masks  # Anh add rois_object_indexes
  return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois_object_indexes, sentences, answers, answers_masks, positive_rois
  
def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """

  clss = bbox_target_data[:, 0]
  bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
  bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
  inds = np.where(clss > 0)[0]
  for ind in inds:
    cls = clss[ind]
    start = int(4 * cls)
    end = start + 4
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
    bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return np.hstack(
    (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _find_rois_obj_indexes(gt_boxes, rois, labels, fg_rois_per_image):  ## NOT USE
    '''
    find index of item in gt_boxes for each roi which is object (= fg roi)
    
    fg_rois_per_image: number of fg rois (roi overlab with object) in the current image
    '''
    rois_obj_indexes = []
    
#     # only choose fg roi
    selected_rois = rois[0:fg_rois_per_image,:]
    selected_labels = labels[0:fg_rois_per_image]
    
    
    curr_overlaps = bbox_overlaps(np.ascontiguousarray(selected_rois[:, 1:5], dtype=np.float),
                                  np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
      
    gt_assignment = curr_overlaps.argmax(axis=1)   ## gt_assigment chinh la INDEX can tim!!!!
    max_overlaps = curr_overlaps.max(axis=1)
  
      
    if cfg.DEBUG_VERBOSE == 1:
        print('proposal_target_layer.py -- current overlapss          {:s}'.format(curr_overlaps))
        print('proposal_target_layer.py -- gt_assignment              {:s}'.format(gt_assignment))
        print('proposal_target_layer.py -- max_overlaps               {:s}'.format(max_overlaps))
        
        print('proposal_target_layer.py -- number of fg rois          {:d}'.format(fg_rois_per_image))
        print('proposal_target_layer.py -- selected rois              {:s}'.format(selected_rois))
        print('proposal_target_layer.py -- selected selected_labels   {:s}'.format(selected_labels))
    
    return rois_obj_indexes
    

## sample_rois --> not compact version --> padding when missing 
def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, all_phrases, img):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  # overlaps: (rois x gt_boxes)
  overlaps = bbox_overlaps(
    np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
    np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
  gt_assignment = overlaps.argmax(axis=1)
  max_overlaps = overlaps.max(axis=1)
  labels = gt_boxes[gt_assignment, 4] ## labels = gt_boxes
  
  if cfg.DEBUG_VERBOSE == 1:
      print('proposal_target_layer.py -- max_overlaps shape    {:s}'.format(max_overlaps.shape))       # (2000, ) 
      #print('proposal_target_layer.py -- max_overlaps          {:s}'.format(max_overlaps))             # (256, 0)   # ALL SHAPES ARE FOR PASCAL_VOC 
      print('proposal_target_layer.py -- gt_boxes              {:s}'.format(gt_boxes))                 # (256, 0)   # ALL SHAPES ARE FOR PASCAL_VOC

  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                     (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.size > 0 and bg_inds.size > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
    fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    to_replace = bg_inds.size < bg_rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
  elif fg_inds.size > 0:
    to_replace = fg_inds.size < rois_per_image
    fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = rois_per_image
  elif bg_inds.size > 0:
    to_replace = bg_inds.size < rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  # The indices that we're selecting (both fg and bg)
  keep_inds = np.append(fg_inds, bg_inds)
  # Select sampled values from various arrays:
  labels = labels[keep_inds]
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image):] = 0
  rois = all_rois[keep_inds]
  roi_scores = all_scores[keep_inds]

  bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)
    

  # get number of positive roi      
  fg_rois_per_image = int(fg_rois_per_image)
  
  rois_phrase_obj_indexes = gt_assignment[keep_inds] # gt_assignment keep the index of object that overlaps with the current roi
  rois_phrase_obj_indexes = rois_phrase_obj_indexes[0:fg_rois_per_image] # only keep indexes for for fg rois

  
  ## PREPARE sentence AND answer output. Shape is (num_of_positive_roi, num_lstm_steps) --> DO NOT PAD ANYMORE
  sentences = np.full((fg_rois_per_image, cfg.MAX_PHRASE_LENGTH), 0, dtype='int32') # 
  answers = np.full((fg_rois_per_image, cfg.MAX_PHRASE_LENGTH), 0, dtype='int32')
  answers_masks = np.full((fg_rois_per_image, cfg.MAX_PHRASE_LENGTH), 0, dtype='float32') ## mask the loss later  
  
  positive_rois = rois[0:fg_rois_per_image, ]
  
  
  for i in range(int(fg_rois_per_image)):
      bbox_index = rois_phrase_obj_indexes[i]  ## index of current gt box (with order form .xml file)
      curr_roi_phrases = [] ## keep all phrases of current rois
      for j in range(all_phrases.shape[0]):
          curr_phrase = all_phrases[j] # get current phrase
          if curr_phrase[0] == bbox_index:
              curr_roi_phrases.append(curr_phrase[1:cfg.MAX_PHRASE_LENGTH+1])   # 1st item is the index --> ignore
      
      ## select 2 random indexes
      total_phrases = len(curr_roi_phrases) ## total phrases of current roi
      id_sen, id_ans = -1, -1 # not valid value
      if total_phrases >= 2:
          if cfg.TRAIN.USE_ONE_PHRASE:
              id_sen, id_ans = 0, 0 # same phrase ## TESTING HERE - ONLY USE 1 1ST PHRASE OF ALL PHRASES
          else:
              id_sen, id_ans = random.sample(range(0, total_phrases), 2) # use a random pharse of all phrases
          #print('proposal_target_layer.py -- =========== id_sen {:d} and id_ans {:d}'.format(id_sen, id_ans))
      elif total_phrases == 1:
          id_sen, id_ans = 0, 0 # same phrase
      else:
          print('proposal_target_layer.py -- =========== WARNING ========== found 1 roi does not have any phrases')
          
      if id_sen != -1 and id_ans != -1:
          #print('proposal_target_layer.py -- =========== curr_roi_phrases[id_sen] shape {:s}'.format(curr_roi_phrases[id_sen].shape))  # (1, h, w, c) or (1, w, h, c) ?
          sentences[i] = curr_roi_phrases[id_sen]
          answers[i] = curr_roi_phrases[id_ans]


  ## sanity check, advoid fg_rois_per_image = 0
  if fg_rois_per_image == 0:
      print('proposal_target_layer.py ----- WARNING ----- fg_rois_per_image = 0: There are no positive rois')
      ## consider 1st roi as the only positive rois with empty sentence
      positive_rois = rois[0:1, ]
      sentences = np.full((1, cfg.MAX_PHRASE_LENGTH), 0, dtype='int32') # 
      answers = np.full((1, cfg.MAX_PHRASE_LENGTH), 0, dtype='int32')
      answers_masks = np.full((1, cfg.MAX_PHRASE_LENGTH), 0, dtype='float32') ## mask the loss later  
      
           
  # find answer masks  
  nonzeros = np.array(map(lambda x: (x != 0).sum()+1, answers))  # count all words in the sentence  + 1 --> care about the last word
  for ind, row in enumerate(answers_masks):  
    row[:nonzeros[ind]] = 1.0
                              
                              
  if cfg.DEBUG_VERBOSE == 1:
      print('proposal_target_layer.py -------- sentences          {:s}'.format(sentences))  #
      print('proposal_target_layer.py -------- answers            {:s}'.format(answers))    #
      print('proposal_target_layer.py -------- answers_masks      {:s}'.format(answers_masks))    #  
            
  
  ## Anh them vo
  if cfg.VIS_VERBOSE == 1:
    import cv2
    # load ixtoword dic
    ixtoword = np.load(cfg.IXTOWORD_PATH).item() ## LOAD AS DICT
        
    for i in range(int(fg_rois_per_image)):
        obj_id = labels[i]
        bbox_index = rois_phrase_obj_indexes[i] ## index of current gt bbox 
        curr_roi = rois[i][1:5] ## current roi
        curr_box = gt_boxes[bbox_index] ## current bbox gt
        
        # convert to int
        curr_roi = [int(val) for val in curr_roi]
        curr_box = [int(val) for val in curr_box]
           
        ## DRAW ALL PHRASES
#         all_obj_phrases = 'cls_id=' + str(obj_id) + ' | '
#         for j in range(all_phrases.shape[0]): # find all frame
#             curr_phrase = all_phrases[j]
#             if curr_phrase[0] == bbox_index:
#                 curr_phrase = curr_phrase[1:cfg.MAX_PHRASE_LENGTH+1]  # 1st item is index
#                 curr_str = ''
#                 for ph_id in curr_phrase:
#                     ph_str = ixtoword[int(ph_id)]
#                     if ph_str != '.':
#                         curr_str = curr_str + ph_str + ' '
#                      
#                 all_obj_phrases = all_obj_phrases + curr_str + '*'
                 
                 
        ## DRAW ONLY 1 SELECTED SENTENCE
        all_obj_phrases = 'cls_id=' + str(obj_id) + ' | '
        curr_answer_ids = answers[i]
        curr_answer_str = ''
        for ph_id in curr_answer_ids:
            ph_str = ixtoword[int(ph_id)]
            if ph_str != '.':
                curr_answer_str = curr_answer_str + ph_str + ' '
                
        if cfg.DEBUG_VERBOSE:
            print ('********** proposal_target_layer.py ------- curr anser str: {}'.format(curr_answer_str))
            
        all_obj_phrases = all_obj_phrases + curr_answer_str
        
        
        # draw debug
        curr_img = img.copy()
        curr_img = np.squeeze(curr_img, 0)
        #print('+++++++++++++++++++ curr_img shape                 {:s}'.format(curr_img.shape))  # (h, w, c) or (w, h, c) ?
         
        curr_img = draw_reg_text(curr_img, curr_box, '' , (255, 0, 0))
        curr_img = draw_reg_text(curr_img, curr_roi, all_obj_phrases , (0, 255, 0))
        cv2.imshow('IMG', curr_img)
        cv2.waitKey(0)
  
  ## And them vo
  if cfg.DEBUG_VERBOSE == 1:
      print('proposal_target_layer.py -- img shape                 {:s}'.format(img.shape))  # (1, h, w, c) or (1, w, h, c) ?
      print('proposal_target_layer.py -- selected_rois_object_idx  {:s}'.format(rois_phrase_obj_indexes))                 
      print('proposal_target_layer.py -- fg_rois_per_img           {:d}'.format(int(fg_rois_per_image)))                 
      print('proposal_target_layer.py -- labels shape              {:s}'.format(labels.shape))                 # (256, 0)   # ALL SHAPES ARE FOR PASCAL_VOC
      print('proposal_target_layer.py -- rois shape                {:s}'.format(rois.shape))                   # (256, 5)
      print('proposal_target_layer.py -- roi_scores shape          {:s}'.format(roi_scores.shape))             # (256, 1)
      print('proposal_target_layer.py -- bbox_targets shape        {:s}'.format(bbox_targets.shape))           # (256, 84)  # 84 = 21*4 (21 obj classes
      print('proposal_target_layer.py -- bbox_inside_weights shape {:s}'.format(bbox_inside_weights.shape))    # (256, 84)
      
      print('proposal_target_layer.py -- labels value              {:s}'.format(labels))    
      print('proposal_target_layer.py -- rois value                {:s}'.format(rois))    
      #print('proposal_target_layer.py -- rois score value          {:s}'.format(roi_scores))
      print('proposal_target_layer.py -- bbox_targets value        {:s}'.format(bbox_targets))
      print('proposal_target_layer.py -- all phrases               {:s}'.format(all_phrases))
      print('proposal_target_layer.py -- positive roi              {:s}'.format(positive_rois))
      print('proposal_target_layer.py -- positive roi shape        {:s}'.format(positive_rois.shape))   ## (xx, 5) changing based on num_of_positive_roi
     
     
  sentences = sentences.astype(np.int32)  ## from float to int32 --> for embed_lookup later     
  answers = answers.astype(np.int32)
   
  #return labels, rois, roi_scores, bbox_targets, bbox_inside_weights, rois_phrase_obj_indexes, sentences, answers, answers_masks
  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights, rois_phrase_obj_indexes, sentences, answers, answers_masks, positive_rois
