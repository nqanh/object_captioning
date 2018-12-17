import cv2
import numpy as np
import random
import os
import matplotlib
import matplotlib.pyplot as plt

from model.config import cfg

#force matplotlib use any Xwindows
matplotlib.use('Agg')

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def draw_loss_history(history_rpn_loss_cls, history_rpn_loss_box, history_loss_cls, history_loss_box, history_mask_loss, history_total_loss, output_log_losses_dir, current_interation):
    # summarize history for loss
    plt.plot(history_rpn_loss_cls, color='m')
    plt.plot(history_rpn_loss_box, color='y')
    plt.plot(history_loss_cls, color='k')
    plt.plot(history_loss_box, color='g')
    plt.plot(history_mask_loss, color='b')
    plt.plot(history_total_loss, color='r')
    
    
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['rpn_loss_cls', 'rpn_loss_box', 'loss_cls', 'loss_box', 'mask_loss', 'total_loss'], loc='upper right')
      
    plot_file_name = 'plot_epoch_e' + str(current_interation) + '.png'
    out_plot_path = os.path.join(output_log_losses_dir, plot_file_name)
    plt.savefig(out_plot_path)
    
    

def transform_mono_to_rgb(color_maps, in_mask):
    '''
    convert mono (1 channel) in_mask to 3 channels for visualization
    '''
    out_mask = color_maps.take(in_mask.astype('uint8'), axis=0).astype('uint8')
    return out_mask


def get_label_colours(num_class):
    """
    return color code for visualizing affordance/object map
    """
    all_label_colors = []
    random.seed(255)
    for i in range(num_class):
        curr_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        all_label_colors.append(curr_color)
        
    all_label_colors = np.asarray(all_label_colors)
        
    return all_label_colors  


def fix_resized_mask(input_mask, unique_labels):
    '''
    Convert a "resized_mask" with float value to "integer" mask based on original unique labels and THRESHOLD
    '''
    input_mask = unique_labels[(np.abs(input_mask - unique_labels[:, None, None]) < cfg.THRESHOLD_RESIZE).argmax(0)]
    
    return input_mask


def draw_arrow(image, p, q, color, arrow_magnitude, thickness, line_type, shift):
    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)

    
def draw_reg_text(img, bbox, text='', col=(0, 0, 255)): 
    ''' draw a rectangle with option "text" on the image
    Usage: draw_reg_text(image, bbox, 'input_text' , (0, 255, 0))   
    '''
    
    #img = img.astype(np.int8, copy=True)  ## convert from anything to int8
    
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
    
    # draw rectangle
    draw_arrow(img, (xmin, ymin), (xmax, ymin), col, 0, 5, 8, 0)
    draw_arrow(img, (xmax, ymin), (xmax, ymax), col, 0, 5, 8, 0)
    draw_arrow(img, (xmax, ymax), (xmin, ymax), col, 0, 5, 8, 0)
    draw_arrow(img, (xmin, ymax), (xmin, ymin), col, 0, 5, 8, 0)
    
    # put text
    cv2.putText(img, text, (xmin, ymin-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # draw with red
    #cv2.putText(img, text, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2) # draw with red
    
    return img


def reset_mask_ids(mask, before_uni_ids):
    '''
    reset ID mask values from any values to continuous (e.g., [0, 1, 4] to [0, 1, 2])
    '''
     
    before_uni_ids.sort()
    counter = 0
    for id in before_uni_ids:
        mask[mask == id] = counter
        counter += 1
    return mask