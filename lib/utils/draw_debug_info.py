import cv2
import numpy as np


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

    
def draw_reg_text(img, bbox, text, col):
    #print 'tbd'
    
    img = img.astype(np.int8, copy=True)  ## convert from float to int8
    
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
    
    
    draw_arrow(img, (xmin, ymin), (xmax, ymin), col, 0, 5, 8, 0)
    draw_arrow(img, (xmax, ymin), (xmax, ymax), col, 0, 5, 8, 0)
    draw_arrow(img, (xmax, ymax), (xmin, ymax), col, 0, 5, 8, 0)
    draw_arrow(img, (xmin, ymax), (xmin, ymin), col, 0, 5, 8, 0)
    
    # put text
    txt_obj = text
    
    #cv2.putText(img, txt_obj, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1) # draw with red
    cv2.putText(img, txt_obj, (xmin, ymin+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # draw with red
    
    #cv2.putText(img, txt_obj, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, col_map[obj_id], 2)
    
#     # draw center
#     center_x = (xmax - xmin)/2 + xmin
#     center_y = (ymax - ymin)/2 + ymin
#     cv2.circle(img,(center_x, center_y), 3, (0, 255, 0), -1)
    
    return img