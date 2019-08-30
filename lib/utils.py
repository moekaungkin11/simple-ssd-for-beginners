import numpy as np
import torch
import cv2

def point_form(boxes):
    '''
     Convert [cx, cy, w, h] type to [xmin, ymin, xmax, ymax] form
    '''

    tl = boxes[:, :2] - boxes[:, 2:]/2
    br = boxes[:, :2] + boxes[:, 2:]/2

    return np.concatenate([tl, br], axis=1)


def detection_collate(batch):
    '''
    Because the gt number of each sample is not necessarily, we have to define the splicing function ourselves.
    '''
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs), np.array(targets)

def bbox_iou(box_a, box_b):
    '''
    Calculate the iou of two box arrays
    box_a : (m, 4)
    box_b : (n, 4)
    '''
    m = box_a.shape[0]
    n = box_b.shape[0]

    #broadcasting, this is equivalent to (m,1,2) and (1,n,2) operations, and finally get (m,n,2) size
    tl = np.maximum(box_a[:, None, :2], box_b[None, :, :2])
    br = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:])

    wh = np.maximum(br-tl, 0)
    
    inner = wh[:, :, 0]*wh[:, :, 1]

    a = box_a[:, 2:] - box_a[:, :2]
    b = box_b[:, 2:] - box_b[:, :2]

    a = a[:, 0] * a[:, 1]
    b = b[:, 0] * b[:, 1]

    a = a[:, None]
    b = b[None, :]

    #The last is equivalent to (m,n) / (m, 1) + (1,n) - (m,n)
    #Get a matrix of (m, n), where each point (i, j) represents iou of i and j

    return inner / (a+b-inner)


def nms(boxes, score, threshold=0.4):
    '''
    Performing nms operation through iou
    boxes : (n, 4)
    score: (n, )
    '''

    sort_ids = np.argsort(score)
    pick = []
    while len(sort_ids) > 0:
        #First take a box with the highest probability
        i = sort_ids[-1]
        pick.append(i)
        if len(sort_ids) == 1:
            break

     #Use this box to perform iou calculation with the remaining boxes, and delete the box whose iou is larger than a certain threshold, and end when the remaining boxes are less than one.
     sort_ids = sort_ids[:-1]
     box = boxes[i].reshape(1, 4)
     ious = bbox_iou(box, boxes[sort_ids]).reshape(-1)
     sort_ids = np.delete(sort_ids, np.where(ious > threshold)[0])
     return pick

def detect(locations, scores, nms_threshold, gt_threshold):
    '''
   Locations : coordinates after decoding (num_anchors, 4)
    Scores : predicted scores (num_anchors, 21)
    Nms_threshold : threshold of nms
    Gt_threshold : is considered the threshold of the ground truth of the real object
    '''
    scores = scores[:, 1:] #Class 0 is the background, filtered out

    #Store the information of the last retained object, his coordinates, confidence and which category it belongs to
    keep_boxes = []
    keep_confs = []
    keep_labels = []
    
    #Detect each class
    for i in range(scores.shape[1]):
        #Get an anchor of objects in this class
        mask = scores[:, i] >= gt_threshold
        label_scores = scores[mask, i] 
        label_boxes = locations[mask]
        #Did not find the next next category
        if len(label_scores) == 0:
            continue

        #Performing nms
        pick = nms(label_boxes, label_scores, threshold=nms_threshold)
        label_scores = label_scores[pick]
        label_boxes = label_boxes[pick]
        
        keep_boxes.append(label_boxes.reshape(-1))
        keep_confs.append(label_scores)
        keep_labels.extend([i]*len(label_scores))
    
    #Did not find any object
    if len(keep_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
         
    keep_boxes = np.concatenate(keep_boxes, axis=0).reshape(-1, 4)
    keep_confs = np.concatenate(keep_confs, axis=0)
    keep_labels = np.array(keep_labels).reshape(-1)
#     print(keep_boxes.shape)
#     print(keep_confs.shape)
#     print(keep_labels.shape)

    return keep_boxes, keep_confs, keep_labels

def draw_rectangle(src_img, labels, conf, locations, label_map):
    '''
    Src_img : the picture to be framed
    Labels : objects get labels, numbers
    Conf : probability of having objects at this place
    Locations : coordinates
    Label_map : map labels back to category names
    Return
        Picture on the frame
    '''
    num_obj = len(labels)
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    img = src_img.copy()
    for i in range(num_obj):
        tl = tuple(locations[i][:2])
        br = tuple(locations[i][2:])
        
        cv2.rectangle(img,
                      tl,
                      br,
                      COLORS[i%3], 3)
        cv2.putText(img, label_map[labels[i]], tl,
                    FONT, 1, (255, 255, 255), 2)
    
    img = img[:, :, ::-1]

    return img
