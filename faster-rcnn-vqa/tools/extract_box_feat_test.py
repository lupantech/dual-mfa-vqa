#!/usr/bin/env python
#coding:utf-8

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import _init_paths  # import _init_paths.py
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt 
import numpy as np   
import scipy.io as sio
import caffe, os, sys, cv2
import argparse  
import time
import h5py

CLASSES = ( '__background__',
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
            'bus', 'train', 'truck', 'boat', 'traffic light', 
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 
            'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
            'cake', 'chair', 'couch', 'potted plant', 'bed', 
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush') 

NETS = {'vgg16': ('VGG16',  
                  'VGG16_faster_rcnn_final.caffemodel',       
                  'coco_vgg16_faster_rcnn_final.caffemodel'),  
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


### visualize detextion: bounding boxes, classes, scores
def vis_detections(im, dets, num):
    """Draw detected bounding boxes."""

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in xrange(num):
        bbox = dets[i, :4]  # box: Xmin,Ymin,Xmax,Ymax
        score = dets[i, -2] # score
        class_name = CLASSES[ dets[i, -1].astype(np.int32) ] # class name

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()


### Apply pre-trained model to testing image
def demo(net, image_name, image_file):
    """Detect object classes in an image using pre-computed object proposals."""

    im_file = os.path.join(image_file, image_name)
    im = cv2.imread(im_file) 

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im) 
    feature = net.blobs["fc7"].data # maybe (146, 4096), (300, 4096)

    # Get 300 boxes and its most confident class, dets = [ [cls_box,score, cls] ]
    for i in xrange(boxes.shape[0]):
        score = max(scores[i, 1:])
        cls_ind = np.argmax(scores[i, 1:]) + 1
        box = boxes[i, 4*cls_ind:4*(cls_ind + 1)]
        det = np.hstack( (box, score) ).astype(np.float32) 
        if i == 0:
            dets = det
        else:
            dets = np.vstack( (dets, det) ) 

    # Narrow boxes with THRES: maybe 30 boxes
    NMS_THRESH = 0.3
    keep = []
    while len(keep) < 19 :
        keep = nms(dets, NMS_THRESH)    # narrow dets
        NMS_THRESH += 0.1
    assert len(keep) >= 19
    dets = dets[keep, :]

    scores = scores[keep, 1:] #(20,80)
    cls_inds = np.argmax( scores, axis = 1 ) + 1
    dets = np.hstack( (dets, cls_inds[:, np.newaxis]) ).astype(np.float32) 

    # Select top 19 boxes
    NUM_THRESH = 19 
    dets = dets[:NUM_THRESH, :]
    keep = keep[:NUM_THRESH]
    num = len(keep)

    assert num == 19
    print "Qualifed object proposals = ", num

    # Visualize detections for each class
    if args.vis_result:
        vis_detections(im, dets, num) 

    # Save feature
    feat = np.hstack( (feature[keep,:], dets[:,-2][:, np.newaxis] ) ).astype(np.float32) # fc7_feature_4096, score
    output_file.create_dataset(image_name, data=feat)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset_type',     # basketball_val
                        help='dataset type to test',
                        default='coco', type=str)
    parser.add_argument('--conf', dest='confidence_socre', help='confidence socre',
                        default=0.8, type=float)
    parser.add_argument('--vis', dest='vis_result', help='visualize result',
                        default=False, type=float)

    args = parser.parse_args()

    return args


if __name__ == '__main__':   
    
    args = parse_args() 

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.MODELS_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, 'models', args.dataset_type))
    CONF_THRESH = args.confidence_socre
    
    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    # /home/plu/py-faster-rcnn/models/coco/VGG16/faster_rcnn_end2end/test.prototxt
    
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][2])
    # /home/plu/py-faster-rcnn/data/faster_rcnn_models/coco_vgg16_faster_rcnn_final.caffemodel

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)   # network configuration
    print '\n\n Loaded network {:s}'.format(caffemodel)

    target_path = '../../VQA/Images/mscoco/'
    subtype = ['test2015/']
    output_file = h5py.File('../../VQA/Features/faster-rcnn_features_19_test.h5','w')

        total_images = 0
    t1 = time.time()

    for filename in subtype:
        image_file = os.path.join(target_path, filename)
        for i, im_name in enumerate(os.listdir(image_file)):
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Demo for {}'.format(im_name)
            feat = demo(net, im_name, image_file)  # one image by one
        total_images = total_images + i + 1
        
    if args.vis_result:
        plt.show()  

    t2 = time.time()
    print("\nEclipse %.2f seconds for %d images") %(t2-t1, total_images)

    output_file.close()
    print "Done!"
