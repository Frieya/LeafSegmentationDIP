import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import glob
from PIL import Image
import gc 
import colorsys
import random


def checkcolour(masks, hsv):
    colours = np.zeros((0,3))

    for i in range(len(masks)):
        color = hsv[masks[i]['segmentation']].mean(axis=(0))
        colours = np.append(colours,color[None,:], axis=0)

    idx_green = (colours[:,0]<75) & (colours[:,0]>35) & (colours[:,1]>35)
    if idx_green.sum()==0:
        # grow lights on adjust
        idx_green = (colours[:,0]<100) & (colours[:,0]>35) & (colours[:,1]>35)

    return(idx_green)

def checkfullplant(masks):
    mask_all = np.zeros(masks[0]['segmentation'].shape[:2])

    for mask in masks:
        mask_all +=mask['segmentation']*1

    iou_withall = []
    for mask in masks:
        iou_withall.append(iou(mask['segmentation'], mask_all>0))

    idx_notall = np.array(iou_withall)<0.9
    return idx_notall

def getbiggestcontour(contours):
    nopoints = [len(cnt) for cnt in contours]
    return(np.argmax(nopoints))

def checkshape(masks):
    cratio = []

    for i in range(len(masks)):
        test_mask = masks[i]['segmentation']

        if not test_mask.max():
            cratio.append(0)
        else:

            contours,hierarchy = cv2.findContours((test_mask*255).astype('uint8'), 1, 2)

            # multiple objects possibly detected. Find contour with most points on it and just use that as object
            cnt = contours[getbiggestcontour(contours)]
            M = cv2.moments(cnt)

            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)

            (x,y),radius = cv2.minEnclosingCircle(cnt)

            carea = np.pi*radius**2

            cratio.append(area/carea)
    idx_shape = np.array(cratio)>0.1
    return(idx_shape)

def iou(gtmask, test_mask):
    intersection = np.logical_and(gtmask, test_mask)
    union = np.logical_or(gtmask, test_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return (iou_score)

def issubset(mask1, mask2):
    # is mask2 subpart of mask1
    intersection = np.logical_and(mask1, mask2)
    return(np.sum(intersection)/mask2.sum()>0.9)

def istoobig(masks):
    idx_toobig = []

    mask_all = np.zeros(masks[0]['segmentation'].shape[:2])

    for mask in masks:
        mask_all +=mask['segmentation']*1

    for idx in range(len(masks)):
        if idx in idx_toobig:
            continue
        for idx2 in range(len(masks)):
            if idx==idx2:
                continue
            if idx2 in idx_toobig:
                continue
            if issubset(masks[idx2]['segmentation'], masks[idx]['segmentation']):
                # check if actually got both big and small copy delete if do
                if mask_all[masks[idx2]['segmentation']].mean() > 1.5:

                    idx_toobig.append(idx2)

    idx_toobig.sort(reverse=True)
    return(idx_toobig)

def remove_toobig(masks, idx_toobig):
    masks_ntb = masks.copy()

    idx_del = []
    for idxbig in idx_toobig[1:]:
        maskbig = masks_ntb[idxbig]['segmentation'].copy()
        submasks = np.zeros(maskbig.shape)

        for idx in range(len(masks_ntb)):
            if idx==idxbig:
                continue
            if issubset(masks_ntb[idxbig]['segmentation'], masks_ntb[idx]['segmentation']):
                submasks +=masks_ntb[idx]['segmentation']

        if np.logical_and(maskbig, submasks>0).sum()/maskbig.sum()>0.9:
            # can safely remove maskbig
            idx_del.append(idxbig)
            del(masks_ntb[idxbig])

    return(masks_ntb)

def show_mask(mask, results, random_color=False):
    if random_color:
        color = np.random.random(3)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255])
    #print(np.shape(color))
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #print(type(mask_image))
    #plt.imshow(mask_image)
    #print(type(results))
    #print(np.shape(results))
    #plt.imshow(results)
    results = np.add(results, mask_image)
    np.clip(results, 0, 255, results)
    del mask
    gc.collect()
    return results

def show_masks_on_image_color(raw_image,masks):
    h, w = np.shape(raw_image)
    r = np.full((h,w,3), (0,0,0), dtype=np.uint8)
    total_masks = len(masks)
    print("Processing", total_masks, "masks...")
    processing_mask = 1
    for idx in range(total_masks):
        converted_mask = masks[idx]['segmentation']
        r = show_mask(converted_mask, r, random_color=True)
        processing_mask+=1
    del masks
    gc.collect()
    return r

def show_mask_on_image_grayscale(raw_image, masks):
    #print("Processing", len(masks), "masks...")
    h, w = np.shape(raw_image)
    results = np.full((h,w), 0, dtype=np.uint8)
    sorted_masks = sorted(masks, key=(lambda x: x['area']),      reverse=True)
    count = 1
    # Plot for each segment area
    for val in sorted_masks:
        mask = val['segmentation']*count
        results = np.add(results, mask)
        count +=1
    return results

def show_output(image, predicted_masks, add_image=False):
    image = np.array(image)
    label_ids = np.unique(predicted_masks)[1:]  # no background
    masks = []
    instance_counter = 0
    for label_id in label_ids:
        instance_counter = instance_counter + 1
        mask = np.isclose(predicted_masks, label_id)
        masks.append(mask)
    num_instances = instance_counter 
    colors = random_colors(num_instances)
    full_mask = np.zeros_like(image).astype(int)
    #print(np.shape(full_mask))
    if add_image:
        full_mask = image
    for idx in range(num_instances):
        full_mask = apply_mask(full_mask, masks[idx], colors[idx], 0.4)
    if add_image:
        return Image.fromarray(full_mask)
    return Image.fromarray((full_mask*255).astype(np.uint8))

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
