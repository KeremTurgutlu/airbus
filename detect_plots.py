from matplotlib.patches import Rectangle
from matplotlib import patheffects
import matplotlib.pyplot as plt
from detect_utils import actn_to_bb
from fastai.core import to_np
import torch.nn.functional as F
import numpy as np
from detect_utils import nms

def draw_bbox(bbox, ax, col='red'):
    """min row, min col, max row, max col - like np """
    min_y, min_x, max_y, max_x = bbox
    ax.add_patch(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
     fill=False, color=col))

def draw_bbox_on_img(img, bboxes, fig_sz=(10, 10), ax=None):
    """img: image array bboxes: list of list of size 4"""
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=fig_sz)
    ax.imshow(img)
    for bbox in bboxes: draw_bbox(bbox, ax)
    
def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, b, color='white'):
    """b : (min_x, min_y), h, w """
    patch = ax.add_patch(Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14, color='white'):
    text = ax.text(*xy, txt,
        verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)

def plot_detect_pred(img, pred_clas, pred_bbox, clas, bbox, anchors,
                     anc_sizes, sz, cls_idx=0, thresh=0.5, do_nms=True, figsize=(20,20)):
    # activation to bbox corner
    a_ic = actn_to_bb(pred_bbox, anchors, anc_sizes)
    # copy bbox
    actual_bb = to_np(bbox).copy()
    # get probabilities
    pred_proba = to_np(F.softmax(pred_clas))[:, cls_idx]
    # get thresholded predictions
    pred_bbox_mask = pred_proba > thresh
    pred_bbox_probas = np.round(pred_proba[pred_bbox_mask], 2)
    pred_bbox_corner = to_np(a_ic*sz).astype(int)[pred_bbox_mask]
    # do non max suppresion
    if do_nms:
        # remaining boxes
        post_nms_idxs = []
        # non max supression
        sorted_idx = np.argsort(pred_bbox_probas)[::-1]
        post_nms_idxs = nms(sorted_idx, pred_bbox_corner)
        pred_bbox_corner = pred_bbox_corner[post_nms_idxs]
        pred_bbox_probas = pred_bbox_probas[post_nms_idxs]
    # actual bbox
    actual_bbox = actual_bb*sz
    # plot
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    for i, (bbox, proba) in enumerate(zip(pred_bbox_corner, pred_bbox_probas)): 
        draw_bbox(bbox, axes)
        draw_text(axes, bbox[:2][::-1], f"pred_{i}")
        draw_text(axes, bbox[:2][::-1], f"proba:{proba}")

    for i, bbox in enumerate(actual_bbox):
        draw_bbox(bbox, axes, "white")
        draw_text(axes, bbox[:2][::-1], f"gt_{i}")
    axes.imshow(img)
