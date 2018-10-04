import torch
import torch.nn.functional as F
from anchor_box import hw2corners 

def get_y(bbox,clas,sz):
    bbox = bbox.view(-1,4)/sz
    bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
    return bbox[bb_keep],clas[bb_keep]

def actn_to_bb(actn, anchors, anc_sizes):
    actn_bbs = torch.tanh(actn) 
    actn_centers = (actn_bbs[:,:2] * anc_sizes) + anchors[:,:2] 
    actn_hw = torch.exp(actn_bbs[:,2:]) * anc_sizes
    return hw2corners(actn_centers, actn_hw)

def intersect(box_a, box_b):
    # box_i -> (min row, min col, max row, max col)
    max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:]) 
    min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2]) 
    inter = torch.clamp((max_xy - min_xy), min=0) 
    return inter[:, :, 0] * inter[:, :, 1]

def box_sz(b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    union = box_sz(box_a).unsqueeze(1) + box_sz(box_b).unsqueeze(0) - inter
    return inter / union

def map_to_ground_truth(overlaps, print_it=False):
    prior_overlap, prior_idx = overlaps.max(1)
    if print_it: print(prior_overlap)
    gt_overlap, gt_idx = overlaps.max(0)
    gt_overlap[prior_idx] = 1.99
    for i,o in enumerate(prior_idx): gt_idx[o] = i #?
    return gt_overlap, gt_idx

def ssd_1_loss(b_c, b_bb, bbox, clas, sz, anchors, anchor_cnr, anc_sizes, loss_f, n_cls=1, print_it=False):
    bbox,clas = get_y(bbox,clas, sz)
    a_ic = actn_to_bb(b_bb, anchors, anc_sizes)
    overlaps = jaccard(bbox.data, anchor_cnr.data)
    gt_overlap, gt_idx = map_to_ground_truth(overlaps,print_it)
    gt_clas = clas[gt_idx]
    pos = gt_overlap > 0.5
    pos_idx = torch.nonzero(pos)[:,0]
    gt_clas[1-pos] = n_cls # set idx for background class
    gt_bbox = bbox[gt_idx] 
    loc_loss = F.smooth_l1_loss(a_ic[pos_idx], gt_bbox[pos_idx])
    clas_loss  = loss_f(b_c, gt_clas) 
    return loc_loss, clas_loss