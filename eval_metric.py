import numpy as np
import cv2
from rle import rle_decode
from skimage.measure import label

# ref: https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric

def sigmoid(x): return 1  /(1 + np.exp(-x))

def get_pred_masks(labels, sz=(768, 768)):
    """
    labels: np.array with label indexes, e.g. output from label(X)
        Assuming all images are 768x768 originally,
    get independent instance mask predictions from a mask image output
    """
    uniq_labels = np.unique(labels)
    pred_mask_arrays = []
    for l in uniq_labels[1:]:
        label = (labels == l).astype(np.uint8)
        if (sz is not None) and (label.shape[:2] != sz):
            pred_mask = cv2.resize(label, sz)
            pred_mask_arrays.append(pred_mask)
        else:
            pred_mask = label
            pred_mask_arrays.append(pred_mask)
    return pred_mask_arrays

def get_gt_masks(gt_rles, sz=(768, 768)):
    """
    gt_rles: ground truth run length encodings, list-np.array
        Assuming all images are 768x768 originally,
    decode each ground truth label independently
    """
    gt_mask_arrays = []
    for gt_rle in gt_rles:
        gt_mask = rle_decode(gt_rle, sz) 
        gt_mask_arrays.append(gt_mask)
    return gt_mask_arrays

def create_iou_matrix(pred_mask_arrays, gt_mask_arrays, print_matrix=False):
    """Create IOU matrix preds vs actual"""
    IOU = []
    for pred_mask in pred_mask_arrays:
        for gt_mask in gt_mask_arrays:
            intersection = np.sum(pred_mask*gt_mask)
            union = np.sum((pred_mask+gt_mask)>0)
            IOU.append(intersection/union)
    IOU = np.array(IOU)
    if print_matrix:
        print(f"""
            IOU matrix
            {IOU}
            """)
    return IOU.reshape(len(pred_mask_arrays), len(gt_mask_arrays))

def f2_score(tp, fp, fn, beta=2): return (1+beta**2)*tp / ((1+beta**2)*tp + (beta**2)*fn + fp)

def f2_IOU(IOU):
    """
    Calculate mean thresholded f2 score given IOU matrix,
    preds vs actual.
    calculates metric from IOU matrix
    """
    avg_f2 = 0
    for t in np.arange(0.5, 1.0, 0.05):
        IOU_at_t = (IOU > t)*1;# print(f"IOU at {t}: {IOU_at_t}")
        tp = IOU_at_t.sum(); #print(f"tp at {t}: {tp}")
        fp = np.sum(np.sum(IOU_at_t, axis=1) == 0); #print(f"fp at {t}: {fp}")
        fn = np.sum(np.sum(IOU_at_t, axis=0) == 0); #print(f"fn at {t}: {fn}")
        f2 = f2_score(tp, fp, fn)
        avg_f2 += f2
    return avg_f2 / 10

def single_image_score(labels, gt_rles):
    """return avg thresholded f2 score for single image"""
    if len(np.unique(labels)) == 0:
        if gt_rles is None: 
            return 1
        else: return 0 
    else:
        pred_mask_arrays = get_pred_masks(labels)
        gt_mask_arrays = get_gt_masks(gt_rles)
        IOU = create_iou_matrix(pred_mask_arrays, gt_mask_arrays)
        return f2_IOU(IOU)
    
def single_detect_image_score(labels, gt_rles):
    """return avg thresholded f2 score for single image"""
    if len(np.unique(labels)) == 0:
        if gt_rles is None: 
            return 1
        else: return 0 
    else:
        pred_mask_arrays = get_pred_masks(labels)
        gt_mask_arrays = get_gt_masks(gt_rles)
        IOU = create_iou_matrix(pred_mask_arrays, gt_mask_arrays)
        return f2_IOU(IOU)