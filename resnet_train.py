

import sys
import shutil
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


sys.path.append("../fastai/old")
from fastai.conv_learner import *


path = Path("../DATA/airbus-ship/")
files = list(path.iterdir())


from fastai.models.unet import *
from fastai.dataset import *
from fastai.core import *

train_seg_lbs = pd.read_csv(path/"segmentation/trn_segmentation.csv")
test_seg_lbs = pd.read_csv(path/"segmentation/test_segmentation.csv")


unique_img_ids = train_seg_lbs.ImageId.unique()
trn_fnames, val_fnames = train_test_split(unique_img_ids, test_size=0.1, random_state=42)
test_fnames = test_seg_lbs.ImageId.unique()

TRN_X = [f"train/{fname}" for fname in trn_fnames]
TRN_Y = [f"segmentation/train_masks/{fname}.npy" for fname in trn_fnames]

VAL_X = [f"train/{fname}" for fname in val_fnames]
VAL_Y = [f"segmentation/train_masks/{fname}.npy" for fname in val_fnames]

TEST_X = [f"test/{fname}" for fname in test_fnames]
TEST_Y = [f"segmentation/test_masks/{fname}.npy" for fname in test_fnames]

test_sub_fnames = list((path/"test_v2").glob("*.jpg"))
TEST_SUB_X = [f"test_v2/{fname.name}" for fname in test_sub_fnames]

class FilesEncodedDataset(BaseDataset):
    def __init__(self, fnames, fnames2, transform, path):
        self.fnames = fnames
        self.fnames2 = fnames2
        self.path = path
        super().__init__(transform)
    
    def get_sz(self): return self.transform.sz
    def get_x(self, i): return open_image(os.path.join(self.path, self.fnames[i]))
    def get_y(self, i): 
        
        mask = np.load(os.path.join(self.path, self.fnames2[i])).astype('float32')
        #mask = cv2.resize(mask, (sz, sz)).astype('float32')
        #mask = np.round(mask)
        return mask
        
    def get_n(self): return len(self.fnames)
    def get_c(self): return 0

    def resize_imgs(self, targ, new_path):
        dest = resize_imgs(self.fnames, targ, self.path, new_path)
        return self.__class__(self.fnames, self.y, self.transform, dest)

    def denorm(self,arr):
        """Reverse the normalization done to a batch of images.

        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        return self.transform.denorm(np.rollaxis(arr,1,4))
    
    
class Transform():
    """ A class that represents a transform.

    All other transforms should subclass it.
    All subclasses should override
    do_transform.

    Arguments
    ---------
        tfm_y : TfmType
            type of transform
    """
    def __init__(self, tfm_y=TfmType.NO):
        self.tfm_y=tfm_y
        self.store = threading.local()

    def set_state(self): pass
    def __call__(self, x, y):
        self.set_state()
        x,y = ((self.transform(x),y) if self.tfm_y==TfmType.NO
                else self.transform(x,y) if self.tfm_y in (TfmType.PIXEL, TfmType.CLASS)
                else self.transform_coord(x,y))
        return x, y

    def transform_coord(self, x, y): return self.transform(x),y

    def transform(self, x, y=None):
        x = self.do_transform(x,False)
        return (x, self.do_transform(y,True)) if y is not None else x

    @abstractmethod
    def do_transform(self, x, is_y): raise NotImplementedError
        
        
        
class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b,self.c = b,c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1/(c-1) if c<0 else c+1
        x = lighting(x, b, c)
        return x
    
class RandomDihedral(CoordTransform):
    """
    Rotates images by random multiples of 90 degrees and/or reflection.
    Please reference D8(dihedral group of order eight), the group of all symmetries of the square.
    """
    def set_state(self):
        self.store.rot_times = random.randint(0,3)
        self.store.do_flip = random.random()<0.5

    def do_transform(self, x, is_y):
        x = np.rot90(x, self.store.rot_times)
        return np.fliplr(x).copy() if self.store.do_flip else x
    
def rotate_cv(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    """ Rotate an image by deg degrees

    Arguments:
        deg (float): degree to rotate.
    """
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c//2,r//2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

class RandomRotate(CoordTransform):
    """ Rotates images and (optionally) target y.

    Rotating coordinates is treated differently for x and y on this
    transform.
     Arguments:
        deg (float): degree to rotate.
        p (float): probability of rotation
        mode: type of border
        tfm_y (TfmType): type of y transform
    """
    def __init__(self, deg, p=0.75, mode=cv2.BORDER_REFLECT, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.deg,self.p = deg,p
        if tfm_y == TfmType.COORD or tfm_y == TfmType.CLASS:
            self.modes = (mode,cv2.BORDER_CONSTANT)
        else:
            self.modes = (mode,mode)

    def set_state(self):
        self.store.rdeg = rand0(self.deg)
        self.store.rp = random.random()<self.p

    def do_transform(self, x, is_y):
        if self.store.rp: x = rotate_cv(x, self.store.rdeg, 
                mode= self.modes[1] if is_y else self.modes[0],
                interpolation=cv2.INTER_NEAREST)
        return x
    
    
def zoom_cv(x,z):
    """ Zoom the center of image x by a factor of z+1 while retaining the original image size and proportion. """
    if z==0: return x
    r,c,*_ = x.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),0,z+1.)
    return cv2.warpAffine(x,M,(c,r), borderMode=cv2.BORDER_CONSTANT, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_NEAREST)

class RandomZoom(CoordTransform):
    def __init__(self, zoom_max, zoom_min=0, mode=cv2.BORDER_REFLECT, tfm_y=TfmType.NO, p=1):
        super().__init__(tfm_y)
        self.zoom_max, self.zoom_min = zoom_max, zoom_min
        self.p = p

    def set_state(self):
        self.store.zoom = self.zoom_min+(self.zoom_max-self.zoom_min)*random.random()
        self.store.rp = random.random()<self.p
        
    def do_transform(self, x, is_y):
        if self.store.rp:
            x = zoom_cv(x, self.store.zoom)
        return x
    
#RandomRotate(deg=30, p=0.7, tfm_y=TfmType.PIXEL)
f = vgg16
sz = 768
tfms = tfms_from_model(f,
                       sz,
                       aug_tfms=[
                                 RandomRotate(20, p=0.2, mode=cv2.BORDER_REFLECT, tfm_y=TfmType.PIXEL),
                       
                                 RandomDihedral(tfm_y=TfmType.PIXEL),
                           
                                 RandomZoom(zoom_max=1.5, zoom_min=0, mode=cv2.BORDER_CONSTANT,
                                            tfm_y=TfmType.PIXEL, p=0.2),
                                 
                                 RandomBlur(blur_strengths=3, probability=0.2, tfm_y=TfmType.NO),
                                 
                                 RandomLighting(0.05, 0.05)],
                       
                       tfm_y=TfmType.PIXEL,
                       norm_y=False,
                       crop_type=CropType.NO) 


dataset = ImageData.get_ds(FilesEncodedDataset, 
                           trn=(TRN_X, TRN_Y),
                           val=(VAL_X, VAL_Y), tfms=tfms,
                           test=(TEST_X, TEST_Y) ,path=path)


md = ImageData(path, dataset, bs=16, num_workers=8, classes=None)

# load defined model# load  
def get_encoder(f, cut):
    base_model = (cut_model(f(True), cut))
    return nn.Sequential(*base_model)

def get_model(f=resnet18, sz=128):
    """gets dynamic unet model"""
    # cut encoder
    cut, cut_lr = model_meta[f]

    # define encoder
    encoder = get_encoder(f, cut)

    # init model
    # binary: ship - not ship
    m = DynamicUnet(encoder, n_classes=1) 

    # init upsample on cpu
    inp = torch.ones(1, 3, sz, sz)
    out = m(V(inp).cpu())

    # put model to gpu if desired# put mo 
    m = m.cuda(0)
    return m

def dice_loss(logits, target):
    logits = torch.sigmoid(logits)
    smooth = 1.0

    iflat = logits.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, logits, target):
        logits = logits.squeeze(1)
        probas = torch.sigmoid(logits)
        pt = (target)*probas + (1 - target)*(1 - probas)
        loss = (-(1 - pt)**gamma)*torch.log(pt)
        return loss.mean()
    
    
class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, logits, target):
        logits = logits.squeeze(1)
        if not (target.size() == logits.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), logits.size()))

        max_val = (-logits).clamp(min=0)
        loss = logits - logits * target + max_val + \
            ((-max_val).exp() + (-logits - max_val).exp()).log()

        invprobs = F.logsigmoid(-logits * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()    
    
    
class BCELoss2D(nn.Module):
    def __init__(self):
        super(BCELoss2D, self).__init__()
        
    def forward(self, logits, targets):
        logits = logits.squeeze(1)
        logits = F.sigmoid(logits)
        return F.binary_cross_entropy(logits, targets)
    
    
    
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        
    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()
    
class UpsampleModel():
    def __init__(self, model, cut_lr, name='upsample'):
        self.model,self.name, self.cut_lr = model, name, cut_lr

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.encoder), [self.cut_lr]))
        return lgs + [children(self.model)[1:]]
 
from skimage.measure import label
from eval_metric import sigmoid, get_gt_masks, create_iou_matrix, f2_IOU, get_pred_masks



def single_image_score(labels, gt_rles, gt):
    """
    return avg thresholded f2 score for single image
    labels : labeled image array
    gt_rles : array of rles
    """
    if len(np.unique(labels)) == 1:
        if gt is None: 
            """original image has no instance"""
            return 1
        else:
            """no prediction is made tp = 0"""
            return 0
    else:
        pred_mask_arrays = get_pred_masks(labels)
        gt_mask_arrays = get_gt_masks(gt_rles)
        IOU = create_iou_matrix(pred_mask_arrays, gt_mask_arrays)
        return f2_IOU(IOU)
    
    
shift = 0 # shift to keep track of file index 
n_valids = len(md.val_ds) # total # of validation samples
fnames = md.val_ds.fnames # validation filenames
df = train_seg_lbs

def fastai_metric(preds, targs):
    global shift
    global df
    global fnames
    global n_valids
    
    mask_thresh = 0.5 
    n_x = len(preds)

    scores = [] 
    preds = (sigmoid(to_np(preds).squeeze(1)) > mask_thresh).astype('uint8')
    gts = to_np(targs)
    
    for i, (gt_i, pred_i) in enumerate(zip(gts, preds)):
        fname = fnames[i+shift]
        gt_rles = df[df.ImageId == fname.split("/")[-1]]['EncodedPixels'].values
        labels = label(pred_i)
        scores.append(single_image_score(labels, gt_rles, gt_i))

    shift += n_x
    if shift == n_valids: shift = 0
    return np.mean(scores)



init_model = False
f = resnet18
cut, cut_lr = model_meta[f]
model = get_model(f, sz=768)
models = UpsampleModel(model, cut_lr)

if init_model:
    cls_weights = torch.load(path/"models/resnet34_classification_ft_v2.224.h5")
    state_dict_keys = list(model.encoder.state_dict().keys())
    for k in state_dict_keys: model.encoder.state_dict()[k].copy_(cls_weights[k])
        
        
learn = ConvLearner(md, models)
learn.opt_fn=optim.Adam
learn.crit = MixedLoss(10, 2)
learn.metrics = [fastai_metric]


print("load model")
learn.load("resnet18_segmentation_v7.768")
learn.fit(3e-3, n_cycle=1, cycle_len=10, use_clr=(20, 10))
learn.save("resnet18_segmentation_v8.768")









        
        
        
        




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    












































    
    
    
    
    
    
    
    
    



    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
































