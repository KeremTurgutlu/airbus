# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import sys
import shutil
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

sys.path.append("../fastai/old/")
from fastai.conv_learner import *

path = Path("../DATA/airbus-ship/")
files = list(path.iterdir())

f_model=resnet34
sz= 768

aug_tfms=[RandomDihedral(tfm_y=TfmType.COORD),
            RandomLighting(0.05, 0.05, tfm_y=TfmType.COORD)]
aug_tfms = []

tfms = tfms_from_model(f_model, sz,
                       crop_type=CropType.NO,
                       tfm_y=TfmType.COORD,
                       aug_tfms=aug_tfms)

md = ImageClassifierData.from_csv(path,
                                  "train",
                                  path/"detection/train_bbox_lbs2.csv",
                                  tfms=tfms, continuous=True, bs=16, test_name="test_detection")

denorm = md.trn_ds.denorm


class ConcatLblDataset(Dataset):
    def __init__(self, ds, mcls):
        self.ds = ds
        self.sz = ds.sz
        self.mcls = mcls
        
    def __len__(self): return len(self.ds)
    
    def __getitem__(self, i):
        x, y = self.ds[i]
        y2 = self.mcls[i]
        return (x, (y, y2))
    
train_bbox_lbs = pd.read_csv(path/"detection/train_bbox_lbs2.csv")
mcls = train_bbox_lbs.BoundingBox.apply(lambda x: np.array([0]*(len(x.split()) //4)))

trn_ds2 = ConcatLblDataset(md.trn_ds, mcls)
val_ds2 = ConcatLblDataset(md.val_ds, mcls)
md.trn_dl.dataset = trn_ds2
md.val_dl.dataset = val_ds2

from anchor_box import plot_anchors, hw2corners, get_anchor_boxes
from detect_plots import draw_bbox


anc_grids = [12]
anc_zooms = [0.5, 1., 2]
anc_ratios = [(1.,1.), (1.,0.5), (0.5,1.)]

anchors, grid_sizes, anchor_cnr, scales, anc_sizes = get_anchor_boxes(anc_grids, anc_zooms, anc_ratios)

k = len(scales)

# number of classes excluding background
n_cls = 1

class StdConv(nn.Module):
    """conv - relu - batchnorm - dropout"""
    def __init__(self, nin, nout, stride=2, drop=0.1):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(drop)
    def forward(self, x): return self.drop(self.bn(F.relu(self.conv(x))))
        
def flatten_conv(x,k):
    bs,nf,gx,gy = x.size()
    x = x.permute(0,2,3,1).contiguous()
    return x.view(bs,-1,nf//k)


class OutConv(nn.Module):
    def __init__(self, k, nin, bias, n_cls=1):
        super().__init__()
        self.k = k
        self.oconv1 = nn.Conv2d(nin, (n_cls+1)*k, 3, padding=1) 
        self.oconv2 = nn.Conv2d(nin, 4*k, 3, padding=1) 
        self.oconv1.bias.data.zero_().add_(bias)
        
    def forward(self, x):
        return [flatten_conv(self.oconv1(x), self.k),
                flatten_conv(self.oconv2(x), self.k)]
    
class SSD_Head(nn.Module):
    def __init__(self, k, bias):
        """
        SSD adds 6 more auxiliary convolution layers after the VGG16. 
        Five of them will be added for object detection.
        In three of those layers, we make 6 predictions instead of 4.
        In total, SSD makes 8732 predictions using 6 layers.
        
        """
        super().__init__()
        self.drop = nn.Dropout(0.25)
        self.sconv0 = StdConv(512,256, stride=1)
        self.sconv2 = StdConv(256,256)
        self.out = OutConv(k, 256, bias)
        
    def forward(self, x):
        x = self.drop(F.relu(x))
        x = self.sconv0(x)
        x = self.sconv2(x)
        return self.out(x)

head_reg4 = SSD_Head(k, -3.)
f_model=resnet34
sz=sz
models = ConvnetBuilder(f_model, c=0, is_multi=0, is_reg=0, custom_head=head_reg4)
learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam


from detect_utils import get_y, actn_to_bb
from detect_utils import jaccard
from detect_utils import map_to_ground_truth
from detect_utils import ssd_1_loss

def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()].cuda()

class BCE_Loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred, targ):
        t = one_hot_embedding(targ, self.num_classes+1)
        t = V(t[:,:-1].contiguous())
        x = pred[:,:-1]
        w = self.get_weight(x,t)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False) / self.num_classes
    
    def get_weight(self,x,t): return None

cls_loss = BCE_Loss(num_classes=n_cls)


def ssd_loss(pred, targ, alpha=1, print_it=False):
    lcs,lls = 0.,0.
    for b_c,b_bb,bbox,clas in zip(*pred,*targ):
        loc_loss,clas_loss = ssd_1_loss(b_c, b_bb, bbox, clas, sz, anchors.cpu(),
                                        anchor_cnr.cpu(), anc_sizes, cls_loss,
                                        n_cls=1, print_it=False)
        lls += loc_loss
        lcs += clas_loss
    if print_it: print(f'loc: {lls.data[0]}, clas: {lcs.data[0]}')
    return lls*alpha+lcs


model = learn.model.cpu()

for *x, y in md.trn_dl:
    out = model(V(x[0]).cpu())
    loss = ssd_loss(out.cpu(), V(y).cpu())
    print(loss)
    
    










