import matplotlib.pyplot as plt
import torch 
from torch.autograd import Variable
import numpy as np

def plot_anchors(anc_x, anc_y):
    plt.scatter(anc_x, anc_y)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

def hw2corners(ctr, hw): 
    return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)

def get_anchor_boxes(anc_grids, anc_zooms, anc_ratios):
    """ 
    anc_grids : number of square grids from feature map
    anc_zooms : how much relative zoom to grid h-w
    anc_ratios : aspect ratio for different shaped images

    order:
        x1: min_x, y1 : min_y

        x1, y1, k1
        x1, y1, k2
        ...
        x1, y1, kn
        x1, y2, k1
        ...
        xn, yn, kn
    """
    # scale of each anchor
    anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
    k = len(anchor_scales)
    # offset between anchors
    anc_offsets = [1/(o*2) for o in anc_grids]
    # center x coordinates, complete each grid first
    anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                        for ao,ag in zip(anc_offsets, anc_grids)])
    # center y coordinates, complete each grid first
    anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                        for ao,ag in zip(anc_offsets,anc_grids)])
    # anchor centers
    # repeat all center k times 
    anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)

    # anchor scales = aspect ratio * zoom
    # height and width, given x = row and y = column
    # same order as anc_ctrs
    anc_sizes = np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                for ag in anc_grids])

    # for each anchor grid and for each anchor scale create grid size 
    # fix space between two adj anchor centers
    grid_sizes = Variable(torch.FloatTensor(np.concatenate([np.array([ 1/ag for i in range(ag*ag)
                        for o,p in anchor_scales])
                        for ag in anc_grids])), 
                        requires_grad=False).unsqueeze(1)

    # create anchors from xy center and height-width
    anchors = Variable( torch.FloatTensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)), requires_grad=False).float()
    # transform to minx, miny, maxx, maxy format
    anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])
    # torch anchor sizes
    anc_sizes = Variable(torch.FloatTensor(anc_sizes))

    if torch.cuda.is_available(): anchors, grid_sizes, anchor_cnr, anc_sizes = anchors.cuda(), grid_sizes.cuda(), anchor_cnr.cuda(), anc_sizes.cuda()
    return anchors, grid_sizes, anchor_cnr, anchor_scales, anc_sizes


"""
## IMPORTANT FOR CORRECT RECEPTIVE FIELD MATCHING ##

anchors orders

x = row, y = column, h = height, w = width

grid1 - n1 = 12

total = 12*12*9(3 zoom * 3 aspect ratio)

x1, y1, 0.7, 0.7
x1, y1, 0.7, 0.35
x1, y1, 0.35, 0.7
x1, y1, 1.0, 1.0
x1, y1, 1.0, 0.5
x1, y1, 0.5, 1.0
x1, y1, 1.3, 1.3
x1, y1, 1.3, 0.6
x1, y1, 0.65, 1.3

x1, y2, 0.7, 0.7
x1, y2, 0.7, 0.35
x1, y2, 0.35, 0.7
x1, y2, 1.0, 1.0
x1, y2, 1.0, 0.5
x1, y2, 0.5, 1.0
x1, y2, 1.3, 1.3
x1, y2, 1.3, 0.6
x1, y2, 0.65, 1.3

xn1, yn1, 0.7, 0.7
xn1, yn1, 0.7, 0.35
xn1, yn1, 0.35, 0.7
xn1, yn1, 1.0, 1.0
xn1, yn1, 1.0, 0.5
xn1, yn1, 0.5, 1.0
xn1, yn1, 1.3, 1.3
xn1, yn1, 1.3, 0.6
xn1, yn1, 0.65, 1.3

grid2 - n2 = 18

total = 18*18*9(3 zoom * 3 aspect ratio)


x1, y1, 0.7, 0.7
x1, y1, 0.7, 0.35
x1, y1, 0.35, 0.7
x1, y1, 1.0, 1.0
x1, y1, 1.0, 0.5
x1, y1, 0.5, 1.0
x1, y1, 1.3, 1.3
x1, y1, 1.3, 0.6
x1, y1, 0.65, 1.3

x1, y2, 0.7, 0.7
x1, y2, 0.7, 0.35
x1, y2, 0.35, 0.7
x1, y2, 1.0, 1.0
x1, y2, 1.0, 0.5
x1, y2, 0.5, 1.0
x1, y2, 1.3, 1.3
x1, y2, 1.3, 0.6
x1, y2, 0.65, 1.3

xn2, yn2, 0.7, 0.7
xn2, yn2, 0.7, 0.35
xn2, yn2, 0.35, 0.7
xn2, yn2, 1.0, 1.0
xn2, yn2, 1.0, 0.5
xn2, yn2, 0.5, 1.0
xn2, yn2, 1.3, 1.3
xn2, yn2, 1.3, 0.6
xn2, yn2, 0.65, 1.3

TOTAL = (12**2 + 18**2)*9 = 4212
"""