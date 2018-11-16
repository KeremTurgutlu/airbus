from fastai.dataset import open_image
from fastai.core import to_np
import matplotlib.pyplot as plt
from rle import rle_encode, rle_decode
import numpy as np
import torch
from torch.autograd import Variable 
import cv2
from eval_metric import sigmoid

def show_img_masks(img, masks, size=(10, 10), axes=None):
    """
    show image, mask and image-mask together
        
        img: np.array
        mask: np.array
    """
    if axes is None: _, axes = plt.subplots(1, 3, figsize=size)
    for ax in axes: ax.axis('off')
    axes[0].imshow(img)
    axes[1].imshow(masks)
    axes[2].imshow(img)
    axes[2].imshow(masks, alpha=0.4)

def plot_segmentation_df(df, path, n=100, size=(20, 20), verbose=True):
    """
    randomly plot segmentation data
    
        df : dataframe with columns [image id, encoded pixels]
        path : image path 
    """
    img_ids = np.unique(df.iloc[:, 0])
    img_ids = np.random.choice(img_ids, n, replace=False)
    for i, img_id in enumerate(img_ids):        
        rles = df[df.iloc[:, 0] == img_id].iloc[:,1].values
        img = open_image(path/img_id)
        masks = sum([rle_decode(rle, img.shape[:2]) for rle in rles])
        show_img_masks(img, masks, size=size, axes=None)
    plt.close()

def plot_batch(path, model, dl, fnames, n=5):
    """
    Plot first n batch predictions
        path : data path
        model : model 
        dl: dataloader
        fnames: filenames like path/fnames[i]
    """
    for i, (*x, y) in enumerate(dl):
        bs = len(y)
        out = model(Variable(*x))
        out_np = to_np(out)
        batch_fnames = fnames[bs*i:bs*(i+1)]
        for fname, out in zip(batch_fnames, out_np):
            im = open_image(path/fname)
            mask = (sigmoid(out.squeeze(0)) > 0.5).astype("uint8")
            mask = cv2.resize(mask, (768, 768))
            show_img_masks(im, mask)
        plt.close()
        if i+1 == n: break
                    
                    
                    
                    