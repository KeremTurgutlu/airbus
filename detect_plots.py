from matplotlib.patches import Rectangle
from matplotlib import patheffects
import matplotlib.pyplot as plt

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