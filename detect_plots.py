from matplotlib.patches import Rectangle
from matplotlib import patheffects

def draw_bbox(bbox, ax, col='red'):
    """min row, min col, max row, max col - like np """
    min_y, min_x, max_y, max_x = bbox
    ax.add_patch(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
     fill=False, color=col))

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