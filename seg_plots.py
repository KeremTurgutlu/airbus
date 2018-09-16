import matplotlib.pyplot as plt

def show_imgmasks(img, masks, size=(10, 10)):
    """
    img: np.array
    mask: np.array
    show image, mask and image-mask together
    """
    _, axes = plt.subplots(1, 3, figsize=size)
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    axes[0].imshow(img)
    axes[1].imshow(masks)
    axes[2].imshow(img)
    axes[2].imshow(masks, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    #plt.show()