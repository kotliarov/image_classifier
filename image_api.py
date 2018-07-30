import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def make_image(path):
    """
        Return numpy image pre-processed for use 
        in a PyTorch model.
    """
    image = Image.open(path)

    # Resize image, keeping aspect ratio, such that 
    # smaller side of the image is set to 256 px.
    width, height = image.size
    if width > height:
        ratio = width / height)
        new_w, new_h = (int(0.5 + 256*ratio), 256)
    else:
        ratio = height / width
        new_w, new_h = (256, int(0.5 + 256 * ratio))
    image.thumbnail((new_w, new_h))
    width, height = image.size

    # Crop center of the image
    crop_width, crop_height = get_crop_size()
    dx = (width - crop_width) // 2
    dy = (height - crop_height) // 2
    left, right = dx, width - dx
    top, bottom = dy, height - dy
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image, dtype=np.float32)

    np_image /= 255.
    np_image -= get_image_means()
    np_image /= get_image_std()
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def imshow(image, ax=None, title=None):
    """Display image.        
        image ::= image as a tensor.
        ax    ::= matplotlib axes object.
        title ::= image title.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = np.array(get_image_std()) * image + np.array(get_image_means())
    # Image needs to be clipped between 0 and 1 
    # or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    if title is not None:
        ax.set_title(title)
    ax.imshow(image)
    return ax

def get_crop_size():
    """Return crop size tuple (width, height)
    """
    return (224, 224)

def get_image_means():
    """Return image statistical prop: 
        mean values per channel.
    """
    return [0.485, 0.456, 0.406]

def get_image_std():
    """Return image statistical prop: 
        deviation values per channel.
    """
    return [0.229, 0.224, 0.225]

