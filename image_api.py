import numpy as np
from PIL import Image

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

