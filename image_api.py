import numpy as np
from PIL import Image

_CROP_SIZE = (224, 224)
_IMAGE_MEANS = [0.485, 0.456, 0.406]
_IMAGE_STDDEVS = [0.229, 0.224, 0.225]

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
    dx = (width - _CROP_SIZE[0]) // 2
    dy = (height - _CROP_SIZE[1]) // 2
    left, right = dx, width - dx
    top, bottom = dy, height - dy
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image, dtype=np.float32)

    np_image /= 255.
    np_image -= _IMAGE_MEANS
    np_image /= _IMAGE_STDDEVS
    np_image = np_image.transpose((2, 0, 1))

    return np_image


