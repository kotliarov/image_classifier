import argparse
import os
import json

import torch
import numpy as np

from image_api import make_image
from model_api import make_model_from_checkpoint

def predict(image_path, model, topk=5):
    """
        Predict the class (or classes) of an image using a trained deep learning model.        
        Return pair (topk-probability, topk-category)
    """
    model.eval()
    with torch.no_grad():
        image = torch.tensor(make_image(image_path))
        image = image.reshape((1, 3, 224, 224))
        image = image.to(model.device)
        y_hat = model.forward(image)
        estimate = torch.exp(y_hat)
    estimate = estimate.to('cpu').numpy()
    index = np.argsort(estimate, axis=1)[0][-topk:][::-1]
    probs = estimate[0][index]
    return probs, [model.mapper.class_by_index(x) for x in index]

def main():
    """
        Application's main function.
    """
    def arg_as_filepath(path):
        """ 
            Verify that specified path is valid and refers to a file.
            Return file path.
        """
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError('Not a valid path: {}'.format(path))
        if not os.isfile(path):
            raise argparse.ArgumentTypeError('Not a file: {}'.format(path))
        return path

    def arg_as_json(path):
        """ 
        Return object load from specified json file.
        """
        with open(path,'r') as src:
            try:
                obj = json.load(src)
            except json.JSONDecodeError as e:
                raise argparse.ArgumentTypeError('File {} contains invalid JSON syntax: {}'.format(path, 
                                                                                                   e.msg))
        return obj
 
    args_parser = argparse.ArgumentParser(description='Predict image class')

    args_parser.add_argument('--image', 
                             dest='image_path', 
                             type=arg_as_filepath,
                             required=True, 
                             help='path to an image file')

    args_parser.add_argument('--checkpoint',
                             dest='checkpoint_path', 
                             type=lambda x: arg_as_json(arg_as_filetype(x)),
                             required=True, 
                             help='path to a file with model checkpoint')

    args_parser.add_argument('--topk', 
                                dest='topk', 
                                type=int, 
                                default=5, 
                                help='top k most likely classes to return')

    args_parser.add_argument('--gpu', 
                                dest='use_gpu', 
                                action='store_true', 
                                default=False,
                                help='use GPU if available')

    args_parser.add_argument('--class-names',
                             dest='class_to_name',
                             type=arg_as_json,
                             help='path to a file with class-to-label dictionary, json')

    args = args_parser.parse_args()

    model = make_model_from_checkpoint(args.path_checkpoint, args.class_to_name, args.use_gpu)    
    probs, classes = predict(args.image_path, model, args.topk)

    print("Predicted class(es) for {}".format(args.image_path))
    for i, (p, k) in enumerate(zip(probs, classes)):
        print("{: 2}. class={: 3} prob={:.04f} label={}".format(i, 
                                                               k,
                                                               p
                                                               model.mapper.label_by_class(k))) 
    return 0

if __name__ == '__main__':
    exit(main())
