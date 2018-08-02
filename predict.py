import argparse
import os
import json

import torch
import numpy as np

from image_api import make_image, get_crop_size
from model_api import make_model_from_checkpoint, make_device
from data_api import Checkpoint

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
        if not os.path.isfile(path):
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

    # Model source could be
    # a) model's checkpoint file.
    # b) reference to a model's checkpoint file, 
    #    where reference is a checkpoint file that contains following object:
    #    { 
    #       checkpoint': <path-to-model-checkpoint>, 
    #       score: <model-score> 
    #    }
    model_source = args_parser.add_mutually_exclusive_group()
    
    model_source.add_argument('--checkpoint',
                             dest='checkpoint_path', 
                             type=arg_as_filepath,
                             required=False, 
                             help='path to a file with model checkpoint')

    model_source.add_argument('--checkpoint-ref',
                             dest='checkpoint_ref_path', 
                             type=arg_as_filepath,
                             required=False, 
                             help='path to a file with reference to model checkpoint')

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
    make_prediction(args)
    return 0

def make_prediction(args):
    """
    """
    checkpoint = Checkpoint.read(args.checkpoint_path)
    model = make_model_from_checkpoint(checkpoint)
    device = make_device(args.use_gpu)
    model.to(device)    

    index_to_class = {index: klass for klass, index in checkpoint['class_to_index'].items()}
    probs, classes = predict(args.image_path, model, device, index_to_class, args.topk)

    print("Predicted class(es) for {}".format(args.image_path))
    for i, (p, k) in enumerate(zip(probs, classes)):
        print("{}. class={} prob={:.04f} label={}".format(i+1, 
                                                          k,
                                                          p,
                                                          args.class_to_name[k])) 

def predict(image_path, model, device, index_to_class, topk=5):
    """
        Predict the class (or classes) of an image using a trained deep learning model.        
        Return pair (topk-probability, topk-category)
    """
    model.eval()
    with torch.no_grad():
        crop_width, crop_height = get_crop_size()
        image = torch.tensor(make_image(image_path))
        image = image.reshape((1, 3, crop_width, crop_height))
        image = image.to(device)
        y_hat = model.forward(image)
        estimate = torch.exp(y_hat)
    estimate = estimate.to('cpu').numpy()
    index = np.argsort(estimate, axis=1)[0][-topk:][::-1]
    probs = estimate[0][index]
    return probs, [index_to_class[x] for x in index]

if __name__ == '__main__':
    exit(main())
