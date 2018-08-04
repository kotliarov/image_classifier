import argparse
import os
import json

from model_api import NeuralNetClassifier
from data_api import CheckpointStore

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

    args_source.add_argument('--checkpoint',
                             dest='checkpoint_path', 
                             type=arg_as_filepath,
                             required=False, 
                             help='path to a file with model\'s checkpoint, or reference to a model')

    args_parser.add_argument('--top-k', 
                                dest='top_k', 
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
                             default={},
                             help='path to a file with class-to-label dictionary, json')

    args = args_parser.parse_args()
    make_prediction(args)
    return 0

def make_prediction(args):
    """
    """
    checkpoint = CheckpointStore.read(args.checkpoint_path)
    if 'reference' == checkpoint['tag']:
        checkpoint = CheckpointStore.read(checkpoint['model'])

    clf = NeuralNetModel(checkpoint=checkpoint, use_gpu=args.use_gpu)
    probs, classes = clf.predict(args.image_path, args.top_k)

    print("Predicted class(es) for {}".format(args.image_path))
    for i, (p, k) in enumerate(zip(probs, classes)):
        print("{}. class={} prob={:.04f} label={}".format(i+1, 
                                                          k,
                                                          p,
                                                          args.class_to_name[k] if args.class_to_name else '-')) 

if __name__ == '__main__':
    exit(main())
