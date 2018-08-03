import argparse
import os

from model_api import NeuralNetClassifier
import data_api

def main():
    """ Application's main function.
    """
    class Range(object):
        def __init__(self, lower, upper):
            self.lower = lower
            self.upper = upper
        def __eq__(self, value):
            return self.lower <= value <= self.upper
        def __repr__(self):
            return "{:.02f} - {:.02f}".format(self.lower, self.upper)

    def arg_as_arch(arch):
        """ Verify that specified name is a name of pre-trained model.
        """
        try:
            model_api.pretrained_model_factory(arch)
        except KeyError:
            raise argparse.ArgumentTypeError('Unknown architecture: {}'.format(arch))
        return arch

    def arg_as_dir(path, create_if_not_exists=False):
        """ Verify that specified path is valid and refers to existing directory.
            Create specified directory if it does not exists.

            Return directory path.
        """
        path = os.path.abspath(path)
        if not os.path.exists(path):
            if not create_if_not_exists:
                raise argparse.ArgumentTypeError('Directory does not exist: {}'.format(path))
            else:
                os.makedirs(path)
        if not os.path.isdir(path):
            raise argparse.ArgumentTypeError('Not a directory: {}'.format(path))
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
 
    parser = argparse.ArgumentParser(description='Train neural network model to classify images.')

    args_parser = argparse.ArgumentParser(add_help=False)
    args_parser.add_argument('--data-path', 
                             dest='data_path', 
                             type=arg_as_dir,
                             required=True, 
                             help='path to a directory with data set')
    args_parser.add_argument('--checkpoint-path',
                             dest='checkpoint_path', 
                             type=lambda x: arg_as_dir(x, True),
                             required=True, 
                             help='path to a directory to store model checkpoints')
    args_parser.add_argument('--gpu', 
                             dest='use_gpu', 
                             action='store_true', 
                             default=False,
                             help='use GPU if available')
    args_parser.add_argument('--class-names',
                             dest='class_to_name',
                             type=arg_as_json,
                             help='path to a file with class-to-label dictionary, json')

    model = args_parser.add_argument_group('model')
    model.add_argument('--arch-name',
                       dest='arch_name',
                       type=arg_as_arch,
                       default='vgg19',
                       help='pre-trained model name')
    model.add_argument('--hidden-layers',
                       dest='hidden_layers',
                       nargs='+',
                       type=int,
                       required=True,
                       help='collection of hidden units size: --hidden-units 1024 2048 512')
    model.add_argument('--output-size',
                       dest='output_size',
                       type=int,
                       required=True,
                       help='output size; number of classes')
    model.add_argument('--dropout',
                       dest='dropout',
                       type=float,
                       default=0.3,
                       help='dropout probability of a neuron cell')
    model.add_argument('--batch-size',
                       dest='batch_size',
                       type=int,
                       default=96,
                       help='training sample batch size')

    sp = parser.add_subparsers()
    const_learn_rate = sp.add_parser('const-learn-rate', parents=[args_parser], help='Train NN with constant learning rate over epochs')
    const_learn_rate.add_argument('--learning-rate',
                            dest='learn_rate',
                            type=float,
                            required=True,
                            help='learning rate')
    const_learn_rate.add_argument('--epochs',
                            dest='epochs',
                            type=int,
                            required=True,
                            help='number of epochs to train model')
   
    const_learn_rate.set_defaults(func=train_with_const_learning_rate)
 
    var_learn_rate = sp.add_parser('adjust-learn-rate', parents=[args_parser], help='Train NN with variable  learning rate over epochs')
    var_learn_rate.add_argument('--max-learning-rate',
                                  dest='max_learning_rate',
                                  type=float,
                                  required=True,
                                  help='max learning rate')
    var_learn_rate.add_argument('--epochs-per-cycle',
                                  dest='epochs_per_cycle',
                                  type=int,
                                  required=True,
                                  help='number of epochs per cycle')
    var_learn_rate.add_argument('--num-cycles',
                                  dest='num_cycles',
                                  type=int,
                                  required=True,
                                  help='number of training cycles')
    var_learn_rate.add_argument('--dump-koeff',
                                  dest='dump_koeff',
                                  type=float,
                                  default=1.0, 
                                  choices=[Range(0.1, 1.0)],
                                  help='koefficient to reduce max learning rate between cycles')
    var_learn_rate.set_defaults(func=train_with_variable_learning_rate)
 
    args = parser.parse_args()

    print(args)
    args.func(args)

    return 0

def train_with_variable_learning_rate(args):
    """
    """
    nn_clf = NeuralNetClassifier(arch=args.arch_name,
                                hidden_layers=args.hidden_layers, 
                                output_size=args.output_size, 
                                dropout=args.dropout,
                                use_gpu=args.use_gpu)

    score, model_path, model_ref = nn_clf.fit(data_path=args.data_path,
                                     learning_rate_policy='cosine',
                                     learning_rate_init=args.max_learning_rate,
                                     num_cycles=args.num_cycles,
                                     epochs_per_cycle=args.epochs_per_cycle,
                                     dump_koeff=args.dump_koeff,
                                     batch_size=args.batch_size,
                                     checkpoint_dir=args.checkpoint_path)
    report_results(score, model_path, model_ref)

def train_with_const_learning_rate(args):
    """
    """
    nn_clf = NeuralNetClassifier(arch=args.arch_name,
                                hidden_layers=args.hidden_layers, 
                                output_size=args.output_size, 
                                dropout=args.dropout,
                                use_gpu=args.use_gpu)

    score, model_path, model_ref = nn_clf.fit(data_path=args.data_path,
                                       learning_rate_policy='constant',
                                       learning_rate_init=args.learning_rate,
                                       num_cycles=args.epochs,
                                       epochs_per_cycle=1,
                                       batch_size=args.batch_size,
                                       checkpoint_dir=args.checkpoint_path)
    report_result(score, model_path, model_ref)

def report_result(score, model_path, model_ref):
    """ Print model's training results.
    """
    print("Model path: {}".format(model_path))
    print("Model ref: {}".format(model_ref))
    print("Model accuracy, validation: {:0.3f}".format(score))

    acc = model_accuracy(model_path, args.data_path, args.use_gpu)
    print("Model accuracy, test: {:0.3f}".format(acc)

def model_accuracy(model_path, data_path, use_gpu):
    """ Compute model's accuracy on test data set.
    """
    checkpoint = data_api.load_model_descriptor(model_path)
    clf = NeuralNetClassifier(checkpoint=checkpoint,
                              use_gpu=use_gpu)
    acc = clf.accuracy(data_path)
    return acc

if __name__ == '__main__':
    exit(main())
