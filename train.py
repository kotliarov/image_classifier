import argparse
import os

import torch
import numpy as np

import model_api
import data_api

_MOMENTUM = 0.8
_REPORT_EVERY_N_ITER = 30

def train_with_variable_learning_rate(args):
    """
    """
    datasource = data_api.make_dataloaders(args.data_path, args.batch_size)
    checkpoint = data_api.make_checkpoint_store(args.checkpoint_dir)
    num_batches = len(iter(datasource.train)) 
    cycle_period = args.epochs_per_cycle * num_batches # cycle period
    print("n={} num-iter-per-epoch={} T={}".format(num_batches*args.batch_size, 
                                                   num_batches, 
                                                   cycle_period))
    device = model_api.make_device(args.use_gpu)
    model = model_api.make_model(model_api.pretrained_model_factory(args.arch_name),
                                 args.output_size,
                                 args.hidden_layers,
                                 args.dropout)                       
    model.to(device)
    
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.classifier.parameters(), 
                                lr=args.max_learning_rate, 
                                momentum=_MOMENTUM)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cycle_period)
    progress_tracker = model_api.ProgressTracker(model, 
                                                device, 
                                                datasource.validate, 
                                                criterion, 
                                                lr_scheduler, 
                                                _REPORT_EVERY_N_ITER)
    best_score = 0.
    best_model_ref = None
    learning_rate = max_learning_rate
    
    for epoch_no in range(1, args.epochs_per_cycle * args.num_cycles + 1):
        progress_tracker.epoch = epoch_no
        train_loss = train(model, 
                           device, 
                           datasource.train, 
                           criterion, 
                           optimizer, 
                           progress_tracker)
        # Make checkpoint at the end of a cycle.
        if epoch_no % args.epochs_per_cycle == 0:
           checkpoint.store("checkpoint_epoch_{}.pth".format(epoch_no), 
                            model, 
                            classifier = {"architecture": args.arch_name,
                                          "output-size": args.output_size,
                                          "hidden-layers": args.hidden_layers,
                                          "dropout": args.dropout},
                            epoch=epoch_no, 
                            report=progress_tracker.performance_report,
                            class_to_index=datasource.train.class_to_idx)

            _, valid_acc = validation_score(model, 
                                           device, 
                                           datasource.validate, 
                                           criterion)
            if valid_acc > best_score:
                best_score = valid_acc
                best_model_ref = checkpoint_filename
                
            # Reset learning rate to max_learning_rate * dumping_koeff
            learning_rate = learning_rate * args.dumping_koeff
            optimizer = torch.optim.SGD(model.classifier.parameters(), 
                                        lr=learning_rate, 
                                        momentum=_MOMENTUM)
            progress_tracker.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                                       cycle_period)

    # Persist reference to best model's checkpoint
    checkpoint.store_reference("best_model.pth", 
                                best_score, 
                                best_model_ref)
    return model, progress_tracker.performance_report

def train_with_const_learn_rate(args):
    pass

def main():
    """
        Application's main function.
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
                             dest='checkpoint_dir_path', 
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
                                  dest='max_learn_rate',
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
                                  dest='dumpig_koeff',
                                  type=float,
                                  default=1.0, 
                                  choices=[Range(0.1, 1.0)],
                                  help='koefficient to reduce max learning rate between cycles')
    var_learn_rate.set_defaults(func=train_with_variable_learning_rate)
 
    args = parser.parse_args()

    print(args)
    args.func(args)

    return 0

if __name__ == '__main__':
    exit(main())
