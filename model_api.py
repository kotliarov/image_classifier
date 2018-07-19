import numpy as np
import torch
import torchvision as tv

def make_model_from_checkpoint(path, class_to_name, use_gpu=True):
    """
        Return trained model from checkpoint file.
    """
    # Read checkpoint file and re-map storage
    # to lowest common denominator - 'cpu'.
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    clf_descriptor = checkpoint['classifier-descriptor']
    # Read architecture name.
    try:
        arch = checkpoint['arch']
    except KeyError:
        arch = 'vgg19'
    
    model = make_model(pretrained_model_factory(arch),
                       clf_descriptor['output-size'],
                       clf_descriptor['hidden-layers'],
                       clf_descriptor['dropout'])
    model.classifier.load_state_dict(checkpoint['classifier.state_dict'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if use_gpu 
                                                                            else "cpu"
    model.to(device)
    model.device = device
    return model

def make_model(pretrained_model_factory, output_size, hidden_layers, drop_prob=0.3):
    """
    Return model with classifier for the problem domain.
    """
    def make_classifier_net(input_size, output_size, hidden_layers, dropout=0.3):
        """
        Return instance of torch.nn.Sequential class
        """
        layers = [input_size] + hidden_layers + [output_size]
        input_output_size = [] 
        for x, y in zip(layers[:-1], layers[1:]):
            input_output_size.append((x, y))

        modules = []
        for i, (inp_size, out_size) in enumerate(input_output_size[:-1]):
            modules.append(("fc{}".format(i), torch.nn.Linear(inp_size, out_size)))
            modules.append(("relu{}".format(i), torch.nn.ReLU()))
            modules.append(("dropout{}".format(i), torch.nn.Dropout(dropout)))
        # Add output layer
        inp_size, out_size = input_output_size[-1]
        modules.append(("fc-output", torch.nn.Linear(inp_size, out_size)))
        modules.append(("logsoftmax", torch.nn.LogSoftmax(dim=1)))

        seq = torch.nn.Sequential(OrderedDict(modules))
        return seq   

    model = pretrained_model_factory()
    for param in model.parameters():
        param.requires_grad = False

    input_size = model.classifier[0].in_features
    model.classifier = make_classifier_net(input_size, output_size, hidden_layers, drop_prob)
    return model

def pretrained_model_factory(arch):
    """
        Return factory function for specified model.
    
        arc ::= architecture name of pre-trained model.
    """
    models = {
       'vgg11': lambda: tv.models.vgg11(pretrained=True),
       'vgg13': lambda: tv.models.vgg13(pretrained=True),
       'vgg16': lambda: tv.models.vgg16(pretrained=True),
       'vgg19': lambda: tv.models.vgg19(pretrained=True),
       'densenet121': lambda: tv.models.densenet121(pretrained=True),
       'densenet169': lambda: tv.models.densenet169(pretrained=True),
       'densenet161': lambda: tv.models.densenet161(pretrained=True),
       'densenet201': lambda: tv.models.densenet201(pretrained=True),
    }
    return  models[name]

