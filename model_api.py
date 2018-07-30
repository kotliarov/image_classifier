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

    classifier = checkpoint['classifier']
    
    model = make_model(pretrained_model_factory(classifier["architecture"]),
                       classifier['output-size'],
                       classifier['hidden-layers'],
                       classifier['dropout'])
    model.classifier.load_state_dict(checkpoint['classifier.state_dict'])

    device = make_device(use_gpu)
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

def make_device(use_gpu):
    """Return device.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if use_gpu else torch.device("cpu")
 
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
    return  models[arch]

def train(model, device, dataloader, criterion, optimizer, progress_tracker):
    """
    Return 
        mean train loss over all iteration of the current epoch.

        model      ::= network model
        device     ::= gpu or cpu device
        dataloader ::= test data source
        criterion  ::= cost function
        optimizer  ::= weights update policy
        progress_tracker ::= callable, keeps track fo training progress.
    """
    train_loss = []
    for x, y in iter(dataloader):
        model.train()
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_hat = model(x)        
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        progress_tracker(loss.item())
    return np.mean(train_loss)

def validation_score(model, device, dataloader, criterion):
    """
    Return pair (loss, accuracy) per validation set.
    Loss and accuracy are averaged over validation batches.
    
        model      ::= network model
        device     ::= gpu or cpu device
        dataloader ::= test data source
        criterion  ::= cost function
    """
    loss = 0
    accuracy = 0
    dataloader = iter(dataloader)
    N = len(dataloader)
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat = model.forward(x)
            loss += criterion(y_hat, y).item()
            probs = torch.exp(y_hat)
            accuracy += (y.data == probs.max(dim=1)[1]).type(torch.FloatTensor).mean()    
    return float(loss) / N, accuracy / N

def accuracy_score(model, device, dataloader):
    """
    Return model accuracy on specified data set.
    """
    accuracy = 0
    dataloader = iter(dataloader)
    N = len(dataloader)
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat = model.forward(x)
            probs = torch.exp(y_hat)
            accuracy += (y.data == probs.max(dim=1)[1]).type(torch.FloatTensor).mean()
    return accuracy / N

class ProgressTracker(object):
    """ Track NN training performance.
    """
    def __init__(self, model, device, dataloader, criterion, lr_scheduler, report_every_n):
        """
            @model       ::= model
            @device      ::= device (cuda or cpu)
            @dataloader  ::= validation data loader
            @criterion   ::= loss criterion
            @lr_scheduler::= learning rate scheduler
            @report_every_n    ::= defines number of training batches between validation tests.
        """
        self.model_ = model
        self.device_ = device
        self.dataloader_ = dataloader
        self.criterion_ = criterion
        self.lr_scheduler_ = lr_scheduler
        self.report_every_n_ = report_every_n
        self.perf_report_ = []
        self.acc_ = []
        self.num_iter_ = 0
        self.epoch_ = 1
    
    @property
    def epoch(self):
        return self.epoch_
    
    @epoch.setter
    def epoch(self, value):
        self.epoch_ = value
    
    @property
    def lr_scheduler(self):
        return self.lr_scheduler_

    @lr_scheduler.setter
    def lr_scheduler(self, value):
        self.lr_scheduler_ = value
    
    @property
    def performance_report(self):
        return self.perf_report_
    
    def __call__(self, train_loss):
        """
        - Advance learning rate scheduler
        - Record performance metrics (train-loss, validation-loss, validation-accuracy).
        """
        # Adjust learning rate
        lr_rate = self.lr_scheduler_.get_lr()[0]
        self.lr_scheduler_.step()
        
        self.acc_.append(train_loss)
        n = len(self.acc_)
        if n % self.report_every_n_ == 0:
            self.num_iter_ = self.num_iter_ + n
            valid_loss, valid_acc = validation_score(self.model_, 
                                                     self.device_, 
                                                     iter(self.dataloader_), 
                                                     self.criterion_)
            self.perf_report_.append((self.num_iter_, np.mean(self.acc_), valid_loss, valid_acc))
            self.acc_ = []
            print("epoch: {} n_iter: {}.. ".format(self.epoch_, self.num_iter_),
                  "Learn-rate: {:.5f}..".format(lr_rate),
                  "Training Loss: {:.3f}.. ".format(self.perf_report_[-1][1]),
                  "Valid Loss: {:.3f}.. ".format(self.perf_report_[-1][2]),
                  "Valid Accuracy: {:.3f}".format(self.perf_report_[-1][3]))

