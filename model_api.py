from collections import OrderedDict
import copy

import numpy as np
import torch
import torchvision as tv

import data_api
from image_api import make_image

class NeuralNetClassifier(object):
    
    def __init__(self, checkpoint=None, 
                       arch='vgg19', hidden_layers=[4096], output_size=2,
                       dropout=0.3, use_gpu=True, 
                       class_to_name=None):
        """
        Parameters:
        ----------
        checkpoint:    descriptor of an existing model.
        arch:          name of pre-trained architecture
        hidden_layers: collection of sizes of hidden layers.
        output_size:   number of classes.
        dropout:       probability of dropping out nodes in the neural net.
        use_gpu:       use GPU if available and use_gpu is set to True.
        class_to_name: target class to name dictionary.
        """
        self.device = make_device(use_gpu)
        if checkpoint is not None:
            self.model = make_model_from_checkpoint(checkpoint)
            self.index_to_class = {index: klass for klass, index in checkpoint['class_to_index'].items()}
            self.arch_name = checkpoint['meta.arch']
            self.hidden_layers = checkpoint['meta.hidden-layers']
            self.output_size = checkpoint['meta.output-size']
            self.dropout = checkpoint['meta.dropout']
        else:
            self.model = make_model(pretrained_model_factory(arch),
                                    output_size, hidden_layers, dropout)
            self.arch_name = arch
            self.hidden_layers = hidden_layers
            self.output_size = output_size
            self.dropout = dropout

        self.model.to(self.device)
        self.class_to_name = class_to_name if class_to_name is not None else {}
            
    def fit(self, data_path, checkpoint_path,
                momentum=0., batch_size=96,
                learning_rate_policy='constant', learning_rate_init=0.02, dump_koeff=1.0,
                epochs_per_cycle=1, num_cycles=10, report_every_n=30):
        """ Fit model to trainig data set.

            Parameters:
            ----------
            data_path: path to a directory with training and validation data.
            checkpoint_path: path to a directory to store checkpoints in.
            momentum: momentum factor for optimizer / koeff to decrease weights fluctuation.
            batch_size: number of samples to include into a training data batch.
            learning_rate_policy: 'constant' or 'cosine'
                                  'constant': learning rate value does not change
                                              between iterations.
                                   'cosine': learning rate changes during training
                                             per cosine annealing schedule.
            learning_rate_init: learning rate value. 
                                Maximum learning rate for cosine annealing schedule.

	    dump_koeff: learning rate dumping koefficient.

            epochs_per_cycle:
                            - set to 1 for 'constant' learning rate policy.
                            - defines number of epochs per cosine annealing schedule cycle.
            num_cycles:
                            - defines number of epochs for 'constant' learning rate policy.
                            - num_cycles * epochs_per_cycle defines number of epochs 
                            for 'cosine' learning rate policy.
            report_every_n: 
                            - defines number of iterattion between reports on training performance.
        """
        self.learning_rate_init = learning_rate_init
        self.report_every_n = report_every_n
        self.perf_report = []

        self.datasource = data_api.make_dataloaders(data_path, batch_size)
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.model.classifier.parameters(), 
                                         lr=learning_rate_init, 
                                         momentum=momentum)
        if learning_rate_policy == 'cosine':
            num_batches = len(self.datasource.train)
            iter_per_cycle = epochs_per_cycle * num_batches   
            epochs = epochs_per_cycle * num_cycles
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iter_per_cycle)
        else:
            epochs_per_cycle = 1
            epochs = num_cycles
            self.lr_scheduler = None

        # Track best classifier.
        best_score_acc = 0.
        best_score_loss = float('inf')
        best_epoch = None
        best_classifier_state = {}

        learning_rate_value = learning_rate_init
        checkpoint = data_api.make_checkpoint_store(checkpoint_path)
        save_as = "model.pth"

        for epoch_no in range(1, epochs + 1):
            train_loss = self._train()
            valid_acc, valid_loss = self._accuracy_loss_score(self.datasource.valid) 
            self.perf_report.append({'train-loss': train_loss,
                                     'valid-loss': valid_loss,
                                     'valid-acc': valid_acc})
            print("Epoch: {} of {}.. ".format(epoch_no, epochs),
                  "Train Loss: {:.3f}.. ".format(train_loss),
                  "Valid Loss: {:.3f}.. ".format(valid_loss),
                  "Valid Accuracy: {:.3f}".format(valid_acc))
                
            if valid_acc > best_score_acc and valid_loss < best_score_loss:
                best_score_acc = valid_acc
                best_score_loss = valid_loss
                best_epoch = epoch_no
                best_classifier_state = copy.deepcopy(self.model.classifier.state_dict())
               
            if epoch_no % epochs_per_cycle == 0:
               if self.lr_scheduler is not None: 
                  learning_rate_value = learning_rate_value * dump_koeff
                  self.optimizer = torch.optim.SGD(self.model.classifier.parameters(), 
                                                     lr=learning_rate_value, 
                                                     momentum=momentum)
                  self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iter_per_cycle)
        # Set best classifier
        self.model.classifier.load_state_dict(best_classifier_state)
        path = checkpoint.save("model.pth", 
                               self.model, 
                               meta = {"arch": self.arch_name,
                                       "output-size": self.output_size,
                                       "hidden-layers": self.hidden_layers,
                                       "dropout": self.dropout},
                               epoch=best_epoch, 
                               report=self.perf_report,
                               class_to_index=self.datasource.class_to_index)
        return path  

    def accuracy(self, dataloader):
        """ Compute model's accuracy on test data set.

            Return:
            ------
            model's accuracy on specified test data set.

            Parameters:
            ----------
            dataloader: dataloader instance
        """
        accuracy = 0
        N = len(dataloader)
        
        self.model.eval()
        with torch.no_grad():
            for x, y in iter(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model.forward(x)
                probs = torch.exp(y_hat)
                accuracy += (y.data == probs.max(dim=1)[1]).type(torch.FloatTensor).mean()
        return accuracy / N      

    def predict(self, image_path, top_k=5):
        """ Predict class of an image.

            Return:
            ------
            Return collection of tuples (class, class-name, probability) sorted by
            value of class probability in descending order.

            Parameters:
            ----------
            image_path: path to image file.
            top_k:      defines size of collection with predictions. 
        """
        self.model.eval()
        with torch.no_grad():
            image = torch.tensor(make_image(image_path))
            channels, width, height = image.shape
            image = image.reshape((1, channels, width, height))
            image = image.to(self.device)
            y_hat = self.model.forward(image)
            estimate = torch.exp(y_hat)
        estimate = estimate.to('cpu').numpy()
        index = np.argsort(estimate, axis=1)[0][-top_k:][::-1]
        probs = estimate[0][index]
        return probs, [self.index_to_class[x] for x in index]

    def _train(self):
        """ Train neural net for one epoch.
            Return mean train loss over all iteration of the current epoch.

            Parameters:
        """
        train_loss = []
        for x, y in iter(self.datasource.train):
            self.model.train()
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)        
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            self._step()
        return np.mean(train_loss)

    def _step(self):
        """ Track progress of the model's fitting.

        - Advance learning rate scheduler
        """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _accuracy_loss_score(self, dataloader):
        """ Return pair (accuracy, loss) per data set.
            Loss and accuracy are averaged over the dataset batches.
            
            Parameters:
            ----------
            dataloader ::= data set loader.
        """
        loss = 0
        accuracy = 0
        N = len(dataloader)
        
        self.model.eval()
        with torch.no_grad():
            for x, y in iter(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model.forward(x)
                loss += self.criterion(y_hat, y).item()
                probs = torch.exp(y_hat)
                accuracy += (y.data == probs.max(dim=1)[1]).type(torch.FloatTensor).mean()    
        return accuracy / N,  loss / N

def make_model_from_checkpoint(checkpoint):
    """ Return trained model from checkpoint.

	Parameters:

	checkpoint ::= dictionary with model state and properties. 
    """
    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        model = make_model(pretrained_model_factory(checkpoint["meta.arch"]),
                           checkpoint['meta.output-size'],
                           checkpoint['meta.hidden-layers'],
                           checkpoint['meta.dropout'])
        model.classifier.load_state_dict(checkpoint['model.classifier.state_dict'])
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

    # vggXX classifier is a Sequential container.
    # densenetXXX classifier is a single linear layer.
    try:
        input_size = model.classifier[0].in_features
    except TypeError:
        input_size = model.classifier.in_features

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


