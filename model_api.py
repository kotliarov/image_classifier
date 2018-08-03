from collections import OrderedDict
import numpy as np
import torch
import torchvision as tv


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
        else:
            self.model = make_model(pretrained_model_factory(arch),
                                    output_size, hidden_layers, dropout)
        self.model.to(self.device)
        self.class_to_name = class_to_name if class_to_name is not None else {}
            
    
    def fit(self, data_path, momentum=0.8, batch_size=96,
                learning_rate_policy='constant', learning_rate_init=0.02,
                epochs_per_cycle=1, num_cycles=10, report_every_n=30):
        """ Fit model to trainig data set.

            Parameters:
            ----------
            data_path: path to a directory with training and validation data.
            momentum: koeff to decrease weights fluctuation.
            batch_size: number of samples to include into a training data batch.
            learning_rate_policy: 'constant' or 'cosine'
                                  'constant': learning rate value does not change
                                              between iterations.
                                   'cosine': learning rate changes during training
                                             per cosine annealing schedule.
            learning_rate_init: learning rate value. 
                                Maximum learning rate for cosine annealing schedule.
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
        self.num_iter = 0
        self.accum = []
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

        best_score = 0.
        best_model_ref = None
        learning_rate_value = learning_rate_init
        checkpoint = data_api.make_checkpoint_store(checkpoint_dir)
       
        for epoch_no in range(1, epochs + 1):
            self.epoch = epoch_no
            train_loss = self._train()
            if epoch_no % epochs_per_cycle == 0:
                filename = "checkpoint_epoch_{}.pth".format(epoch_no)
                checkpoint.store(filename, 
                                 self.model, 
                                 classifier = {"architecture": arch_name,
                                              "output-size": output_size,
                                              "hidden-layers": hidden_layers,
                                              "dropout": dropout},
                                 epoch=epoch_no, 
                                 report=self.perf_report,
                                 class_to_index=datasource.class_to_index)

                valid_acc = accuracy_score(self.model, 
                                           self.device, 
                                           self.datasource.valid)
                if valid_acc > best_score:
                    best_score = valid_acc
                    best_model_ref = filename
                   
                if self.lr_scheduler is not None: 
                    learning_rate_value = learning_rate_value * dump_koeff
                    self.optimizer = torch.optim.SGD(self.model.classifier.parameters(), 
                                                     lr=learning_rate_value, 
                                                     momentum=momentum)
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iter_per_cycle)

        # Persist reference to best model's checkpoint
        checkpoint.store_reference("best_model.pth", 
                                    best_score, 
                                    best_model_ref)
        return best_score, best_model_ref, 'best_model.pth' 

    def accuracy(self, data_path):
        """ Compute model's accuracy on test data set.

            Return:
            ------
            model's accuracy on specified test data set.

            Parameters:
            ----------
            data_path: path to a directory with test data.
        """
        dataloader = data_api.make_dataloaders(data_path).test
        accuracy = 0
        N = len(dataloader)
        
        self.model.eval()
        with torch.no_grad():
            for x, y in iter(dataloader):
                x, y = x.to(device), y.to(device)
                y_hat = self.model.forward(x)
                probs = torch.exp(y_hat)
                accuracy += (y.data == probs.max(dim=1)[1]).type(torch.FloatTensor).mean()
        return accuracy / N      

    def predict(image_path, top_k=5):
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
        crop_width, crop_height = get_crop_size()
        self.model.eval()
        with torch.no_grad():
            image = torch.tensor(make_image(image_path))
            image = image.reshape((1, 3, crop_width, crop_height))
            image = image.to(device)
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
            self._step(loss.item())
        return np.mean(train_loss)

    def _step(self, train_loss):
        """ Track progress of the model's fitting.

        - Advance learning rate scheduler
        - Record performance metrics (train-loss, validation-loss, validation-accuracy).
        """
        # Adjust learning rate
        if self.lr_scheduler is not None:
            lr_rate = self.lr_scheduler.get_lr()[0]
            self.lr_scheduler.step()
        else:
            lr_rate = self.learning_rate_init

        self.accum.append(train_loss)
        n = len(self.accum)
        if n % self.report_every_n == 0:
            self.num_iter = self.num_iter + n
            valid_acc, valid_loss = accuracy_loss_score(self.model, 
                                                        self.device, 
                                                        self.datasource.valid, 
                                                        self.criterion)
            self.perf_report.append((self.num_iter, np.mean(self.accum), valid_loss, valid_acc))
            self.accum = []
            print("epoch: {} n_iter: {}.. ".format(self.epoch, self.num_iter),
                  "Learn-rate: {:.5f}..".format(lr_rate),
                  "Training Loss: {:.3f}.. ".format(self.perf_report[-1][1]),
                  "Valid Loss: {:.3f}.. ".format(self.perf_report[-1][2]),
                  "Valid Accuracy: {:.3f}".format(self.perf_report[-1][3]))

def make_model_from_checkpoint(checkpoint):
    """ Return trained model from checkpoint.

	Parameters:

	checkpoint ::= dictionary with model state and properties. 
    """
    classifier = checkpoint['classifier']
    
    model = make_model(pretrained_model_factory(classifier["architecture"]),
                       classifier['output-size'],
                       classifier['hidden-layers'],
                       classifier['dropout'])
    model.classifier.load_state_dict(checkpoint['classifier.state_dict'])
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

def accuracy_loss_score(model, device, dataloader, criterion):
    """
    Return pair (accuracy, loss) per data set.
    Loss and accuracy are averaged over the dataset batches.
    
        model      ::= network model
        device     ::= gpu or cpu device
        dataloader ::= test data source
        criterion  ::= cost function
    """
    loss = 0
    accuracy = 0
    N = len(dataloader)
    
    model.eval()
    with torch.no_grad():
        for x, y in iter(dataloader):
            x, y = x.to(device), y.to(device)
            y_hat = model.forward(x)
            loss += criterion(y_hat, y).item()
            probs = torch.exp(y_hat)
            accuracy += (y.data == probs.max(dim=1)[1]).type(torch.FloatTensor).mean()    
    return accuracy / N,  loss / N

