import os

import torch
import torch.utils.data
import torchvision as tv

import image_api

def make_dataloaders(data_path, batch_size=64):
    """Return instance of Dataloader class.
    """
    return ImageDataSource(data_path, batch_size)

def make_checkpoint_store(checkpoint_dir):
    """Return instance of Checkpoint class.
    """
    return Checkpoint(checkpoint_dir)

class ImageDataSource(object):
    def __init__(self, path, batch_size):
        self.root = path
        self.train = torch.utils.data.DataLoader(self._make_dataset('train'), 
                                                 batch_size=batch_size, 
                                                 shuffle=True)
        self.validation = torch.utils.data.DataLoader(self._make_dataset('valid'), 
                                                      batch_size=32, 
                                                      shuffle=True)
        self.test = torch.utils.data.DataLoader(self._make_dataset('test'), 
                                                batch_size=32, 
                                                shuffle=False)

    @property
    def class_to_index(self):
        """ Target class to class index dictionary. 
        """
        return self.train.dataset.class_to_idx

    def _make_dataset(self, name):
        return tv.datasets.ImageFolder(os.path.join(self.root, name), 
                                       transform=self._make_transform(name))

    @staticmethod 
    def _make_transform(name):
        crop_size = image_api.get_crop_size()
        image_means = image_api.get_image_means()
        image_std = image_api.get_image_std()

        if name == 'train': 
            return tv.transforms.Compose([
                        tv.transforms.RandomRotation(20),
                        tv.transforms.Resize(256),
                        tv.transforms.RandomResizedCrop(crop_size[0]),
                        tv.transforms.RandomHorizontalFlip(0.5),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(image_means, image_std)])
        elif name == 'valid': 
            return tv.transforms.Compose([
                        tv.transforms.Resize(256),
                        tv.transforms.CenterCrop(crop_size[0]),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize(image_means, image_std)
                    ])
        elif name == 'test': 
            return tv.transforms.Compose([
                    tv.transforms.Resize(256),
                    tv.transforms.CenterCrop(crop_size[0]),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize(image_means, image_std)    
                ])
        else:
            raise Exception('Unknown dataset name: {}'.format(name))

class Checkpoint(object):
    def __init__(self, path):
        self.root = path
    
    def store(self, filename, model, classifier, epoch, report, class_to_index):
        checkpoint = {
            'classifier.state_dict': model.classifier.state_dict(),
            'classifier': classifier,
            'epoch': epoch,
            'class_to_index': class_to_index,
            'perf_report': report
        }
        torch.save(checkpoint, os.path.join(self.root, filename))

    def store_reference(self, filename, score, checkpoint_filename):
        """ Store a reference to
            a checkpoint file with model.
        """
        ref = {
            'score': score,
            'checkpoint': os.path.join(self.root, checkpoint_filename)
        }
        torch.save(ref, os.path.join(self.root, filename))

    @staticmethod
    def read(filepath):
        """ Return checkpoint dictionary.
        """
	# Read checkpoint file and re-map storage
        # to lowest common denominator - 'cpu'.
        return torch.load(filepath, map_location=lambda storage, loc: storage)


