import os

import torch
import torchvision as tv

import image_api

def make_dataloaders(data_path, batch_size=64):
    """Return instance of Dataloader class.
    """
    return Dataloader(data_path, batch_size)

def make_checkpoint_store(checkpoint_dir):
    """Return instance of Checkpoint class.
    """
    return Checkpoint(checkpoint_dir)

class Dataloader(object):
    def __init__(self, path, batch_size):
        self.root = path
        self.train = torch.utils.data.DataLoader(self._make_dataset('train'), 
                                                 batch_size=batch_size, 
                                                 shuffle=True),
        self.validation = torch.utils.data.DataLoader(self._make_dataset('valid'), 
                                                      batch_size=32, 
                                                      shuffle=True),
        self.test = torch.utils.data.DataLoader(self._make_dataset('test'), 
                                                batch_size=32, 
                                                shuffle=False)
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
        elif name == 'validation': 
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
            raise 'Unknown dataset name: {}'.format(name)

class Checkpoint(object):
    def __init__(self, path):
        self.root = path
    
    def store(self, filename, model, classifier, epoch, report, class_to_index):
        checkpoint = {
            'classifier.state_dict': model.classifier.state_dict(),
            'classifier': classifier,
            'epoch': epoch,
            'batch_size': batch_size,
            'class_to_idx': class_to_index,
            'perf_report': report
        }
        torch.save(checkpoint, os.path.join(self.root, filename))

    def store_reference(filename, score, checkpoint_filename):
        """ Store a reference to
            a checkpoint file with model.
        """
        ref = {
            'score': score,
            'checkpoint': os.path.join(self.root, checkpoint_filename)
        }
        torch.save(ref, os.path.join(self.root, filename))

