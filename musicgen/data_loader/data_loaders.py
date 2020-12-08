from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.mididatasetv1 import *


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BasicMidiDataloader(BaseDataLoader):
    """
    Basic midi
    """
    def __init__(self, data_dir, vocab_file, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, crop=None, prefix=None, N=100):
        self.dataset = BasicMidiDataset(data_dir, vocab_file, train=training, crop=crop, prefix=prefix, N=N)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

