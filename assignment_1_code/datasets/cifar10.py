import pickle
from typing import Tuple
import numpy as np
import os


from assignment_1_code.datasets.dataset import Subset, ClassificationDataset


class CIFAR10Dataset(ClassificationDataset):
    """
    Custom CIFAR-10 Dataset.
    """

    def __init__(self, fdir: str, subset: Subset, transform=None):
        """
        Initializes the CIFAR-10 dataset.
        """
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        self.fdir = fdir
        self.subset = subset
        self.transform = transform

        self.images, self.labels = self.load_cifar()

    def load_cifar(self) -> Tuple:
        """
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Depending on which subset is selected, the corresponding images and labels are returned.

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        Labels should be returned either as a Python list of ints or as a
        numpy array with dtype int64.
        """

        # TODO implement
        # See the CIFAR-10 website on how to load the data files
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        if self.subset == Subset.TRAINING:
            files = [f"data_batch_{i}" for i in range(1, 5)]
        elif self.subset == Subset.VALIDATION:
            files = ["data_batch_5"]
        elif self.subset == Subset.TEST:
            files = ["test_batch"]
        else:
            raise ValueError("Unknown subset")

        all_images = []
        all_labels = []

        for file_name in files:
            file_path = os.path.join(self.fdir, file_name)
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            entry = unpickle(file_path)
            all_images.append(entry[b'data'])
            all_labels.extend(entry[b'labels'])

        # CIFAR-10: (N, 3072) -> R(1024), G(1024), B(1024) 
        images = np.concatenate(all_images, axis=0)
        
        # (N, 3, 32, 32) reshape -> (N, 32, 32, 3)transpose
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # data type (uint8)
        images = images.astype(np.uint8)
        
        # labels to int64 numpy array
        labels = np.array(all_labels, dtype=np.int64)
        
        return images, labels

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        # TODO implement
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        """
        # TODO implement
        if idx >= len(self):
            raise IndexError("Index out of bounds")

        image = self.images[idx]
        label = self.labels[idx]

        # transform is not none (v2.ToImage(), v2.ToDtype)
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        # TODO implement
        return len(self.classes)
