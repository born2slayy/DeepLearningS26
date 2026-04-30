from abc import ABCMeta, abstractmethod
import torch
from typing import Dict


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """

        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """

        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0  # number of correct predictions
        self.n_total = 0
        self.per_class_accuracies = {}  # dict mapping class name to accuracy

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """

        # TODO implement
        
        if len(prediction.shape) != 2:
            raise ValueError(f"Prediction must have 2 dimensions (batch_size, n_classes), got {len(prediction.shape)}")
        if len(target.shape) != 1:
            raise ValueError(f"Target must have 1 dimension (batch_size,), got {len(target.shape)}")
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("Batch size of prediction and target must match")

        _, predicted_indices = torch.max(prediction, dim=1)

        correct = (predicted_indices == target)
        self.n_matching += correct.sum().item()
        self.n_total += target.size(0)

        for i, classname in enumerate(self.classes):
            class_mask = (target == i)
            num_class_samples = class_mask.sum().item()
            
            if num_class_samples > 0:
                self.total_pred[classname] += num_class_samples
                self.correct_pred[classname] += (predicted_indices[class_mask] == i).sum().item()

    def __str__(self):
        """
        Return a string representation of the performance including:
        - overall accuracy
        - mean per-class accuracy
        - individual per-class accuracies for all classes
        """

        # TODO implement
        overall_acc = self.accuracy()
        mean_per_class_acc = self.per_class_accuracy()

        output = f"accuracy: {overall_acc:.4f}\n"
        output += f"per class accuracy: {mean_per_class_acc:.4f}\n"

        for classname in self.classes:
            acc = self.per_class_accuracies.get(classname, 0.0)
            output += f"Accuracy for class: {classname:6s} is {acc:.2f}\n"

        return output

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """

        # TODO implement
        if self.n_total == 0:
            return 0.0
        return self.n_matching / self.n_total

    def per_class_accuracy(self) -> float:
        """
        Compute and return the mean per-class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        Saves the individual per-class accuracies in self.per_class_accuracies as a dict mapping class name to accuracy.
        """
        # TODO implement
        if not self.classes:
            return 0.0

        accuracies = []
        for classname in self.classes:
            total = self.total_pred[classname]
            if total > 0:
                acc = self.correct_pred[classname] / total
            else:
                acc = 0.0 
            
            self.per_class_accuracies[classname] = acc
            accuracies.append(acc)

        return sum(accuracies) / len(accuracies)
