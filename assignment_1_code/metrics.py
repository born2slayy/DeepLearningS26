from abc import ABCMeta, abstractmethod
import torch


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
        self.n_matching = 0
        self.n_total = 0
        self.per_class_accuracies = {classname: 0.0 for classname in self.classes}

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """

        if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError("prediction and target must be torch tensors")

        if len(prediction.shape) != 2:
            raise ValueError("prediction must have shape (batchsize, n_classes)")

        if len(target.shape) != 1:
            raise ValueError("target must have shape (batchsize,)")

        if prediction.shape[0] != target.shape[0]:
            raise ValueError("prediction and target must have the same batch size")

        if prediction.shape[1] != len(self.classes):
            raise ValueError("prediction has wrong number of classes")

        if torch.any(target < 0) or torch.any(target >= len(self.classes)):
            raise ValueError("target values must be between 0 and c-1")

        predicted_classes = torch.argmax(prediction, dim=1)

        self.n_matching += int((predicted_classes == target).sum().item())
        self.n_total += int(target.shape[0])

        for pred_label, true_label in zip(predicted_classes, target):
            class_name = self.classes[int(true_label.item())]
            self.total_pred[class_name] += 1
            if int(pred_label.item()) == int(true_label.item()):
                self.correct_pred[class_name] += 1

    def __str__(self):
        """
        Return a string representation of the performance including:
        - overall accuracy
        - mean per-class accuracy
        - individual per-class accuracies for all classes
        """

        accuracy = self.accuracy()
        per_class_accuracy = self.per_class_accuracy()

        lines = [
            f"accuracy: {accuracy:.4f}",
            f"per class accuracy: {per_class_accuracy:.4f}",
        ]

        for class_name in self.classes:
            class_accuracy = self.per_class_accuracies[class_name]
            lines.append(f"Accuracy for class: {class_name:<5} is {class_accuracy:.2f}")

        return "\n".join(lines)

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """

        if self.n_total == 0:
            return 0.0

        return self.n_matching / self.n_total

    def per_class_accuracy(self) -> float:
        """
        Compute and return the mean per-class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        Saves the individual per-class accuracies in self.per_class_accuracies as a dict mapping class name to accuracy.
        """
        if self.n_total == 0:
            self.per_class_accuracies = {classname: 0.0 for classname in self.classes}
            return 0.0

        for class_name in self.classes:
            total = self.total_pred[class_name]
            if total == 0:
                self.per_class_accuracies[class_name] = 0.0
            else:
                self.per_class_accuracies[class_name] = (
                    self.correct_pred[class_name] / total
                )

        return sum(self.per_class_accuracies.values()) / len(self.classes)
