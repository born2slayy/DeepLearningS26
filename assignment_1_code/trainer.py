import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

from assignment_1_code.wandb_logger import WandBLogger


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class of all Trainers.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Holds validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Holds training logic for one epoch.
        """

        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        num_epochs: int,
        training_save_dir: Path,
        batch_size: int = 4,
        val_frequency: int = 5,
        run_name: str = "resnet18_wd005",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency

        self.train_data_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.val_data_loader = torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        self.training_save_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_per_class_accuracy = -1.0

        self.train_losses = []
        self.train_accuracies = []
        self.train_per_class_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_per_class_accuracies = []

        self.logger = WandBLogger(enabled=True, run_name=run_name)

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()
        self.train_metric.reset()

        total_loss = 0.0

        for images, labels in tqdm(self.train_data_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            prediction = self.model(images)
            loss = self.loss_fn(prediction, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.shape[0]
            self.train_metric.update(prediction.detach(), labels.detach())

        epoch_loss = total_loss / len(self.train_data)
        epoch_accuracy = self.train_metric.accuracy()
        epoch_per_class_accuracy = self.train_metric.per_class_accuracy()

        print(f"______epoch {epoch_idx} train")
        print(f"loss: {epoch_loss:.4f}")
        print(self.train_metric)

        return epoch_loss, epoch_accuracy, epoch_per_class_accuracy

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.eval()
        self.val_metric.reset()

        total_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(self.val_data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                prediction = self.model(images)
                loss = self.loss_fn(prediction, labels)

                total_loss += loss.item() * images.shape[0]
                self.val_metric.update(prediction, labels)

        epoch_loss = total_loss / len(self.val_data)
        epoch_accuracy = self.val_metric.accuracy()
        epoch_per_class_accuracy = self.val_metric.per_class_accuracy()

        print(f"______epoch {epoch_idx} validation")
        print(f"loss: {epoch_loss:.4f}")
        print(self.val_metric)

        return epoch_loss, epoch_accuracy, epoch_per_class_accuracy

    def train(self) -> None:
        for epoch_idx in range(self.num_epochs):
            train_loss, train_accuracy, train_per_class_accuracy = self._train_epoch(
                epoch_idx
            )
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.train_per_class_accuracies.append(train_per_class_accuracy)

            if self.logger is not None:
                self.logger.log(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "train_per_class_accuracy": train_per_class_accuracy,
                    },
                    step=epoch_idx,
                )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            should_validate = (
                (epoch_idx + 1) % self.val_frequency == 0
                or epoch_idx == self.num_epochs - 1
            )

            if should_validate:
                val_loss, val_accuracy, val_per_class_accuracy = self._val_epoch(
                    epoch_idx
                )
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
                self.val_per_class_accuracies.append(val_per_class_accuracy)

                if self.logger is not None:
                    self.logger.log(
                        {
                            "val_loss": val_loss,
                            "val_accuracy": val_accuracy,
                            "val_per_class_accuracy": val_per_class_accuracy,
                        },
                        step=epoch_idx,
                    )

                if val_per_class_accuracy > self.best_val_per_class_accuracy:
                    self.best_val_per_class_accuracy = val_per_class_accuracy
                    self.model.save(self.training_save_dir, suffix="best")
