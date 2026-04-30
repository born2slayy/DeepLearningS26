import torch
from torch.utils.data import DataLoader
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

# for wandb users:
# from assignment_1_code.wandb_logger import WandBLogger


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
    ) -> None:
        """
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        """

        ## TODO implement
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.val_frequency = val_frequency

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

        self.best_val_mPCAcc = 0.0

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        ## TODO implement
        self.model.train()
        self.train_metric.reset()
        running_loss = 0.0

        for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch_idx} [Train]"):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            self.train_metric.update(outputs, labels)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        mAcc = self.train_metric.accuracy()
        mPCAcc = self.train_metric.per_class_accuracy()

        print(f"\n______epoch {epoch_idx} (Train)")
        print(self.train_metric) 

        return epoch_loss, mAcc, mPCAcc

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        ## TODO implement
        self.model.eval()
        self.val_metric.reset()
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc=f"Epoch {epoch_idx} [Val]"):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                running_loss += loss.item() * images.size(0)
                self.val_metric.update(outputs, labels)

        epoch_loss = running_loss / len(self.val_loader.dataset)
        mAcc = self.val_metric.accuracy()
        mPCAcc = self.val_metric.per_class_accuracy()

        print(f"\n______epoch {epoch_idx} (Validation)")
        print(self.val_metric)

        return epoch_loss, mAcc, mPCAcc

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        ## TODO implement
        for epoch in range(self.num_epochs):
            # 1. Train
            train_loss, train_acc, train_mPCAcc = self._train_epoch(epoch)
            
            # 2. LR Scheduler step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # 3. Validation
            if (epoch + 1) % self.val_frequency == 0 or epoch == self.num_epochs - 1:
                val_loss, val_acc, val_mPCAcc = self._val_epoch(epoch)
                # 4. best model.pt 
                if val_mPCAcc > self.best_val_mPCAcc:
                    self.best_val_mPCAcc = val_mPCAcc
                    self.model.save(self.training_save_dir, suffix="best")
                    print(f"New best model saved with mPCAcc: {val_mPCAcc:.4f}")
