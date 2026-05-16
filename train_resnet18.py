# Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path

from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.metrics import Accuracy
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from config import DATA_DIR, MODEL_SAVE_DIR


def train(args):
    # Run notes:
    # 1. baseline: default transforms, AdamW(lr=0.001, amsgrad=True, weight_decay=0.01)
    # 2. augmentation: baseline + RandomCrop(32, padding=4) + RandomHorizontalFlip(0.5)
    # 3. stronger regularization: default transforms, weight_decay=0.05
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_data = CIFAR10Dataset(DATA_DIR, Subset.TRAINING, train_transform)
    val_data = CIFAR10Dataset(DATA_DIR, Subset.VALIDATION, val_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchvision.models import resnet18

    net = resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, train_data.num_classes())
    model = DeepClassifier(net)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, amsgrad=True, weight_decay=0.05
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path(MODEL_SAVE_DIR)
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    trainer = ImgClassificationTrainer(
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        args.num_epochs,
        model_save_dir,
        batch_size=128,
        val_frequency=val_frequency,
    )
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 30

    train(args)
