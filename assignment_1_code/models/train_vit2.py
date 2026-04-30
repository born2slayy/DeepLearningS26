import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
from vit import ViT

from assignment_1_code.models.class_model import (
    DeepClassifier,
)
from assignment_1_code.metrics import Accuracy
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from config import DATA_DIR, MODEL_SAVE_DIR


def train(args):
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2470, 0.2435, 0.2616]

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cifar10_mean, std=cifar10_std),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random"),
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])

    train_data = CIFAR10Dataset(DATA_DIR, Subset.TRAINING, transform=train_transform)
    val_data = CIFAR10Dataset(DATA_DIR, Subset.VALIDATION, transform=val_transform)

    use_cuda = torch.cuda.is_available() and args.use_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using device: {device}")

    base_net = ViT(
        patch_size=4,
        emb_size=384,
        depth=8,
        num_heads=6,
        dropout=0.1,
        n_classes=10,
    )
    model = DeepClassifier(base_net)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    warmup_epochs = min(10, max(1, args.num_epochs // 10))
    cosine_epochs = max(1, args.num_epochs - warmup_epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=1e-5,
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)

    model_save_dir = Path(MODEL_SAVE_DIR)
    model_save_dir.mkdir(exist_ok=True)

    trainer = ImgClassificationTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=lr_scheduler,
        train_metric=train_metric,
        val_metric=val_metric,
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_epochs=args.num_epochs,
        training_save_dir=model_save_dir,
        batch_size=args.batch_size,
        val_frequency=args.val_frequency,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT on CIFAR-10")
    parser.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-2)
    parser.add_argument("--val_frequency", type=int, default=1)
    parser.add_argument("--use_cuda", action="store_true", default=True)
    parser.add_argument("--no_cuda", action="store_false", dest="use_cuda")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    train(args)
