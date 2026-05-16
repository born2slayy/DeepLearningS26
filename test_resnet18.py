# Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2

from torchvision.models import resnet18
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from config import DATA_DIR, MODEL_SAVE_DIR


def test(args):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_data = CIFAR10Dataset(DATA_DIR, Subset.TEST, transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=128, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_test_data = len(test_data)

    net = resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, test_data.num_classes())
    model = DeepClassifier(net)
    model.load(args.path_to_trained_model)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    test_metric = Accuracy(classes=test_data.classes)

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            prediction = model(images)
            loss = loss_fn(prediction, labels)

            total_loss += loss.item() * images.shape[0]
            test_metric.update(prediction, labels)

    test_loss = total_loss / num_test_data
    print(f"test loss: {test_loss:.4f}")
    print()
    print(test_metric)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0
    args.path_to_trained_model = str(MODEL_SAVE_DIR / "ResNet_model_best.pth")

    test(args)
