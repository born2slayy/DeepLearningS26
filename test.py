# Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

from torchvision.models import resnet18  # change to the model you want to test
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

    # Use config.py for machine-dependent paths, e.g. DATA_DIR and MODEL_SAVE_DIR.
    test_data = CIFAR10Dataset(DATA_DIR, Subset.TEST, transform=transform)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_test_data = len(test_data)

    base_net = resnet18(num_classes=10)
    model = DeepClassifier(base_net)
    model.load(args.path_to_trained_model) 
    model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    test_metric = Accuracy(classes=test_data.classes)
    
    running_loss = 0.0

    # Below implement testing loop and print final loss
    # and metrics to terminal after testing is finished
    # ...

    print("Starting Testing...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            test_metric.update(outputs, labels)
    
    final_loss = running_loss / len(test_data)
    print(f"\ntest loss: {final_loss}")
    print(test_metric)

if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description="Testing")
    args.add_argument("-d", "--gpu_id", default="0", type=str, help="GPU ID")
    args = args.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.path_to_trained_model = MODEL_SAVE_DIR / "model_best.pth"

    test(args)
    
