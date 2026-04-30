import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from pathlib import Path

# 필수 클래스 임포트
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from vit import ViT 
from config import DATA_DIR, MODEL_SAVE_DIR

def test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_data = CIFAR10Dataset(DATA_DIR, Subset.TEST, transform=transform)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    
    base_net = ViT(
        patch_size=4,
        emb_size=256,   
        depth=8,        
        num_heads=8,
        n_classes=10
    )
    
    model = DeepClassifier(base_net)
    

    checkpoint_path = args.path_to_trained_model
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading weights from: {checkpoint_path}")
    model.load(checkpoint_path)
    model.to(device)
    model.eval() 

    test_metric = Accuracy(classes=test_data.classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    running_loss = 0.0

    print("Evaluating model on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            test_metric.update(outputs, labels)

    final_loss = running_loss / len(test_data)
    print(f"\ntest loss: {final_loss:.6f}")
    print("-" * 30)
    print(test_metric) 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Improved ViT")
    parser.add_argument("-d", "--gpu_id", default="0", type=str, help="GPU ID to use")
    parser.add_argument("--path", default=str(MODEL_SAVE_DIR / "model_best.pth"), type=str)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.path_to_trained_model = args.path

    test(args)