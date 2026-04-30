# import torch
# import torchvision
# import torchvision.transforms.v2 as v2
# import matplotlib.pyplot as plt
# import numpy as np


# from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
# from assignment_1_code.datasets.dataset import Subset

# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.imsave("test_1.png", np.transpose(npimg, (1, 2, 0)))


# if __name__ == "__main__":
#     classes = (
#         "plane",
#         "car",
#         "bird",
#         "cat",
#         "deer",
#         "dog",
#         "frog",
#         "horse",
#         "ship",
#         "truck",
#     )

#     transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

#     train_data = CIFAR10Dataset(
#         fdir="C:/Users/ganks/Downloads/Deeplearnings26/dlvc-ss26/cifar-10-batches-py", subset=Subset.TRAINING, transform=transform
#     )
#     train_data_loader = torch.utils.data.DataLoader(
#         train_data, batch_size=8, shuffle=False, num_workers=2
#     )

#     # get some random training images
#     dataiter = iter(train_data_loader)
#     images, labels = next(dataiter)

#     # show images
#     imshow(torchvision.utils.make_grid(images))
#     # print labels
#     print(" ".join(f"{classes[labels[j]]:5s}" for j in range(8)))

import torch
import torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np
import os

from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from config import DATA_DIR # config에서 경로 가져오기

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave("test_1.png", np.transpose(npimg, (1, 2, 0)))
    plt.show()
    print("이미지가 'test_1.png'로 저장되었습니다. 파일을 열어 내용을 확인하세요.")

if __name__ == "__main__":
    # --- 검증 파트 시작 ---
    
    # 1. 각 서브셋 로드
    transform_simple = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    
    train_data = CIFAR10Dataset(fdir=DATA_DIR, subset=Subset.TRAINING, transform=None)
    val_data = CIFAR10Dataset(fdir=DATA_DIR, subset=Subset.VALIDATION, transform=None)
    test_data = CIFAR10Dataset(fdir=DATA_DIR, subset=Subset.TEST, transform=None)

    # 2. 샘플 개수 확인 (요구사항: 40000, 10000, 10000)
    print(f"Train samples: {len(train_data)} (Expected: 40000)")
    print(f"Val samples: {len(val_data)} (Expected: 10000)")
    print(f"Test samples: {len(test_data)} (Expected: 10000)")

    # 3. 이미지 형태 및 타입 확인 (요구사항: (32, 32, 3), uint8)
    sample_img, sample_lbl = train_data[0]
    print(f"Image shape: {sample_img.shape} (Expected: (32, 32, 3))")
    print(f"Image type: {sample_img.dtype} (Expected: uint8)")

    # 4. 첫 10개 학습 데이터 라벨 확인 (요구사항: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3])
    first_10_labels = [train_data.labels[i] for i in range(10)]
    print(f"First 10 labels: {list(first_10_labels)}")
    print(f"Expected labels: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3]")

    # --- 시각화 파트 (기존 viz.py 기능) ---
    
    # 시각화를 위해 transform 적용된 버전 다시 생성
    train_data.transform = transform_simple
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size=8, shuffle=False, num_workers=0
    )

    dataiter = iter(train_data_loader)
    images, labels = next(dataiter)

    # 이미지 저장 및 라벨 출력
    imshow(torchvision.utils.make_grid(images))
    classes = train_data.classes
    print("\nFirst 8 images classes:")
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(8)))
    print("Expected: frog  truck truck deer  car   car   bird  horse")