from torchvision import transforms

__all__ = ["BASIC_TRAIN_TRANS", "BASIC_TEST_TRANS"]

normalize = transforms.Normalize(mean=[0.5663, 0.4194, 0.3581],
                                 std=[0.3008,  0.2395, 0.2168])
BASIC_TRAIN_TRANS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    normalize
    ])
BASIC_TEST_TRANS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])