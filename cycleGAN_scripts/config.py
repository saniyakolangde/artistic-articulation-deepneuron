import torch
from torchvision import transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "../data/train"
VAL_DIR = "../data/validation"
TEST_DIR = "../data/test/"
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 100
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_A2B = "genA2B.pth.tar" #genh
CHECKPOINT_GEN_B2A = "genB2A.pth.tar" #gend
CHECKPOINT_DISC_A = "discA.pth.tar" #disch
CHECKPOINT_DISC_B = "discB.pth.tar" #discB

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(p=0.3),
        # transforms.RandomVerticalFlip(p=0.3),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ],
)