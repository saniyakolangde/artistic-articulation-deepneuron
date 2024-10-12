import torch
from torchvision import transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "../data/train"
VAL_DIR = "../data/validation"
TEST_DIR = "../data/test/"
BATCH_SIZE = 8
SEED_SIZE = 16
LEARNING_RATE = 2e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_Pokemon = "genLogos.pth.tar" #genh
CHECKPOINT_DISC_Pokemon = "discLogos.pth.tar" #disch


transforms1 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((64,64)),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ],
)

transforms2 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ],
)