import config

import torch.optim as optim
from discriminator import Discriminator
from generator import Generator
from utils import load_checkpoint
from torchvision.utils import save_image
import os
from PIL import Image
def eval():
    discriminator_Human = Discriminator(in_channels=3).to(config.DEVICE)
    discriminator_Dog = Discriminator(in_channels=3).to(config.DEVICE)
    generator_Human = Generator(in_channels=3, num_residuals=9).to(config.DEVICE)
    generator_Dog = Generator(in_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(discriminator_Human.parameters()) + list(discriminator_Dog.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(generator_Human.parameters()) + list(generator_Dog.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_A2B,
            generator_Human,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_B2A,
            generator_Dog,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_A,
            discriminator_Human,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_B,
            discriminator_Dog,
            opt_disc,
            config.LEARNING_RATE,
        )

    test = os.listdir(config.TEST_DIR)
    test_img = test[3]

    human_image = (config.transforms(Image.open(config.TEST_DIR + test_img).convert("RGB")).unsqueeze(0).cuda()) # need to add batch dimension -> [1,3,256,256]
    # print((human_image.unsqueeze(0).cuda()).shape)
    fake_dog = generator_Human(human_image)
    fake_human = generator_Dog(fake_dog)

    save_image(fake_dog * 0.5 + 0.5, f"data/test/test_saved/fake_G2_{0}.png")
    save_image(fake_human * 0.5 + 0.5, f"data/test/test_saved/fake_G1_{0}.png")


if __name__ == "__main__":
    eval()