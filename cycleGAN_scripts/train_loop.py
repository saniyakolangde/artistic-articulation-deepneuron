import torch
import config
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
from dataset import HumanDogDataset
from torch.utils.data import DataLoader
from utils import save_checkpoint, load_checkpoint

# discF1 = human faces
# discF2 = paintings
# genG1 = human face generator
# genG2 = painting generator
def train_fn(discA, discB, genA2B, genB2A, loader, opt_disc, opt_gen, cycle_consistency, adversarial, d_scaler, g_scaler): # from tutorial
    Real1 = 0
    Fake1 = 0
    loop = tqdm(loader, leave=True)

    for idx, (imgA, imgB) in enumerate(loop):
        imgA = imgA.to(config.DEVICE)
        imgB = imgB.to(config.DEVICE)

        # Train Discriminators A and B
        with torch.cuda.amp.autocast():
            fake_imgA = genB2A(imgB) # generate
            discA_real = discA(imgA)
            discA_fake = discA(fake_imgA.detach())
            Real1 += discA_real.mean().item()
            Fake1 += discA_fake.mean().item()
            discA_real_loss = adversarial(discA_real, torch.ones_like(discA_real))
            discA_fake_loss = adversarial(discA_fake, torch.zeros_like(discA_fake))
            discA_loss = discA_real_loss + discA_fake_loss

            fake_imgB = genA2B(imgA)
            discB_real = discB(imgB)
            discB_fake = discB(fake_imgB.detach())
            discB_real_loss = adversarial(discB_real, torch.ones_like(discB_real))
            discB_fake_loss = adversarial(discB_fake, torch.zeros_like(discB_fake))
            discB_loss = discB_real_loss + discB_fake_loss

            # put it togethor
            D_loss = (discA_loss + discB_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators A2B and B2A
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            discA_fake = discA(fake_imgA)
            discB_fake = discB(fake_imgB)
            loss_genA2B = adversarial(discA_fake, torch.ones_like(discA_fake))
            loss_genB2A = adversarial(discB_fake, torch.ones_like(discB_fake))

            # cycle loss
            cycle_genB2A = genB2A(fake_imgB)
            cycle_genA2B = genA2B(fake_imgA)
            cycle_genB2A_loss = cycle_consistency(imgB, cycle_genA2B)
            cycle_genA2B_loss = cycle_consistency(imgA, cycle_genB2A)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_genB2A = genB2A(imgA) # B2A should just output A if imgA fed
            identity_genA2B = genA2B(imgB) # A2B should just output B if imgB fed
            identity_genB2A_loss = cycle_consistency(imgA, identity_genB2A)
            identity_genA2B_loss = cycle_consistency(imgB, identity_genA2B)

            # add all togethor
            G_loss = (
                    loss_genB2A
                    + loss_genA2B
                    + cycle_genB2A_loss * config.LAMBDA_CYCLE
                    + cycle_genA2B_loss * config.LAMBDA_CYCLE
                    + identity_genA2B_loss * config.LAMBDA_IDENTITY
                    + identity_genB2A_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_imgA * 0.5 + 0.5, f"saved_images/genGA2B_{idx}.png")
            save_image(fake_imgB * 0.5 + 0.5, f"saved_images/genGB2A_{idx}.png")

        loop.set_postfix(H_real=Real1 / (idx + 1), H_fake=Fake1 / (idx + 1))


def main():
    discriminatorA = Discriminator(in_channels=3).to(config.DEVICE)
    discriminatorB = Discriminator(in_channels=3).to(config.DEVICE)
    generatorA2B = Generator(in_channels=3, num_residuals=9).to(config.DEVICE)
    generatorB2A = Generator(in_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(discriminatorA.parameters()) + list(discriminatorB.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(generatorA2B.parameters()) + list(generatorB2A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    cycle_loss = nn.L1Loss() # cycle consistency, indentity loss
    adversarial_loss = nn.MSELoss() # adversarial loss

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_A2B,
            generatorA2B,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_B2A,
            generatorB2A,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_A,
            discriminatorA,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_B,
            discriminatorB,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = HumanDogDataset(
        path_dset1=config.TRAIN_DIR + "/human_faces",
        path_dset2=config.TRAIN_DIR + "/monet_pruned", # monet pruned
        transform=config.transforms,
    )

    # val_dataset = HumanDogDataset(
    #     path_human=config.VAL_DIR + "/human",
    #     path_dog=config.VAL_DIR + "/dog",
    #     transform=config.transforms,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    # )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            discriminatorA,
            discriminatorB,
            generatorA2B,
            generatorB2A,
            loader,
            opt_disc,
            opt_gen,
            cycle_loss,
            adversarial_loss,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(generatorA2B, opt_gen, filename=config.CHECKPOINT_GEN_A2B) # human faces
            save_checkpoint(generatorB2A, opt_gen, filename=config.CHECKPOINT_GEN_B2A) # pizza
            save_checkpoint(discriminatorA, opt_disc, filename=config.CHECKPOINT_DISC_A)
            save_checkpoint(discriminatorB, opt_disc, filename=config.CHECKPOINT_DISC_B) # pizza


if __name__ == "__main__":
    main()
