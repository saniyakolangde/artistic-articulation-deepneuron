import torch
import config
import torch.nn.functional as F
from tqdm import tqdm
from discriminatorGAN import Discriminator
from generatorGAN import Generator
from datasetGAN import CustomDataset
from torch.utils.data import DataLoader
from utils import save_checkpoint, load_checkpoint, save_results
import matplotlib.pyplot as plt

def train_discriminator(discriminator, generator, real_pokemon, disc_optimizer, batch_size, seed_size, device):
    # Reset the gradients for the optimizer
    disc_optimizer.zero_grad()

    # Train on the real images
    real_predictions = discriminator(real_pokemon)
    #print(real_predictions)
    # real_targets = torch.zeros(real_pokemon.size(0), 1, device=device) # All of these are real, so the target is 0.
    real_targets = torch.rand(real_pokemon.size(0), 1, device=device) # * (0.1 - 0) + 0  # Add some noisy labels to make the discriminator think harder.
    real_loss = F.binary_cross_entropy(real_predictions, real_targets)  # Can do binary loss function because it is a binary classifier
    real_score = torch.mean(real_predictions).item()  # How well does the discriminator classify the real pokemon? (Higher score is better for the discriminator)

    # Make some latent tensors to seed the generator
    latent_batch = torch.randn(batch_size, seed_size, 1, 1, device=device)

    # Get some fake pokemon
    fake_pokemon = generator(latent_batch)

    # Train on the generator's current efforts to trick the discriminator
    gen_predictions = discriminator(fake_pokemon)
    # gen_targets = torch.ones(fake_pokemon.size(0), 1, device=device)
    gen_targets = torch.rand(fake_pokemon.size(0), 1, device=device) * (1 - 0.9) + 0.9  # Add some noisy labels to make the discriminator think harder.
    gen_loss = F.binary_cross_entropy(gen_predictions, gen_targets)
    gen_score = torch.mean(gen_predictions).item()  # How well did the discriminator classify the fake pokemon? (Lower score is better for the discriminator)

    # Update the discriminator weights
    total_loss = real_loss + gen_loss
    total_loss.backward()
    disc_optimizer.step()

    return total_loss.item(), real_score, gen_score


def train_generator(discriminator, generator, gen_optimizer, batch_size, seed_size, device):
    # Clear the generator gradients
    gen_optimizer.zero_grad()

    # Generate some fake pokemon
    latent_batch = torch.randn(batch_size, seed_size, 1, 1, device=device)
    fake_pokemon = generator(latent_batch)

    # Test against the discriminator
    disc_predictions = discriminator(fake_pokemon)
    targets = torch.zeros(fake_pokemon.size(0), 1, device=device)  # We want the discriminator to think these images are real.
    loss = F.binary_cross_entropy(disc_predictions, targets)  # How well did the generator do? (How much did the discriminator believe the generator?)

    # Update the generator based on how well it fooled the discriminator
    loss.backward()
    gen_optimizer.step()

    # Return generator loss
    return loss.item()

def train(discriminator, generator, data, epochs, disc_optimizer, gen_optimizer, start_idx=1):
    # Empty the GPU cache to save some memory
    torch.cuda.empty_cache()

    # For results viewing
    fixed_latent_batch = torch.randn(64, config.SEED_SIZE, 1, 1, device=config.DEVICE)

    # Track losses and scores
    disc_losses = []
    disc_scores = []
    gen_losses = []
    gen_scores = []

    # Run the loop
    for epoch in range(epochs):
        # Go through each image
        for real_img in tqdm(data):
            real_img = real_img.to(config.DEVICE) # make sure to send data to gpu
            # Train the discriminator
            disc_loss, real_score, gen_score = train_discriminator(discriminator, generator, real_img, disc_optimizer, batch_size=config.BATCH_SIZE, seed_size=config.SEED_SIZE, device=config.DEVICE)

            # Train the generator
            gen_loss = train_generator(discriminator, generator, gen_optimizer, batch_size=config.BATCH_SIZE, seed_size=config.SEED_SIZE, device=config.DEVICE)

        # Collect results
        disc_losses.append(disc_loss)
        disc_scores.append(real_score)
        gen_losses.append(gen_loss)
        gen_scores.append(gen_score)

        # Print the losses and scores
        print("Epoch [{}/{}], gen_loss: {:.4f}, disc_loss: {:.4f}, real_score: {:.4f}, gen_score: {:.4f}".format(
            epoch + start_idx, epochs, gen_loss, disc_loss, real_score, gen_score))

        # Save the images and show the progress
        save_results(generator, epoch + start_idx, fixed_latent_batch, show=False)

        if config.SAVE_MODEL:
            save_checkpoint(generator, gen_optimizer, filename=config.CHECKPOINT_GEN_Pokemon) # human faces
            save_checkpoint(discriminator, disc_optimizer, filename=config.CHECKPOINT_DISC_Pokemon)

    # Return stats
    return disc_losses, disc_scores, gen_losses, gen_scores

def main():
    discriminator = Discriminator().to(config.DEVICE)
    generator = Generator(config.SEED_SIZE).to(config.DEVICE)

    # Create the optimizers
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.9))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.9))

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_Pokemon,
            generator,
            gen_optimizer,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_Pokemon,
            discriminator,
            disc_optimizer,
            config.LEARNING_RATE,
        )

    dset1 = CustomDataset("data/train/logos_dataset", transform=config.transforms1)
    dset2 = CustomDataset("data/train/logos_dataset", transform=config.transforms2)
    dsetlist = [dset1, dset2]
    dataset = torch.utils.data.ConcatDataset(dsetlist)

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    results = train(discriminator, generator, loader, config.NUM_EPOCHS, disc_optimizer, gen_optimizer)

    disc_losses, disc_scores, gen_losses, gen_scores = results

    plt.plot(disc_losses, '-')
    plt.plot(gen_losses, '-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses')
    plt.show()

if __name__ == "__main__":
    main()