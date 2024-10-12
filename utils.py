import random, torch, os, numpy as np
import config
from torchvision.utils import save_image

RESULTS_DIR = 'saved_images'
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=1234):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(generator, index, latent_batch, show=True):
    # Generate fake logo
    fake_pokemon = generator(latent_batch)

    # Make the filename for the output
    fake_file = "result-image-{0:0=4d}.png".format(index)

    # Save the image
    save_image(fake_pokemon, os.path.join(RESULTS_DIR, fake_file), nrow=8)
    print("Result Saved!")

    # if show:
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     ax.set_xticks([]);
    #     ax.set_yticks([])
    #     ax.imshow(make_grid(fake_pokemon.cpu().detach(), nrow=8).permute(1, 2, 0))