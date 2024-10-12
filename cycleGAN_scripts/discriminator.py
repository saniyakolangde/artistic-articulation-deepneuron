import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride): # again, details about discriminator architecture in CycleGan paper
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        return self.conv(input)

class Discriminator(nn.Module): # following the discriminator architecture from CycleGan paper
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512]): # pretty basic initialisation
        super().__init__() # make sure to call nn.Module's init

        # define our initial layer
        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=4,stride=2,padding=1,padding_mode="reflect"),  # initial layer in CycleGan does not use instance normalisation
            nn.LeakyReLU(0.2, inplace=True), # discriminator uses leaky relu activation function with slope of 0.2
        )

        model_layers = []
        in_channels = hidden_dims[0] # we will now construct sequential layers with hidden_dim dimensions
        for hidden_dim in hidden_dims[1:]: # skip first hidden_dim as that is covered in initial_layer
            if hidden_dim != hidden_dims[-1]: # if not last layer, stride = 2
                model_layers.append(ConvBlock(in_channels, hidden_dim, stride=2))
            else: # if last layer, stride = 1
                model_layers.append(ConvBlock(in_channels, hidden_dim, stride=1))
            in_channels = hidden_dim # move to next layer

        # final conv layer to map discriminator output to [0,1] for a fake/real image label
        model_layers.append(nn.Conv2d(in_channels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))

        # define model
        self.model = nn.Sequential(*model_layers) # unpack model layers into sequential

    # def block(self, in_channels, out_channels, stride): # again, details about discriminator architecture in CycleGan paper
    #     return nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode="reflect"),
    #         nn.InstanceNorm2d(out_channels),
    #         nn.LeakyReLU(0.2),
    #     )

    def forward(self, input): # input is data
        x = self.initial_layer(input) # pass to initial layer first
        return torch.sigmoid(self.model(x)) # pass through rest of model next, output will be passed through sigmoid (want [0,1] for fake/real labels)

# def test():
#     x = torch.randn((5, 3, 256, 256))
#     model = Discriminator(in_channels=3)
#     preds = model(x)
#     print(preds.shape)
#
# if __name__ == "__main__":
#     test()