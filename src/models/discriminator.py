import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator: Classifies each 70x70 patch as Real or Fake.
    Takes (Original Image + Redesigned Image) or (Original Image + Target Image).
    """
    def __init__(self, in_channels=6): # Combined channels (3 from input, 3 from target/gen)
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_input, img_gen):
        # Concatenate input image and generated image along channels
        img_input = torch.cat((img_input, img_gen), 1)
        return self.model(img_input)

if __name__ == "__main__":
    # Test with 6-channel input (RGB + RGB)
    disc = Discriminator()
    dummy_input = torch.randn(1, 3, 256, 256)
    dummy_gen = torch.randn(1, 3, 256, 256)
    output = disc(dummy_input, dummy_gen)
    print(f"Discriminator Input: {dummy_input.shape}, Output: {output.shape}")
