import os
import torch
import time
import numpy as np
import torch.nn as nn


class GeneratorBase(nn.Module):
    def __init__(self):
        super(GeneratorBase, self).__init__()

    def forward(self, *input):
        return NotImplementedError

    def __str__(self):
        return NotImplementedError

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location={'cuda:0': 'cpu'}))


def total_params(model):
    total_parameters = sum(p.numel() for p in model.parameters())
    train_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'total parameters: ' + str(total_parameters) + '\n' \
           + 'train parameters: ' + str(train_parameters) + '\n'


class Generator(GeneratorBase):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()

        self.img_shape = img_shape
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]

            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            *block(2048, 4096),
            nn.Linear(4096, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

    def infer(self, z):
        self.model.eval()
        return self.model(z)

    def __str__(self):
        return 'Generator::' + str(self.model)
    