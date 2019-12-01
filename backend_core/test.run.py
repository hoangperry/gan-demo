import os
import torch
import time
import numpy as np
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from cifar import Generator
# from mnist import Generator
#
# gen = Generator()
# gen.load('mnist_g.dg')
# z = Variable(torch.Tensor(np.random.normal(0, 1, (1, 100))))
# img = gen.infer(z)
#
#
# save_image(img.view(1, 28, 28).data[:25], 'd.png',  nrow=5, normalize=True)
# img = img.data.numpy().reshape((28, 28))
# cv2.imwrite('abc.png', img)

G = Generator().to('cpu')