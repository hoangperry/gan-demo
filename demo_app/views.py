from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from torch.autograd import Variable
from torchvision.utils import save_image
from backend_core.mnist import Generator
import os
import torch
import time
import numpy as np


gen = Generator()
gen.load('./backend_core/mnist_g.dg')


class VerifyView(APIView):
    @staticmethod
    def get(request, *args, **kwargs):
        return Response('Server is running', status=status.HTTP_200_OK)


def gan_demo(request):
    if request.method == 'POST':
        z = Variable(torch.Tensor(np.random.normal(0, 1, (1, 100))))
        session_name = str(time.time()).replace('.', '_')
        img = gen.infer(z)
        img_dir = 'static/img/' + session_name + '_mnist.png'
        print(img_dir)
        save_image(img.view(1, 28, 28).data[:25], img_dir, nrow=5, normalize=True)
        return render(request, 'index.html', {'imgdir': "/".join(img_dir.split('/')[1:])})

    return render(request, 'index.html')
