from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
import os
import time


class VerifyView(APIView):
    @staticmethod
    def get(request, *args, **kwargs):
        return Response('Server is running', status=status.HTTP_200_OK)

def gan_demo(request):
    res = {'test': '1234'}
    return render(request, 'index.html', res)
