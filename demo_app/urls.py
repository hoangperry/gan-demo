# accounts/urls.py
from django.urls import path
from .views import gan_demo, VerifyView

urlpatterns = [
    path('verify/', VerifyView.as_view(), name='verify_view'),
    path('', gan_demo, name='index'),
]