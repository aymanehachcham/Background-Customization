
from django.urls import include, path
from rest_framework import routers
from . import views

app_name = 'api'

urlpatterns = [
    path(r'test/', views.test_api, name='test_api_communication'),
    path(r'images/', views.get_images, name='get_images'),
    path(r'inference/', views.run_inference, name='run_inference_on_images'),
    path(r'grayscale/', views.run_grayscale_inference, name='run_grayscale_inference_on_images'),
    path(r'clean/', views.clean_folders, name='clean_output_folder')
]