from django.shortcuts import render

from django.views.decorators.cache import never_cache
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import ImageSegmentation
from .utils import *
import shutil
from .serializers import OutputImageSerializer, ImageSerializer


@api_view(['GET'])
@never_cache
def test_api(request):
    return Response({'response':"You are successfully connected to the Semantic Segmentation API"})


@api_view(['POST'])
@never_cache
def run_grayscale_inference(request):
    file_ = request.FILES['image']
    image = ImageSegmentation.objects.create(input_image=file_, name='image_%02d' % uuid.uuid1())
    RunDeepLabInference(image).save_grayscale_output()
    serializer = OutputImageSerializer(image)
    return Response(serializer.data)


@api_view(['POST'])
@never_cache
def run_inference(request):
    property_id = request.POST.get('property_id')

    # converts querydict to original dict
    images = dict((request.data).lists())['image']
    flag = 1
    arr = []
    for img_name in images:
        modified_data = modify_input_for_multiple_files(property_id,
                                                        img_name)
        file_serializer = ImageSerializer(data=modified_data)
        if file_serializer.is_valid():
            file_serializer.save()
            arr.append(file_serializer.data)
        else:
            flag = 0

    if flag == 1:
        image_path = os.path.relpath(arr[0]['image'], '/')
        bg_path = os.path.relpath(arr[1]['image'], '/')
        input_image = ImageSegmentation.objects.create(input_image=image_path, name='image_%02d' % uuid.uuid1())
        bg_image = ImageSegmentation.objects.create(input_image=bg_path, name='image_%02d' % uuid.uuid1())
        RunDeepLabInference(input_image, bg_image).save_bg_custom_output()
        serializer = OutputImageSerializer(input_image)
        return Response(serializer.data)


@api_view(['POST'])
@never_cache
def get_images(request):
    property_id = request.POST.get('property_id')

    # converts querydict to original dict
    images = dict((request.data).lists())['image']
    flag = 1
    arr = []
    for img_name in images:
        modified_data = modify_input_for_multiple_files(property_id,
                                                        img_name)
        file_serializer = ImageSerializer(data=modified_data)
        if file_serializer.is_valid():
            file_serializer.save()
            arr.append(file_serializer.data)
        else:
            flag = 0

    if flag == 1:
        return Response(arr, status=status.HTTP_201_CREATED)
    else:
        return Response(arr, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@never_cache
def clean_folders(request):
    folder_input = 'Media/Input_image/'
    for filename in os.listdir(folder_input):
        file_path = os.path.join(folder_input, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    folder_output = 'Media/Output_image/'
    for filename in os.listdir(folder_output):
        file_path = os.path.join(folder_output, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    return Response({'response': "Media folders were cleaned up!!"})
