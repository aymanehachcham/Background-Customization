
from rest_framework import serializers
from .models import ImageSegmentation, Image


class InputImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageSegmentation
        fields = ('uuid', 'name', )

class OutputImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageSegmentation
        fields = ('uuid', 'name', 'input_image', 'output_image', 'created_at', 'updated_at')

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ('property_id', 'image')