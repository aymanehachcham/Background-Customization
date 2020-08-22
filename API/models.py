from django.db import models

from django.db import models
from API.utils import get_input_image_path, get_output_image_path
import uuid


# Create your models here.

class ImageSegmentation(models.Model):
    uuid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, null=True, blank=True)
    input_image = models.FileField(upload_to=get_input_image_path, null=True, blank=True)
    output_image = models.FileField(upload_to=get_output_image_path, null=True, blank=True)
    verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return "%s" % self.name


class Image(models.Model):
    property_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.FileField(upload_to=get_input_image_path, null=True, blank=True)

