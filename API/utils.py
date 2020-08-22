
import os
import uuid
from PIL import Image
import numpy as np
from SemanticModels.Dataset.semantic_data import SegmentationSample
from SemanticModels.DeepLabV3.deeplab_implementation import SemanticSeg

def get_input_image_path(instance, filename):
    _, ext = os.path.splitext(filename)
    return 'Media/Input_image/{}{}'.format(uuid.uuid4(), ext)

def get_output_image_path(instance, filename):
    _, ext = os.path.splitext(filename)
    return 'Media/Output_image/{}{}'.format(uuid.uuid4(), ext)

def modify_input_for_multiple_files(property_id, image):
    dict = {}
    dict['property_id'] = property_id
    dict['image'] = image
    return dict

class RunDeepLabInference():
    def __init__(self, image_file, background_file=None):
        self.file_image = image_file

        if background_file is not None:
            self.background_file = background_file
            self.background_base_path, self.background_filename = os.path.split(self.background_file.input_image.path)
            self.background_sample_image = SegmentationSample(root_dir=self.background_base_path,image_file=self.background_filename, device='cuda')

        self.output_folder = 'Media/Output_image/'
        self.foreground_base_path, self.foreground_filename = os.path.split(self.file_image.input_image.path)
        self.foreground_sample_image = SegmentationSample(root_dir=self.foreground_base_path, image_file=self.foreground_filename, device='cuda')
        self.model = SemanticSeg(pretrained=True, device='cuda')

    def save_bg_custom_output(self):
        res = self.model.run_inference(self.foreground_sample_image, self.background_sample_image)
        image_to_array = Image.fromarray((res * 255).astype(np.uint8))
        image_to_array.save(self.output_folder + self.foreground_filename)
        self.file_image.output_image = self.output_folder + self.foreground_filename
        self.file_image.save()

    def save_grayscale_output(self):
        res = self.model.run_grayscale_inference(self.foreground_sample_image)
        image_to_array = Image.fromarray((res * 255).astype(np.uint8))
        image_to_array.save(self.output_folder + self.foreground_filename)
        self.file_image.output_image = self.output_folder + self.foreground_filename
        self.file_image.save()