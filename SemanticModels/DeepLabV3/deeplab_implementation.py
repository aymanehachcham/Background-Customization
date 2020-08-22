import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models.segmentation as models
from SemanticModels.Dataset.semantic_data import SegmentationSample

class SemanticSeg(nn.Module):
    def __init__(self, pretrained: bool, device):
        super(SemanticSeg, self).__init__()
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        if device == 'cpu':
            self.device = 'cpu'

        self.model = self.load_model(pretrained)

    def __getitem__(self, item):
        return self.model

    # Add the Backbone option in the parameters
    def load_model(self, pretrained=False):
        if pretrained:
            model = models.deeplabv3_resnet101(pretrained=True)
        else:
            model = models.deeplabv3_resnet101()

        model.to(self.device)
        model.eval()
        return model

    def run_inference(self, image_foreground: SegmentationSample, image_background: SegmentationSample):
        # Run the model in the respective device:
        with torch.no_grad():
            output = self.model(image_foreground.processed_image)['out']

        reshaped_output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        return self.background_custom(reshaped_output, image_foreground.image_file, image_background.image_file)

    def run_grayscale_inference(self, image_foreground: SegmentationSample):
        # Run the model in the respective device:
        with torch.no_grad():
            output = self.model(image_foreground.processed_image)['out']

        reshaped_output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        return self.grayscale_background(reshaped_output, image_foreground.image_file)


    def background_custom(self, input_image, source, background_source,number_channels=21):

        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        # Defining empty matrices for rgb tensors:
        r = np.zeros_like(input_image).astype(np.uint8)
        g = np.zeros_like(input_image).astype(np.uint8)
        b = np.zeros_like(input_image).astype(np.uint8)

        for l in range(0, number_channels):
            if l == 15:
                idx = input_image == l
                r[idx] = label_colors[l, 0]
                g[idx] = label_colors[l, 1]
                b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        # return rgb

        # and resize image to match shape of R-band in RGB output map
        foreground = cv2.imread(source)
        foreground = cv2.resize(foreground, (r.shape[1], r.shape[0]))
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)

        background = cv2.imread(background_source, cv2.IMREAD_COLOR)
        background = cv2.resize(background, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        # Create a binary mask using the threshold
        th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)

        # Convert uint8 to float
        foreground = foreground.astype(float)
        background = background.astype(float)
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = alpha.astype(float) / 255
        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(alpha, foreground)
        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - alpha, background)
        # Add the masked foreground and background.
        outImage = cv2.add(foreground, background)

        return outImage / 255

    def grayscale_background(self, input_image, source, number_channels=21):

        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        # Defining empty matrices for rgb tensors:
        r = np.zeros_like(input_image).astype(np.uint8)
        g = np.zeros_like(input_image).astype(np.uint8)
        b = np.zeros_like(input_image).astype(np.uint8)

        for l in range(0, number_channels):
            if l == 15:
                idx = input_image == l
                r[idx] = label_colors[l, 0]
                g[idx] = label_colors[l, 1]
                b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        # return rgb

        # and resize image to match shape of R-band in RGB output map
        foreground = cv2.imread(source)
        foreground = cv2.resize(foreground, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)

        # Create a background image by copying foreground and converting into grayscale
        background = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

        # Convert uint8 to float
        foreground = foreground.astype(float)
        background = background.astype(float)

        # Create a binary mask of the RGB output map using the threshold value 0
        th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)
        alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
        alpha = alpha.astype(float) / 255

        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)

        # Add the masked foreground and background
        outImage = cv2.add(foreground, background)

        return outImage / 255