
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch

class SegmentationSample(Dataset):

    def __init__(self, root_dir, image_file, device):
        # Process the image input file
        self.image_file = os.path.join(root_dir, image_file)
        self.image = Image.open(self.image_file)

        # Asses the internal device: cpu or gpu:
        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        if device == 'cpu':
            self.device = 'cpu'

        # Define the Transforms applied to the input image:
        self.preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.unload_tensor = transforms.ToPILImage()

        # Output returned: processed tensor with an additional dim for the batch:
        self.processed_image = self.preprocessing(self.image)
        self.processed_image = self.processed_image.unsqueeze(0).to(self.device)

    def __getitem__(self, item):
        return self.processed_image

    def print_image(self, title=None):
        image = self.image
        if title is not None:
            plt.title = title
        plt.imshow(image)
        plt.pause(5)
        plt.figure()

    def print_processed(self, title='After processing'):
        image = self.processed_image.squeeze(0).detach().cpu()
        image = self.unload_tensor(image)
        plt.title = title
        plt.imshow(image)
        plt.pause(5)
        plt.figure()