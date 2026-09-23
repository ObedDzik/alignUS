import PIL 
#import cv2 
import numpy as np
import torch 
from PIL import Image
import skimage.transform


class MicroSegNetInference: 
    def __init__(self, model, device='cpu'): 
        self.model = model 
        self.model.eval()
        self.device = device
        self.model.to(device)

    def __call__(self, image: Image.Image): 
        size = image.size
        image = np.array(image)
        image = image / 255 
        image = skimage.transform.resize(image, (224, 224), anti_aliasing=True)        
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(self.device)
        outputs = self.model(input).sigmoid() 
        outputs = (outputs[0][0] > 0.5).numpy().astype('uint8')
        return Image.fromarray(outputs).convert('L').resize(size, Image.Resampling.NEAREST)