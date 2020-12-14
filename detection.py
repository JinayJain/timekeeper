# Detecting clock bounding boxes within an image using PyTorch

import torch
import torchvision
import numpy as np

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = "cpu"

if torch.cuda.is_available():
    model.cuda()
    device = torch.device("cuda")

def preprocess(image):
    '''Preprocesses images for detection with the object detector'''

    image = image.copy()
    image = image.transpose(2, 0, 1)
    image = image / 255.

    return image



def detect_clock(image):
    '''Detects a clock within a given image and returns the bounding box'''


    inp = [torch.from_numpy(preprocess(image)).float().to(device)]
    preds = model(inp)[0]

    boxes = preds['boxes'].detach()
    labels = preds['labels']
    scores = preds['scores']

    for i in range(len(labels)):
        label = labels[i].item()

        if label == 85: # the COCO id for clocks is 85
            return boxes[i].cpu().numpy().round().astype(np.uint16)

    return None # no clock was found

