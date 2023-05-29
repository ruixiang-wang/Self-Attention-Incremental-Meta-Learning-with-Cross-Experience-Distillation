import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_grad_cam(model, image, target_class):
    model.eval()
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    features = model.features(image_tensor)
    output = model(image_tensor)
    predicted_class = torch.argmax(output, dim=1).item()
    model.zero_grad()
    output[0, target_class].backward()
    gradients = model.features.get_gradients()
    weights = torch.mean(gradients, dim=(2, 3))
    cam = torch.sum(features * weights.unsqueeze(-1).unsqueeze(-1), dim=1)
    cam = F.relu(cam)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)
    cam = F.interpolate(cam.unsqueeze(0), size=image.shape[:2], mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().numpy()

    return cam, predicted_class

image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255.0

target_class = 7

cam, predicted_class = get_grad_cam(model, image, target_class)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cam, cmap='jet')
plt.title('Grad-CAM')
plt.axis('off')

plt.tight_layout()
plt.show()
