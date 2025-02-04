import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import ConvNet


# Load the model
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)
model.load_state_dict(torch.load("output/mnist_cnn.pt", map_location=device))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    print(image.shape)
    with torch.no_grad():
        outputs = model(image)
        return outputs.shape


if __name__ == "__main__":
    image_path = 'test/2.png'
    prediction = predict(image_path)
    print(f"Predicted digit: {prediction}")