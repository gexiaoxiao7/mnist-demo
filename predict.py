import torch
from torchvision import transforms
from PIL import Image
from model import ConvNet
import platform

if platform.system() == "Darwin": # use gpu on mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else: # use gpu on linux / windows
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvNet().to(device)
model.load_state_dict(torch.load("output/ConvModel.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

if __name__ == "__main__":
    image_path = 'test/0.jpg'
    prediction = predict(image_path)
    print(f"Predicted digit: {prediction}")