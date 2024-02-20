import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time

def load_model():
    # Load a pretrained ResNet-18 model
    model = models.resnet18(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path):
    # Preprocess an input image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = Image.open(image_path)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    return input_batch

def benchmark_inference(model, inputs, num_runs=100):
    # Benchmark the inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(inputs)
    total_time = time.time() - start_time
    inferences_per_second = num_runs / total_time
    return inferences_per_second

def main():
    image_path = 'path/to/your/image.jpg'  # Update this path
    model = load_model()
    inputs = preprocess_image(image_path)
    
    # Move model to CPU (it's on CPU by default if CUDA is not available)
    model.to('cpu')
    inputs = inputs.to('cpu')
    
    num_runs = 100  # Number of runs to compute
    inferences_per_second = benchmark_inference(model, inputs, num_runs)
    print(f"Inferences per second (CPU): {inferences_per_second}")

if __name__ == "__main__":
    main()
