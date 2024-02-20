
This Setup is to calculate Inference per minute for CPU using Pytorch and Benchmark Suite.

I have used Ubuntu machines.

# Steps to Install Pytorch 

# 1. Install pytorch, torchvision, and torchaudio using pip3

Run below command to install dependencies.
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```
# 2. Install the benchmark suite

git clone https://github.com/pytorch/benchmark
cd benchmark
python3 install.py

# 3. Download Sample Image file and update the image path in inference-per-minute.py file.

# 4. Run the inference-per-minute.py file
```
python3 inference-per-minute.py
```

# Code Explaination.

The script provided is a Python program designed to benchmark the performance of a pretrained ResNet-18 model from PyTorch's torchvision models on the CPU. The benchmarking focuses on calculating how many inferences (predictions) per second the model can perform. Here’s a step-by-step explanation of what each part of the script does:

# 1. Import Necessary Libraries

```
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time

```
torch: The main PyTorch library for deep learning.
torchvision.models: Contains definitions for popular models like ResNet-18.
torchvision.transforms: Utilities for transforming input data (images) into a format the model expects.
PIL.Image: Used for opening and manipulating images. PIL (Python Imaging Library) is part of the Pillow library, an image processing library.
time: Used for measuring execution time to benchmark performance.

# 2. Define Functions for Loading the Model and Preprocessing Images
   
load_model Function
```
def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model
```

Loads a pretrained ResNet-18 model.

Sets the model to evaluation mode with model.eval(), which is necessary for inference since it tells the model that it won't be trained further. This disables dropout and batch normalization layers to behave differently during inference.
preprocess_image Function

```
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = Image.open(image_path)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch
```

Defines a series of transformations to prepare the input image for the model. This includes resizing, center cropping, converting to a tensor, and normalizing with preset mean and standard deviation values (these are standard for models trained on the ImageNet dataset).

input_tensor.unsqueeze(0) adds a batch dimension to the tensor, converting it from a 3D tensor to a 4D tensor, as PyTorch models expect a batch dimension.

# 3. Benchmark Inference Function

```
def benchmark_inference(model, inputs, num_runs=100):
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(inputs)
    total_time = time.time() - start_time
    inferences_per_second = num_runs / total_time
    return inferences_per_second
```

Measures the total time taken to perform a fixed number of inferences (num_runs).
torch.no_grad() is used to disable gradient calculation, which is not needed during inference and can save memory and computation.

The total time for all inferences is calculated, and then the number of inferences per second is computed by dividing the total number of runs by the total time taken.

# 4. The Main Function
```
def main():
    image_path = 'path/to/your/image.jpg'
    model = load_model()
    inputs = preprocess_image(image_path)
    model.to('cpu')
    inputs = inputs.to('cpu')
    num_runs = 100
    inferences_per_second = benchmark_inference(model, inputs, num_runs)
    print(f"Inferences per second (CPU): {inferences_per_second}")
```

Specifies the path to the input image. Loads the model and preprocesses the input image. Ensures both the model and inputs are moved to the CPU (this is the default but is explicitly stated for clarity). Calls the benchmark_inference function with the model, preprocessed inputs, and the number of runs to calculate the inferences per second. Prints the result, showing how many inferences per second the model can perform on the CPU.

# 5. Running the Main Function

```
if __name__ == "__main__":
    main()
```
This conditional is used to execute the main function when the script is run directly (as opposed to being imported as a module in another script).

# Note:

Remember to replace 'path/to/your/image.jpg' with the actual path to an image file you want to use for testing. This script provides a simple yet effective way to benchmark the inference performance of a convolutional neural network model on the CPU, which is essential for evaluating deployment scenarios and hardware requirements.
