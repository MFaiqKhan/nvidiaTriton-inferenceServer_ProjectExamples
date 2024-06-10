# Triton Server Image Model Deployment Documentation

## Overview
This project demonstrates the deployment of a machine learning model using NVIDIA Triton Inference Server on an RTX 3060 12GB GPU. It includes setting up the Triton server, preparing the model for deployment, making inference requests, and preprocessing images.

## Prerequisites
- Anaconda(For Virtual Environment, Install cuda and cudnn from conda)
- Docker installed on your system
- NVIDIA Docker support (check for nvidia driver and docker version compatibility , sometimes cuda dont work)
- Python 3.11
- `tritonclient` Python package
- PIL (Python Imaging Library)
- `scipy`

## Installation
1. **Install Docker**: Follow the official Docker installation guide to install Docker on your system.
2. **Install NVIDIA Triton Server**: Pull the Triton Server Docker image using:
   ```sh
   docker pull nvcr.io/nvidia/tritonserver:22.04-py3
   ```
3. **Clone the Project Repository**: Clone the project repository from GitHub to your local machine.
4. **Download the MobileNetV2 Model**:
   ```sh
   wget -O mobilenetv2-12.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx?download=
   mv mobilenetv2-12.onnx triton_sample_project/model_repo/mobilenet/1/mobilenetv2.onnx
   ```
   Or using `curl` in CMD:
   ```cmd
   curl -L -o mobilenetv2-12.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx?download=
   move mobilenetv2-12.onnx triton_sample_project\model_repo\mobilenet\1\mobilenetv2.onnx
   ```

## Model Configuration
Prepare the `config.pbtxt` file, which defines the model configuration for Triton Inference Server.

### `config.pbtxt` File Explanation
The `config.pbtxt` file specifies the configuration for the model, including the input and output parameters, version policy, instance group, and optimization settings. Below is an example configuration for the MobileNetV2 model:

```plaintext
name: "mobilenetv2-12"
backend: "onnxruntime"
max_batch_size: 0

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 1, 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, 1000]
  }
]

default_model_filename:"mobilenetv2-12.onnx"
```

## Docker Compose Setup
Create a `docker-compose.yml` file to manage the Triton Server container with Docker Compose.

### `docker-compose.yml` File
```yaml
version: '3.4'

services:
  triton_server:
    container_name: sample-tis-22.04
    image: nvcr.io/nvidia/tritonserver:22.04-py3
    privileged: true
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             capabilities: [gpu]
    volumes:
      - ${MODEL_REPOSITORY}:/models
    command: ["tritonserver", "--model-repository=/models"]
```

### `.env` File
Create a `.env` file to store environment variables for Docker Compose.
```plaintext
HTTP_P=8000:8000
GRPC_P=8001:8001
PROM_P=8002:8002
IMAGE=nvcr.io/nvidia/tritonserver:22.04-py3
MODEL_REPOSITORY=/e/Github/triton_server_mobilenetv2/models 
```
Check your absolute path where your models are deployed , otherwise triton server won't detect your models, sometimes Disk Letter is written in lower form for it to work 
for Windows use forward slashes when writing paths.

### Running the Docker Compose
Start the Triton Server using Docker Compose:
```sh
docker-compose -f docker-compose.yml up -d
```

### Checking Logs
Verify that the Triton Server is running correctly by checking the logs:
```sh
docker logs sample-tis-22.04
```
You should see output indicating successful connection to the ports:
```plaintext
I0610 19:53:14.057147 1 grpc_server.cc:4544] Started GRPCInferenceService at 0.0.0.0:8001
I0610 19:53:14.057511 1 http_server.cc:3242] Started HTTPService at 0.0.0.0:8000
I0610 19:53:14.100178 1 http_server.cc:180] Started Metrics Service at 0.0.0.0:8002
```

Plus you should see your Model name, version and status of your model in a table like this : 

```
+----------------+---------+--------+
| Model          | Version | Status |
+----------------+---------+--------+
| mobilenetv2-12 | 1       | READY  |
+----------------+---------+--------+
```

and docker should recognize your GPU Device for metrics 

Output will be shown like this in the log if it detects your GPU Correctly:
```
Collecting metrics for GPU 0: NVIDIA GeForce RTX 3060
```

Now Your Triton Server is Running as a service in Docker.
We can check if it works correctly by sending inference requests to the server, for this we will create a script.

## Inference Script: `playground.py`
The `playground.py` script preprocesses images and sends inference requests to the Triton Server.

### `playground.py` File Explanation
```python
import tritonclient.http as httpclient
import numpy as np
from PIL import Image
from scipy.special import softmax

# Image Processing Code
def resize_image(image_path, min_length):
    image = Image.open(image_path)
    scale_ratio = min_length / min(image.size)
    new_size = tuple(int(round(dim * scale_ratio)) for dim in image.size)
    resized_image = image.resize(new_size, Image.BILINEAR)
    return np.array(resized_image)

def crop_center(image_array, crop_width, crop_height):
    height, width, _ = image_array.shape
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2
    return image_array[start_y : start_y + crop_height, start_x : start_x + crop_width]

def normalize_image(image_array):
    image_array = image_array.transpose(2, 0, 1).astype("float32")
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    normalized_image = (image_array / 255 - mean_vec[:, None, None]) / stddev_vec[:, None, None]
    return normalized_image.reshape(1, 3, 224, 224)

def preprocess(image_path):
    image = resize_image(image_path, 256)
    image = crop_center(image, 224, 224)
    image = normalize_image(image)
    image = image.astype(np.float32)
    return image

# Load classes
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Define model parameters
image_path = "./pizzaa.png"
model_input_name = "input"
model_output_name = "output"
model_name = "mobilenetv2-12"
model_vers = "1"
server_url = "localhost:8000"

# Preprocess the image
processed_image = preprocess(image_path)

# Define the Client connection
client = httpclient.InferenceServerClient(url=server_url)

# Define the input tensor placeholder
input_data = httpclient.InferInput(model_input_name, processed_image.shape, "FP32")

# Populate the tensor with data
input_data.set_data_from_numpy(processed_image)

# Send Inference Request
request = client.infer(model_name, model_version=model_vers, inputs=[input_data])

# Unpack the output layer as numpy
output = request.as_numpy(model_output_name)

# Flatten the values
output = np.squeeze(output)

# Apply softmax for image classification
probabilities = softmax(output)

# Get Top 5 prediction labels
top5_class_ids = np.argsort(probabilities)[-5:][::-1]

# Pretty print the results
print("\nInference outputs (TOP 5):")
print("=========================")
padding_str_width = 10
for class_id in top5_class_ids:
    score = probabilities[class_id]
    print(f"CLASS: [{categories[class_id]:<{padding_str_width}}]\t: SCORE [{score*100:.2f}%]")

# Finish..
```

### Explanation
1. **Image Processing**:
   - `resize_image`: Resizes the image while maintaining the aspect ratio.
   - `crop_center`: Crops the image to the center with the specified dimensions.
   - `normalize_image`: Normalizes the image using mean and standard deviation vectors.
   - `preprocess`: Combines the above steps to preprocess the image.

2. **Inference**:
   - Loads ImageNet class labels.
   - Defines model parameters including image path, model name, version, and server URL.
   - Preprocesses the image.
   - Sets up the Triton client connection.
   - Defines and populates the input tensor.
   - Sends the inference request to the Triton Server.
   - Unpacks and processes the server response.
   - Applies softmax to the output for classification.
   - Retrieves and prints the top 5 prediction labels and their scores.

**The Output We Get By Running The Script**:
```
(nvdiaTritonServer_env) E:\Github\triton_server_ImageModel>python playground.py

Inference outputs (TOP5):
=========================
CLASS: [pizza     ]     : SCORE [99.96%]
CLASS: [frying pan]     : SCORE [0.01%]
CLASS: [plate     ]     : SCORE [0.01%]
CLASS: [king crab ]     : SCORE [0.00%]
CLASS: [hotdog    ]     : SCORE [0.00%]
```

## Conclusion
This documentation provides a comprehensive guide to deploying a machine learning model using NVIDIA Triton Inference Server. It covers downloading and configuring the model, setting up and running the server, and making inference requests with preprocessing. By following these steps, you can effectively deploy and utilize machine learning models with Triton Server on an RTX 3060 12GB GPU.