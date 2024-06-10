import tritonclient.http as httpclient
import numpy as np
from PIL import Image
from scipy.special import softmax


## Image Processing Code:Code for resizing the image, 
# equally crop from the center and normalize the pixels with MEAN and STD from 
# ImageNet Dataset*

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

# Inferencing Request:

# Preprocess the image
processed_image = preprocess(image_path)

# Define the Client connection
client = httpclient.InferenceServerClient(url=server_url)

# Define the input tensor placeholder
input_data = httpclient.InferInput(model_input_name, processed_image.shape, "FP32")

# Populdate the tensor with data
input_data.set_data_from_numpy(processed_image)

# Send Inference Request
request = client.infer(model_name, model_version=model_vers, inputs=[input_data])

# Unpack the output layer as numpy
output = request.as_numpy(model_output_name)

# Flatten the values
output = np.squeeze(output)

# Since it's image classification, apply softmax
probabilities = softmax(output)

# Get Top5 prediction labels
top5_class_ids = np.argsort(probabilities)[-5:][::-1]

# Pretty print the results
print("\nInference outputs (TOP5):")
print("=========================")
padding_str_width = 10
for class_id in top5_class_ids:
    score = probabilities[class_id]
for class_id in top5_class_ids:
    score = probabilities[class_id]
    print(
        f"CLASS: [{categories[class_id]:<{padding_str_width}}]\t: SCORE [{score*100:.2f}%]"
    )
