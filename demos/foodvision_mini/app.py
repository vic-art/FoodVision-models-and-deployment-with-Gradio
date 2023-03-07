
### 1. Imports and class names setup
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names 
class_names = ['pizza', 'steak', 'sushi']

### 2. Model and transforms preparation
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes = 3
)

# Load saved weights
effnetb2.load_state_dict(
    torch.load(f = 'pretrained_effnetb2_feature_exractor_pizza_steak_sushi_20_percent.pth',
               map_location = torch.device("cpu")) # load the model to the CPU regardless of the device it trained on (we won't necessarily have a GPU when we deploy)
)

### 3. Predict function (predict())
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
  # Start a timer
  start_time = timer()

  # Transform the input image for use with EffNetB2
  img = effnetb2_transforms(img).unsqueeze(0) # add batch dimension on 0th index

  # Put model into eval mode, make predictions 
  effnetb2.eval()
  with torch.inference_mode():
    # Pass the transformed image through the model and turn the prediction logits into probabilities
    pred_probs = torch.softmax(effnetb2(img), dim=1)

  # Create a prediction label and prediction probability dictionary
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  # Calculate the pred time
  end_time = timer()
  pred_time = round(end_time-start_time, 4)

  # Return pred dict and pred time
  return pred_labels_and_probs, pred_time

### 4. Gradio app - our Gradio interface + launch command

# Create title, description and article
title = "FoodVision Mini 🍕🥩🍣"
description = "An [EfficientNetB2 feature extractor](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2) computer vision model to classify images as pizza, steak or sushi."
article = "Created at [PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/#74-building-a-gradio-interface)"

# Create example list
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(
    fn = predict, 
    inputs = gr.Image(type='pil'),
    outputs = [gr.Label(num_top_classes=3, label ="Predictions"),
               gr.Number(label ='Prediction time (s)')], 
               examples=example_list,
               title=title,
               description=description,
               article=article)               
# Launch the demo
demo.launch()
