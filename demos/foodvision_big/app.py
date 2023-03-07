### Imports and class names setup ###
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names 
with open('class_names.txt', 'r') as f:
  class_names = [food_name.strip() for food_name in f.readlines()]

### 2. Model and transform preparation ###
# Create model and transforms
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes = 101)

# Load saved weights 
effnetb2.load_state_dict(
    torch.load(f='pretrained_effnetb2_feature_extractor_food101_20_precent.pth',
               map_location =torch.device("cpu")) #load to CPU
)

### 3. Predict function ###

def predict(img) -> Tuple[Dict, float]:
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
title = "FoodVision Big 🍕👁"
description = "An [EfficientNetB2 feature extractor](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2) computer vision model to classify images of [101 classes of food from Food101 dataset](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/food101_class_names.txt)."
article = "Created at [PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/#11-turning-our-foodvision-big-model-into-a-deployable-app)"

# Create example list
example_list = [["examples/" + example] for example in os.listdir("examples")]


# Create the Gradio demo
demo = gr.Interface(
    fn = predict, 
    inputs = gr.Image(type='pil'),
    outputs = [gr.Label(num_top_classes=5, label ="Predictions"),
               gr.Number(label ='Prediction time (s)')], 
               examples=example_list,
               title=title,
               description=description,
               article=article)               
# Launch the demo!
demo.launch()
