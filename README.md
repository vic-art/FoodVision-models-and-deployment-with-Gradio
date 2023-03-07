# FoodVision-models-and-deployment-with-Gradio

The task was to create 2 models for food classification present on a given image and deploy them with Gradio App:

* FoodVision Mini was trained to classify 3 classes of food, including pizza, steak and sushi.
The dataset we're going to use for training a FoodVision Mini model is a sample of pizza, steak and sushi 
images from the [Food101 dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html#food101) (101 food classes with 1,000 images each). 
More specifically, we use 20% of images from the pizza, steak and sushi classes selected at random.

* FoodVision Big was trained to classify 101 classes of food. 
FoodVision Big was trained on the images of Food101 dataset to classify all of 101 classes. 




You can find deployed models (and try it out) on my Huggingface Spaces, by clicking the following links:
* [FoodVision Mini Gradio App](https://huggingface.co/spaces/vic-art/foodvision_mini)
* [FoodVision Big Gradio App](https://huggingface.co/spaces/vic-art/foodvision_big)

