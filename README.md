# FoodVision-models-and-deployment-with-Gradio
---

The task was to create 2 models for food classification present on a given image and deploy them with Gradio App:
* **FoodVision Mini**
* **FoodVision Big**

## Dataset

* FoodVision Mini was trained to classify 3 classes of food, including pizza, steak and sushi.
The dataset we're going to use for training a FoodVision Mini model is a sample of pizza, steak and sushi 
images from the [Food101 dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html#food101) (101 food classes with 1,000 images each). 
More specifically, we use 20% of images from the pizza, steak and sushi classes selected at random.

* FoodVision Big was trained to classify 101 classes of food. 
FoodVision Big was trained on the images of Food101 dataset to classify all of 101 classes. 

# Model

For an application of such type as FoodVision, not only accuracy, but also prediction speed is important (it is unlikely that a user will want to wait forever to get prediction results based on a photo of their food). 
That's why we set the following requirement to the model before experimenting:
* Performance: A model should performs at 95%+ accuracy.
* Speed: A model can classify images as close to real-time as possible (~30FPS or 30ms latency)

For experimenting we chose 2 pretrained models:
* EffNetB2;
* ViT-B/16.

Then we trained 2 FoodVision Mini models (EffNetB2 and ViT-B/16) and compared their results in terms of performance and speed of prediction.
It turns out that both models reached the requirement in terms of performance with ViT slightly outperforming EffNetB2, but at the expense of having 11x+ the number of parameters, 11x+ the model size and 2x+ the prediction time per image. 

Given we are prioritize speed, we decided to stick with EffNetB2 for FoodVision Mini and FoodVision Big since it's faster and has a much smaller footprint.


# Deployment 

You can find deployed models (and try it out) on my Huggingface Spaces, by clicking the following links:
* [FoodVision Mini Gradio App](https://huggingface.co/spaces/vic-art/foodvision_mini)
* [FoodVision Big Gradio App](https://huggingface.co/spaces/vic-art/foodvision_big)

