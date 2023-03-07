import torch
import torchvision

from torch import nn

def create_effnetb2_model(num_classes: int = 3,
                          seed: int = 42):
    """Creates an EfficientNetB2 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNetB2 feature extractor model. 
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """
  # Create EffNetB2 pretrained weights, transforms and model
  weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  transforms = weights.transforms()
  model = torchvision.models.efficientnet_b2(weights = weights)

  # 4. Freeze the base layers in the model, this will stop all layers from training
  for param in model.parameters():
    param.requires_grad = False

  # 5. Change classifier head with random seed 
  torch.manual_seed(seed)
  model.classifier = nn.Sequential(
    nn.Dropout(p = 0.3, inplace = True),
    nn.Linear(in_features=1408, out_features=num_classes, bias=True)
  )
  return model, transforms
