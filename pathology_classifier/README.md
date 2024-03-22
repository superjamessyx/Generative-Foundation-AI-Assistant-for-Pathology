# Classifier for Selecting Pathology Images



This is a ConvNext-tiny model trained on 30K annotations on if o

## Usage

> #### Step1: Download model checkpoint in [convnext-pathology-classifier](https://huggingface.co/jamessyx/convnext-pathology-classifier) .



> #### Step2: Load the model

You can use the following code to load the model.

```python
import timm ##timm version 0.9.7
import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image

class CT_SINGLE(nn.Module):
    def __init__(self, model_name):
        super(CT_SINGLE, self).__init__()
        print(model_name)
        self.model_global = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.fc = nn.Linear(768, 2)

    def forward(self, x_global):
        features_global = self.model_global(x_global)
        logits = self.fc(features_global)
        return logits

def load_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("Resume checkpoint %s" % checkpoint_path)

##load the model
model = CT_SINGLE('convnext_tiny')
model_path = 'Your model path'
load_model(model_path, model)
model.eval().cuda()

```



> ### Step3: Construct and predict your own data

In this step, you'll construct your own dataset. Use PIL to load images and employ `transforms` from torchvision for data preprocessing.

```python
def default_loader(path):
    img = Image.open(path)
    return img.convert('RGB')

data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def predict(img_path, model):
    img = default_loader(img_path)
    img = data_transforms(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    output = model(img)
    _, pred = torch.topk(output, 1, dim=-1)
    pred = pred.data.cpu().numpy()[:, 0]
    return pred   ## 0 indicates non-pathology image and 1 indicates pathology image

img_path = 'Your image path'
pred = predict(img_path, model)
print(pred)
```

