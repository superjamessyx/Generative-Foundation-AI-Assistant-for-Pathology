# Classifier for Selecting Pathology Images



This is a ConvNext-tiny model trained on 30K annotations on if o

## Usage

> #### Step1: Download model checkpoint in [convnext-pathology-classifier](https://huggingface.co/jamessyx/convnext-pathology-classifier) .



> #### Step2: Load the model

You can use the following code to load the model.

```python
from model_factory.modeling_convnext import CT_SINGLE

model = CT_SINGLE('convnext_tiny')
model_path = '<PATH OF DOWNLOADED MODEL WEIGHT>'
def load_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("Resume checkpoint %s" % checkpoint_path)

load_model(model_path, model)
```



> ### Step3: Construct and predict your own data

In this step, you'll construct your own dataset. Use PIL to load images and employ `transforms` from torchvision for data preprocessing.

```python
def default_loader(path):
    img = Image.open(path)
    return img.convert('RGB')

transform_val_global = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
```

