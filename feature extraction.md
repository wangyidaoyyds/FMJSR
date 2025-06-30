**Feature extraction using ResNet**
**Installation Steps**
**1.Download Model Resources**
First, you need to download the ResNet - 50 v1.5 model resources from the Hugging Face model repository.
Please visit the following link to download:[https://huggingface.co/microsoft/resnet-50](url)
**2.Detailed Installation Process**
After successfully downloading the model resources, you need to follow these steps to install:
 Install PyTorch: If you haven't installed PyTorch yet, please visit the PyTorch official website
 [(https://pytorch.org/)](url) and follow the prompts to install it.
 Install the Transformers Library: Execute the following command to install the Transformers library:
` pip install transformers`
Import the model: Import the ResNet-50 v1.5 model into your code as follows:
`from transformers import ResNetForImageClassification`
