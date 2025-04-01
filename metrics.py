from torchvision.models import resnet18
from torchscope import scope

model = resnet18()
scope(model, input_size=(3, 224, 224))