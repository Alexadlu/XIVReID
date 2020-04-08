import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            )

        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def alexnet(pretrained=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        ckpt = model_zoo.load_url(model_urls['alexnet'])
        model.load_state_dict(remove_fc(ckpt))
    return model

def remove_fc(state_dict):

  #del state_dict['classifier.1.weight']
  #del state_dict['classifier.1.bias']
  #del state_dict['classifier.4.weight']
  #del state_dict['classifier.4.bias']
  del state_dict['classifier.6.weight']
  del state_dict['classifier.6.bias']

  return state_dict





