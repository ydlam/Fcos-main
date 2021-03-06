# -*- coding: utf-8 -*-
"""
# @file name  : darknet.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-04
# @brief      : DarkNet模型类
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['DarkNet', 'darknet19']

model_urls = {
    'darknet19':
    'https://s3.ap-northeast-2.amazonaws.com/deepbaksuvision/darknet19-deepBakSu-e1b3ec1e.pth'
}


class DarkNet(nn.Module):
    def __init__(self, features, init_weights=True):
        super(DarkNet, self).__init__()
        self.layer0 = features[0]
        self.layer1 = features[1]
        self.layer2 = features[2]
        self.layer3 = features[3]
        self.layer4 = features[4]

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c2, c3, c4, c5

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, flag=True):
    stages = []
    in_channels = 3
    for u in cfg:
        layers = []
        for v in u:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if batch_norm:
                    layers += [
                        nn.Conv2d(in_channels,
                                  v,
                                  kernel_size=(1, 3)[flag],
                                  padding=(0, 1)[flag],
                                  bias=False),
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [
                        nn.Conv2d(in_channels,
                                  v,
                                  kernel_size=(1, 3)[flag],
                                  padding=(0, 1)[flag]),
                        nn.ReLU(inplace=True),
                    ]
                in_channels = v
            flag = not flag
        stages.append(nn.Sequential(*layers))

    return nn.Sequential(*stages)


cfg = [[32, 'M', 64], ['M', 128, 64, 128], ['M', 256, 128, 256],
       ['M', 512, 256, 512, 256, 512], ['M', 1024, 512, 1024, 512, 1024]]


def darknet19(pretrained=False, **kwargs):
    """DarkNet 19-layer model with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = DarkNet(make_layers(cfg, batch_norm=True), **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['darknet19']))
        weight_file = '/Users/ygq/杨广奇/DeepLearning/第7期课件资料/目标检测/第4节实战任务分解/FCOS/weight/darknet19-deepBakSu-e1b3ec1e.pth'
        dic = {}
        for now_keys, values in zip(model.state_dict().keys(), torch.load(weight_file).values()):
            dic[now_keys] = values
        model.load_state_dict(dic)
    return model


if __name__ == "__main__":

    import torch
    from torchsummary import summary

    model = darknet19()
    summary(model, (3, 224, 224), 2, device="cpu")

    x = torch.rand(2, 3, 224, 224)
    out = model(x)
    [print(stage_out.shape) for stage_out in out]
