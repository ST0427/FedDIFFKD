from PIL import Image
from os.path import join
import imageio
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models


class EncoderFemnist(nn.Module):
    def __init__(self, code_length):
        super(EncoderFemnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320), code_length)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        z = F.relu(self.fc1(x))
        return z       
        
class CNNFemnist(nn.Module):
    def __init__(self, args,code_length=50,num_classes = 62):
        super(CNNFemnist, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = EncoderFemnist(self.code_length)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(self.code_length, self.num_classes),
                                        nn.LogSoftmax(dim=1))
        
    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z,p
       
        
# class ResNet181(nn.Module):
#     def __init__(self, args,code_length=64,num_classes = 10):
#         super(ResNet181, self).__init__()
#         self.code_length = code_length
#         self.num_classes = num_classes
#         self.feature_extractor = models.resnet18(num_classes=self.code_length)
#         self.classifier =  nn.Sequential(
#                                 nn.Linear(self.code_length, self.num_classes))
#
#     def forward(self,x):
#         z = self.feature_extractor(x)
#         p = self.classifier(z)
#         return z,p

class ResNet181(nn.Module):
    def __init__(self, args,code_length=64,num_classes = 10):
        super(ResNet181, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        #加载ResNet18，不使用预训练
        resnet=models.resnet18(pretrained=False)
        #修改第一个卷积层，减少下采样
        resnet.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        # 移除平均池化层和全连接层，以便在平均池化之前获取输出特征
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        num_features=resnet.fc.in_features
        #平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #自定义的分类器
        self.classifier=nn.Linear(num_features,self.num_classes)

    def forward(self,x):
        #通过修改后的ResNet18 backbone获取特征
        z = self.feature_extractor(x)
        #在特征上应用平均池化
        z=self.avgpool(z)
        z=torch.flatten(z,1)
        #通过分类器获取logits
        p = self.classifier(z)
        return z,p



class ResNet182(nn.Module):
    def __init__(self, args,code_length=64,num_classes = 10):
        super(ResNet182, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        #加载ResNet18，不使用预训练
        resnet=models.resnet18(pretrained=False)
        #修改第一个卷积层，减少下采样
        resnet.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        # 移除平均池化层和全连接层，以便在平均池化之前获取输出特征
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        num_features=resnet.fc.in_features
        #平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #自定义的分类器
        self.classifier=nn.Linear(num_features,self.num_classes)

    def forward(self,x):
        #通过修改后的ResNet18 backbone获取特征
        z = self.feature_extractor(x)
        #在特征上应用平均池化
        pooled_output=self.avgpool(z)
        pooled_output=torch.flatten(pooled_output,1)
        #通过分类器获取logits
        p = self.classifier(pooled_output)
        return z,p




# class ShuffLeNet(nn.Module):
#     def __init__(self, args,code_length=64,num_classes = 10):
#         super(ShuffLeNet, self).__init__()
#         self.code_length = code_length
#         self.num_classes = num_classes
#         self.feature_extractor = models.shufflenet_v2_x1_0(num_classes=self.code_length)
#         self.classifier =  nn.Sequential(
#                                 nn.Linear(self.code_length, self.num_classes))
#     def forward(self,x):
#         z = self.feature_extractor(x)
#         p = self.classifier(z)
#         return z,p

class ShuffLeNet1(nn.Module):
    def __init__(self, args,code_length=64,num_classes = 10):
        super(ShuffLeNet1, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        shfflenet = models.shufflenet_v2_x1_0(pretrained=False)
        # 移除平均池化层和全连接层，以便在平均池化之前获取输出特征
        self.feature_extractor = nn.Sequential(*list(shfflenet.children())[:-2])

        num_features = shfflenet.fc.in_features
        # 平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 自定义的分类器

        self.classifier = nn.Linear(num_features, self.num_classes)
    def forward(self,x):
        z = self.feature_extractor(x)
        # 在特征上应用平均池化
        z = self.avgpool(z)
        z = torch.flatten(z, 1)

        # 通过分类器获取logits
        p = self.classifier(z)
        return z,p

class ShuffLeNet2(nn.Module):
    def __init__(self, args,code_length=64,num_classes = 10):
        super(ShuffLeNet2, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        shfflenet = models.shufflenet_v2_x1_0(pretrained=False)
        # 移除平均池化层和全连接层，以便在平均池化之前获取输出特征
        self.feature_extractor = nn.Sequential(*list(shfflenet.children())[:-2])

        num_features = shfflenet.fc.in_features
        # 平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 自定义的分类器

        self.classifier = nn.Linear(num_features, self.num_classes)
    def forward(self,x):
        z = self.feature_extractor(x)
        # 在特征上应用平均池化
        pooled_output = self.avgpool(z)
        pooled_output = torch.flatten(pooled_output, 1)

        # 通过分类器获取logits
        p = self.classifier(pooled_output)
        return z,p

#DiffKD Model

class NoiseAdapter(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size == 3:
            self.feat = nn.Sequential(
                Bottleneck(channels, channels, reduction=8),
                Bottleneck(channels, channels, reduction=8),
                # nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
                # nn.BatchNorm2d(channels),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
                # nn.BatchNorm2d(channels),
                # nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            self.feat = nn.Sequential(
                nn.Conv2d(channels, channels * 2, 1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
            )
        self.pred = nn.Linear(channels, 2)

    def forward(self, x):
        x = self.feat(x).flatten(1)
        x = self.pred(x).softmax(1)[:, 0]
        return x


class DiffusionModel(nn.Module):
    def __init__(self, channels_in=3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.time_embedding = nn.Embedding(1280, channels_in)

        if kernel_size == 3:
            self.pred = nn.Sequential(
                Bottleneck(channels_in, channels_in),
                Bottleneck(channels_in, channels_in),
                # Bottleneck(channels_in, channels_in),
                # nn.Conv2d(channels_in,channels_in,kernel_size=kernel_size,padding=kernel_size//2),
                # nn.BatchNorm2d(channels_in),
                # nn.ReLU(inplace=True),

                # nn.Conv2d(channels_in,channels_in,kernel_size=kernel_size,padding=kernel_size//2),
                # nn.BatchNorm2d(channels_in),
                # nn.ReLU(inplace=True),
                #
                # nn.Conv2d(channels_in, channels_in, kernel_size=kernel_size, padding=kernel_size // 2),
                # nn.BatchNorm2d(channels_in),
                # nn.ReLU(inplace=True),
                nn.Conv2d(channels_in, channels_in, 1),
                nn.BatchNorm2d(channels_in)
            )
        else:
            self.pred = nn.Sequential(
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1),
                nn.BatchNorm2d(channels_in),
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1)
            )

    def forward(self, noisy_image, t):
        if t.dtype != torch.long:
            t = t.type(torch.long)
        feat = noisy_image
        feat = feat + self.time_embedding(t)[..., None, None]
        ret = self.pred(feat)
        return ret


class AutoEncoder(nn.Module):
    def __init__(self, channels, latent_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, latent_channels, 1, padding=0),
            nn.BatchNorm2d(latent_channels)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, channels, 1, padding=0),
        )

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden)
        return hidden, out

    def forward_encoder(self, x):
        return self.encoder(x)


class DDIMPipeline:
    '''
    Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py
    '''

    def __init__(self, model, scheduler, noise_adapter=None, solver='ddim'):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.noise_adapter = noise_adapter
        self._iter = 0
        self.solver = solver

    def __call__(
            self,
            batch_size,
            device,
            dtype,
            shape,
            feat,
            generator=None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            proj=None
    ):

        # Sample gaussian noise to begin loop
        image_shape = (batch_size, *shape)

        if self.noise_adapter is not None:
            noise = torch.randn(image_shape, device=device, dtype=dtype)
            timesteps = self.noise_adapter(feat)
            G=timesteps.mean()
            image = self.scheduler.add_noise_diff2(feat, noise, timesteps)
        else:
            image = feat

        # set step values
        self.scheduler.set_timesteps(num_inference_steps * 2)

        for t in self.scheduler.timesteps[len(self.scheduler.timesteps) // 2:]:
            noise_pred = self.model(image, t.to(device))

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                noise_pred, t, image, eta=eta, use_clipped_model_output=True, generator=generator
            )['prev_sample']

        self._iter += 1
        return image,G


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        return out + x
