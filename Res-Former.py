
from re import X
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary
import copy
from einops import rearrange


def get_imgsize():
    return [(8, 10, 8), (8, 10, 8), (4, 5, 4), (2, 3, 2)]
def get_patchsize():
    return [(3, 3, 3), (2, 2, 2), (1, 1, 1)]

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.conv1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        x = self.avg_pool(input)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return input * x    

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class PatchEmbed(nn.Module):
    """
    3D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_c, embed_dim, drop_path_ratio=0.,norm_layer=None):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (int(math.ceil(img_size[0] / patch_size[0])), int(math.ceil(img_size[1] / patch_size[1])), int(math.ceil(img_size[2] / patch_size[2])))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        # self.avgpool =  nn.AvgPool3d(kernel_size=patch_size[0], stride=patch_size[0], ceil_mode=True, count_include_pad=False)
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
    def forward(self, x):
        B, C, H, W, L = x.shape
        assert H == self.img_size[0] and W == self.img_size[1] and L == self.img_size[2],\
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W, L] -> [B, C, HWL]
        # transpose: [B, C, HWL] -> [B, HWL, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,
                 qkv_scale=None,
                 drop_path_ratio=0.,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qkv_scale or head_dim ** -0.5
        self.QKV = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        
    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qk(): -> [batch_size, num_patches + 1, 2 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 2, num_heads, embed_dim_per_head]
        # permute: -> [2, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.QKV(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=1.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.selfattn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qkv_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, input_data):
        x = input_data
        y = x + self.drop_path(self.selfattn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(y)))
        return x

class AttnBlock(nn.Module):
    def __init__(self,  dim, img_size, patch_size, depth,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):

        super(AttnBlock, self).__init__()
        self.out_c = dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.norm = norm_layer(dim)
        
        if dim == 256:
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_c=64, embed_dim=dim)
            # self.pos_embed = nn.Parameter(torch.zeros(1, 641, dim))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            
            self.res_linear = nn.Identity()
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.downsample = nn.Identity()
            
        elif dim == 512:
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_c=256, embed_dim=dim)
            # self.pos_embed = nn.Parameter(torch.zeros(1, 641, dim))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.res_linear = nn.Linear(256, dim)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.downsample = nn.AvgPool3d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False)
            
        elif dim == 1024:
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_c=512, embed_dim=dim)
            # self.pos_embed = nn.Parameter(torch.zeros(1, 81, dim))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.res_linear = nn.Linear(512, dim)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.downsample = nn.AvgPool3d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False)
        else:
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_c=1024, embed_dim=dim)
            # self.pos_embed = nn.Parameter(torch.zeros(1, 13, dim))  
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.res_linear = nn.Linear(1024, dim)
            nn.init.trunc_normal_(self.cls_token, std=0.02) 
            self.downsample = nn.AvgPool3d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False)
           
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim=dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.relu = nn.ReLU(inplace=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        # Weight init
        self.apply(_init_vit_weights)

    def forward(self, x, res=None):
        B, C, H, W, L = x.shape
        #[B, HWL, C]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
     
        if res is not None:
            res = self.relu(self.res_linear(res))
            res = res.unsqueeze(1)
            cls_token = cls_token + res
        x_train = torch.cat((cls_token, x), dim=1)  # [B, 40+2, 2048]
        x_train = self.pos_drop(x_train)
        x_train = self.blocks(x_train)
        x_train = self.norm(x_train)

        x_cls = x_train[:,0]
        x_out = x_train[:,1:]
        x_out_feature = rearrange(x_out, 'b (h w l) c -> b c h w l', h=H, w=W, l=L)
        x_out_feature = self.downsample(x_out_feature)
        
        return x_cls, x_out_feature

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class ResNet3d_Attn(nn.Module):
    def __init__(self, block, layers, layer_imagesize, layer_patchsize, baseWidth = 26, scale = 4, num_classes=2, input_features=False, se=False, drop_path_ratio=0.):
        self.inplanes = 64
        super(ResNet3d_Attn, self).__init__()
        self.pos_drop = nn.Dropout()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=False),
            nn.Conv3d(32, 32, 3, 2, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=False),
            nn.Conv3d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=False)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        # self.avgpool_attn =  nn.AvgPool3d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        
        self.input_features = input_features
        if self.input_features:
            print("resnet input features conv1 !!!")
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.attnblock1 = AttnBlock(256, layer_imagesize[0], layer_patchsize[2], 1) #math.ceil(layers[0] / 2)
        self.attnblock2 = AttnBlock(512, layer_imagesize[1], layer_patchsize[1], 2)
        self.attnblock3 = AttnBlock(1024, layer_imagesize[2], layer_patchsize[1], 2)
        self.attnblock4 = AttnBlock(2048, layer_imagesize[3], layer_patchsize[1], 2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
    
        
    def copy_param(self, model, weights):
        i = 0
        for param in model.parameters():
            param.data = weights[i]
            i+=1

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool3d(kernel_size=stride, stride=stride, 
                    ceil_mode=True, count_include_pad=False),
                nn.Conv3d(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
           
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        if not self.input_features:
            x = self.conv1(x)
            x = self.bn1(x)
            x0 = self.relu(x)
        else:
            x0 = x
        
        x0 = self.maxpool(x0)
        x1 = self.layer1(x0)
        cls_1, attn_1_feature = self.attnblock1(x0)
        res = cls_1
        attn_cls = cls_1
        x1_attn = x1 + attn_1_feature
        cls_1 = cls_1.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        x1_attn = x1_attn * cls_1
        x1_attn = self.relu(x1_attn)
       
        x2 = self.layer2(x1_attn)
        cls_2, attn_2_feature = self.attnblock2(x1_attn, res=res)
        res = cls_2
        x2_attn = x2 + attn_2_feature 
        cls_2 = cls_2.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        x2_attn = x2_attn * cls_2
        x2_attn = self.relu(x2_attn)

        x3 = self.layer3(x2_attn)
        cls_3, attn_3_feature = self.attnblock3(x2_attn, res=res)
        res = cls_3
        x3_attn = x3 + attn_3_feature
        cls_3 = cls_3.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        x3_attn = x3_attn * cls_3
        x3_attn = self.relu(x3_attn)

        x4 = self.layer4(x3_attn)
        cls_4, attn_4_feature = self.attnblock4(x3_attn, res=res)
        x4_attn = x4 + attn_4_feature
        cls_4 = cls_4.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        x4_attn = x4_attn * cls_4
        x4_attn = self.relu(x4_attn)
        x4_attn_out = self.avgpool(x4_attn)
        x4_attn_out = x4_attn_out.view(x4_attn_out.size(0), -1)

        pre = self.fc(x4_attn_out)
        
        cnn_map = [x1.detach(), x2.detach(), x3.detach(), x4.detach()]
        cnn_attn = [x1_attn.detach(), x2_attn.detach(), x3_attn.detach(), x4_attn.detach()]
        attn_map = [attn_1_feature.detach(), attn_2_feature.detach(), attn_3_feature.detach(), attn_4_feature.detach()]
        attn_cls = [cls_1.squeeze(-1).squeeze(-1).squeeze(-1).detach(), cls_2.squeeze(-1).squeeze(-1).squeeze(-1).detach(), cls_3.squeeze(-1).squeeze(-1).squeeze(-1).detach(), cls_4.squeeze(-1).squeeze(-1).squeeze(-1).detach()]

        del x1, x2, x3, x4
        del x1_attn, x2_attn, x3_attn, x4_attn
        del attn_1_feature, attn_2_feature, attn_3_feature, attn_4_feature
        del cls_1, cls_2, cls_3, cls_4
        return cnn_attn, attn_cls, pre, cnn_map, attn_map
    


def resnet50_v1b(pretrained=False, **kwargs):
    model = ResNet3d_Attn(Bottleneck, [3, 4, 6, 3], get_imgsize(), get_patchsize(), baseWidth = 26, scale = 4, **kwargs)

    return model

def resnet101_v1b(pretrained=False, **kwargs):
    model = ResNet3d_Attn(Bottleneck, [3, 4, 23, 3], get_imgsize(), get_patchsize(), baseWidth = 26, scale = 4, **kwargs)
    return model


def resnet152_v1b_26w_4s(pretrained=False, **kwargs):
    model = ResNet3d_Attn(Bottleneck, [3, 8, 36, 3], get_imgsize(), get_patchsize(), baseWidth = 26, scale = 4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net152_v1b_26w_4s']))
    return model



