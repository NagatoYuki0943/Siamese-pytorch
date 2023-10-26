import torch
import torch.nn as nn

from nets.vgg import VGG16

#----------------------------------------------------------#
#   获取vgg的输出展平的长度
#----------------------------------------------------------#
def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height)

#----------------------------------------------------------#
#   将backbone的输出结果平铺到一维上,两个向量长度相同
#   两个向量进行相减取(相减就是对比)绝对值,得到特征差
#   进行全连接到一维上,取sigmoid,输出到0~1之间,接近1说明相似
#----------------------------------------------------------#
class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()
        self.vgg = VGG16(pretrained, 3)
        del self.vgg.avgpool
        del self.vgg.classifier

        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        #------------------------------------------#
        #   b,512*3*3 -> b,512 -> b,1
        #------------------------------------------#
        self.fully_connect1 = nn.Linear(flat_shape, 512)
        self.fully_connect2 = nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        #------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        #   result: b,512,3,3
        #------------------------------------------#
        x1 = self.vgg.features(x1)
        x2 = self.vgg.features(x2)
        #-------------------------#
        #   铺平
        #   b,512,3,3 -> b,512*3*3
        #-------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        #-------------------------#
        #   相减取绝对值
        #-------------------------#
        x = torch.abs(x1 - x2)
        #-------------------------#
        #   进行两次全连接
        #   b,512*3*3 -> b,512 -> b,1
        #-------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x
