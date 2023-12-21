from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        mid_ch = in_ch if in_ch < out_ch else out_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.activation(x1)

        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        output = self.activation(x1)
        return output




class Squeeze_Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, in_ch, a_ch, out_ch):
        super(Squeeze_Attention_block, self).__init__()


        self.feature = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(a_ch, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, a):
        x = self.feature(x)
        a = self.attention(a)
        output = self.out_conv(x*a)
        return output



class MDFNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(MDFNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4])
        #
        self.conv0_1 = conv_block_nested(filters[0] + filters[1]*2, filters[0])
        self.conv1_1 = conv_block_nested(filters[0] + filters[1] + filters[2]*2, filters[1])
        self.conv2_1 = conv_block_nested(filters[1] + filters[2] + filters[3], filters[2])
        self.conv3_1 = conv_block_nested(filters[2] + filters[3] + filters[4], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1]*2, filters[0])
        self.conv1_2 = conv_block_nested(filters[0] + filters[1]*2 + filters[2]*2, filters[1])
        self.conv2_2 = conv_block_nested(filters[1] + filters[2]*2 + filters[3]*2, filters[2])

        self.conv0_c = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.conv1_c = nn.Conv2d(filters[1], out_ch, kernel_size=1)
        self.conv2_c = nn.Conv2d(filters[2], out_ch, kernel_size=1)
        self.conv3_c = nn.Conv2d(filters[3], out_ch, kernel_size=1)
        self.conv4_c = nn.Conv2d(filters[4], out_ch, kernel_size=1)

        self.conv0_4 = conv_block_nested(filters[0] + in_ch + out_ch, filters[0])
        self.conv1_4 = conv_block_nested(filters[0] + filters[1] + out_ch, filters[1])
        self.conv2_4 = conv_block_nested(filters[1] + filters[2] + out_ch, filters[2])
        self.conv3_3 = conv_block_nested(filters[2] + filters[3] + out_ch, filters[3])
        self.conv4_2 = conv_block_nested(filters[3] + filters[4] + out_ch, filters[4])

        self.conv0_5 = conv_block_nested(filters[0] + filters[1]*2, filters[0])
        self.conv1_5 = conv_block_nested(filters[0] + filters[1] + filters[2]*2, filters[1])
        self.conv2_5 = conv_block_nested(filters[1] + filters[2] + filters[3]*2, filters[2])
        self.conv3_4 = conv_block_nested(filters[2] + filters[3] + filters[4], filters[3])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)


    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x2_1 = self.conv2_1(torch.cat([self.pool(x1_0), x2_0, self.Up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([self.pool(x2_1), x3_0, self.Up(x4_0)], 1))

        x1_1 = self.conv1_1(torch.cat([self.pool(x0_0), x1_0, self.Up(x2_0), self.Up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0), self.Up(x1_1)], 1))

        x2_2 = self.conv2_2(torch.cat([self.pool(x1_1), x2_0, x2_1, self.Up(x3_0), self.Up(x3_1)], 1))
        x1_2 = self.conv1_2(torch.cat([self.pool(x0_1), x1_0, x1_1, self.Up(x2_1), self.Up(x2_2)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1), self.Up(x1_2)], 1))

        x0_c = self.conv0_c(x0_2)
        x1_c = self.conv1_c(x1_2)
        x2_c = self.conv2_c(x2_2)
        x3_c = self.conv3_c(x3_1)
        x4_c = self.conv4_c(x4_0)
        _x0_c = torch.sigmoid(x0_c)
        _x1_c = torch.sigmoid(x1_c)
        _x2_c = torch.sigmoid(x2_c)
        _x3_c = torch.sigmoid(x3_c)
        _x4_c = torch.sigmoid(x4_c)


        x0_4 = torch.cat([x, x0_2, _x0_c], 1)
        x0_4 = self.conv0_4(x0_4)
        x1_4 = torch.cat([self.pool(x0_4), x1_2, _x1_c], 1)
        x1_4 = self.conv1_4(x1_4)
        x2_4 = torch.cat([self.pool(x1_4), x2_2, _x2_c], 1)
        x2_4 = self.conv2_4(x2_4)
        x3_3 = torch.cat([self.pool(x2_4), x3_1, _x3_c], 1)
        x3_3 = self.conv3_3(x3_3)
        x4_2 = torch.cat([self.pool(x3_3), x4_0, _x4_c], 1)
        x4_2 = self.conv4_2(x4_2)
        # print(torch.cat([x, x0_2], 1).shape, x0_c.shape, x0_4.shape)

        x3_4 = self.conv3_4(torch.cat([self.pool(x2_4), x3_3, self.Up(x4_2)], 1))
        x2_5 = self.conv2_5(torch.cat([self.pool(x1_4), x2_4, self.Up(x3_3), self.Up(x3_4)], 1))
        x1_5 = self.conv1_5(torch.cat([self.pool(x0_4), x1_4, self.Up(x2_4), self.Up(x2_5)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_4, self.Up(x1_4), self.Up(x1_5)], 1))

        output = self.final(x0_5)


        return output, x0_c, x1_c, x2_c, x3_c, x4_c




class MDFNet_S(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(MDFNet_S, self).__init__()

        n1 = 48
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 12]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4])
        #
        self.conv0_1 = conv_block_nested(filters[0] + filters[1]*2, filters[0])
        self.conv1_1 = conv_block_nested(filters[0] + filters[1] + filters[2]*2, filters[1])
        self.conv2_1 = conv_block_nested(filters[1] + filters[2] + filters[3], filters[2])
        self.conv3_1 = conv_block_nested(filters[2] + filters[3] + filters[4], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1]*2, filters[0])
        self.conv1_2 = conv_block_nested(filters[0] + filters[1]*2 + filters[2]*2, filters[1])
        self.conv2_2 = conv_block_nested(filters[1] + filters[2]*2 + filters[3]*2, filters[2])

        self.conv0_c = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.conv1_c = nn.Conv2d(filters[1], out_ch, kernel_size=1)
        self.conv2_c = nn.Conv2d(filters[2], out_ch, kernel_size=1)
        self.conv3_c = nn.Conv2d(filters[3], out_ch, kernel_size=1)
        self.conv4_c = nn.Conv2d(filters[4], out_ch, kernel_size=1)

        self.conv0_4 = conv_block_nested(filters[0] + in_ch + out_ch, filters[0])
        self.conv1_4 = conv_block_nested(filters[0] + filters[1] + out_ch, filters[1])
        self.conv2_4 = conv_block_nested(filters[1] + filters[2] + out_ch, filters[2])
        self.conv3_3 = conv_block_nested(filters[2] + filters[3] + out_ch, filters[3])
        self.conv4_2 = conv_block_nested(filters[3] + filters[4] + out_ch, filters[4])

        self.conv0_5 = conv_block_nested(filters[0] + filters[1]*2, filters[0])
        self.conv1_5 = conv_block_nested(filters[0] + filters[1] + filters[2]*2, filters[1])
        self.conv2_5 = conv_block_nested(filters[1] + filters[2] + filters[3]*2, filters[2])
        self.conv3_4 = conv_block_nested(filters[2] + filters[3] + filters[4], filters[3])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)


    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x2_1 = self.conv2_1(torch.cat([self.pool(x1_0), x2_0, self.Up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([self.pool(x2_1), x3_0, self.Up(x4_0)], 1))

        x1_1 = self.conv1_1(torch.cat([self.pool(x0_0), x1_0, self.Up(x2_0), self.Up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0), self.Up(x1_1)], 1))

        x2_2 = self.conv2_2(torch.cat([self.pool(x1_1), x2_0, x2_1, self.Up(x3_0), self.Up(x3_1)], 1))
        x1_2 = self.conv1_2(torch.cat([self.pool(x0_1), x1_0, x1_1, self.Up(x2_1), self.Up(x2_2)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1), self.Up(x1_2)], 1))

        x0_c = self.conv0_c(x0_2)
        x1_c = self.conv1_c(x1_2)
        x2_c = self.conv2_c(x2_2)
        x3_c = self.conv3_c(x3_1)
        x4_c = self.conv4_c(x4_0)
        _x0_c = torch.sigmoid(x0_c)
        _x1_c = torch.sigmoid(x1_c)
        _x2_c = torch.sigmoid(x2_c)
        _x3_c = torch.sigmoid(x3_c)
        _x4_c = torch.sigmoid(x4_c)


        x0_4 = torch.cat([x, x0_2, _x0_c], 1)
        x0_4 = self.conv0_4(x0_4)
        x1_4 = torch.cat([self.pool(x0_4), x1_2, _x1_c], 1)
        x1_4 = self.conv1_4(x1_4)
        x2_4 = torch.cat([self.pool(x1_4), x2_2, _x2_c], 1)
        x2_4 = self.conv2_4(x2_4)
        x3_3 = torch.cat([self.pool(x2_4), x3_1, _x3_c], 1)
        x3_3 = self.conv3_3(x3_3)
        x4_2 = torch.cat([self.pool(x3_3), x4_0, _x4_c], 1)
        x4_2 = self.conv4_2(x4_2)
        # print(torch.cat([x, x0_2], 1).shape, x0_c.shape, x0_4.shape)

        x3_4 = self.conv3_4(torch.cat([self.pool(x2_4), x3_3, self.Up(x4_2)], 1))
        x2_5 = self.conv2_5(torch.cat([self.pool(x1_4), x2_4, self.Up(x3_3), self.Up(x3_4)], 1))
        x1_5 = self.conv1_5(torch.cat([self.pool(x0_4), x1_4, self.Up(x2_4), self.Up(x2_5)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_4, self.Up(x1_4), self.Up(x1_5)], 1))

        output = self.final(x0_5)


        return output, x0_c, x1_c, x2_c, x3_c, x4_c
