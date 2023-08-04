import torch
from torch import nn
from torch.nn import functional as F


class _CrossAttND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, bn_layer=True,sub_sample=True):
        super(_CrossAttND, self).__init__()
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.in_channels= in_channels
        self.in_img_channels = int((in_channels/4)*3)
        self.in_mask_channels=int((in_channels/4)*1)


        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=in_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.in_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_mask_channels, out_channels=self.in_mask_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_img_channels, out_channels=self.in_mask_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        batch_size = x.size(0)
        channel_num=x.size(1)
        assert channel_num % 4 == 0
        channel_index=int(channel_num/4)

        mask_channel=x[:,:channel_index,:,:]
        img_channel=x[:,channel_index:,:,:]


        g_x = self.g(x).view(batch_size, self.in_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(mask_channel).view(batch_size, self.in_mask_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(img_channel).view(batch_size, self.in_mask_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.in_channels, *x.size()[2:]) #as same as the x.shape()
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z



class CrossAttention(_CrossAttND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=False):
        super(CrossAttention, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2,
                                              bn_layer=bn_layer,)



if __name__ == '__main__':
    import torch
    img = torch.randn(4, 4, 128, 128)
    net = CrossAttention(4, bn_layer=True)
    #img,net=img.to('cuda'),net.to('cuda')
    out = net(img)
    print(out.size())



