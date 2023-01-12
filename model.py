import torch
import torch.nn as nn

class MiDaSProNet(nn.Module):
    def __init__(self):
        super(MiDaSProNet, self).__init__()
        # out: [batch, 64, 256]
        model_type = 'MiDaS_small'
        self.midas = torch.hub.load('intel-isl/MiDas', model_type)
        self.conv_same = nn.Conv2d(1, 10, 1)
        self.reconstruct = nn.Sequential(
            nn.Conv2d(1, 5, 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(5, 10, 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(10, 5, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(5, 10, 4),
            nn.ReLU(True),
        )
        self.mix = nn.Conv2d(20, 1, 1)

    def forward(self, imgs):
        """Forward pass.
        Args:
            input (tensor): input data (image) [batch, 3, 64, 256]
        Returns:
            tensor: disparity [batch, 1, 64, 256]
        """
        disp_list = []
        depth = self.midas(imgs).unsqueeze(1)

        disparity_same = self.conv_same(depth)
        disparity_reconstruct = self.reconstruct(depth)
        disparity = torch.cat((disparity_same, disparity_reconstruct), 1)
        disparity = self.mix(disparity).squeeze(1)
        disp_list.append(disparity)

        return torch.stack(disp_list)[0]