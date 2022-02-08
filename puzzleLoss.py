from torch.nn.functional import one_hot

class puzzleLossCE(nn.Module):
    def __init__(self, num_patches):
        super().__init__()
        self.num_patches=num_patches

    def forward(self, x,mask):
        """
        Cross-entropy between softmax outputs of the ViT and the labels of the hidden patches.
        """
        total_loss = 0
        if (sum(mask)>=1):
          ind_hidden=(mask == 1).nonzero(as_tuple=True)[0]
          y_true=one_hot(ind_hidden,num_classes=self.num_patches)
          x = torch.clamp(x, 1e-9, 1 - 1e-9) 
          total_loss=-(y_true * torch.log(x)).sum(dim=1).mean()
        return total_loss
