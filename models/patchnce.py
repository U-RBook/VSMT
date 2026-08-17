import random
from packaging import version
import torch
from torch import nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, detach_tgt=True):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        if detach_tgt:
            feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss



class ShiftContrastiveLoss(nn.Module):
    def __init__(self, pos_temperature=0.15, neg_temperature=0.15):
        super().__init__()
        # self.batch_size = batch_size
        self.register_buffer("pos_temperature", torch.tensor(pos_temperature))			
        self.register_buffer("neg_temperature", torch.tensor(neg_temperature))			
        
    def forward(self, inputs):	
        	
        z_inputs = F.normalize(inputs[0], dim=1)#.detach()     
        z_p2 = F.normalize(inputs[1], dim=1)  
        z_n = F.normalize(inputs[2], dim=1) 
        
        p_sim = (z_inputs*z_p2).sum(dim=1)    
        B, C, H, W = z_inputs.shape
        z_inputs = z_inputs.view(B, C, 1, H*W)
        z_n = z_n.view(B, C, H*W, 1)
        n_sim = (z_inputs * z_n).sum(dim=1) 
        
        nominator = torch.exp(p_sim / self.pos_temperature)             
        denominator = torch.exp((n_sim - 1.2) / self.neg_temperature)
             
        loss_partial = \
        -torch.log((nominator) / (nominator + torch.sum(denominator, dim=[1]).view(B, H, W))) 
       
        loss = torch.sum(loss_partial) / (B*H*W)
        return loss