import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
from kornia.contrib.extract_patches import extract_tensor_patches
from kornia.augmentation import ImageSequential
import kornia.augmentation as K

from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss, ShiftContrastiveLoss
from .gauss_pyramid import Gauss_Pyramid_Conv
import util.util as util
from .detach_dab import HEDTransform
from .trans.rnd_rot import RandomRotationReflection

class VSMTModel(BaseModel):


    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--netF_nc', type=int, default=256)
       
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.set_defaults(pool_size=0)  # no image pooling

        # FDL:
        parser.add_argument('--gp_weights', type=str, default='uniform', help='weights for reconstruction pyramids.')
        parser.add_argument('--n_downsampling', type=int, default=2, help='# of downsample in G')

        parser.add_argument('--weight_std_loss', type=float, default=1.0, help='')
        parser.add_argument('--weight_gp', type=float, default=10.0, help='weight for pyramid loss')
        parser.add_argument('--weight_gan_loss_G', type=float, default=1.0, help='weight for gan loss')
        parser.add_argument('--weight_gan_loss_FG', type=float, default=1.0, help='weight for gan loss')
        parser.add_argument('--weight_sobel', type=float, default=10.0, help='weight for sobel loss')
        parser.add_argument('--weight_dis_loss', type=float, default=5.0, help='weight for final dis loss')
        parser.add_argument('--weight_rgb_loss', type=float, default=5.0, help='weight for final rgb loss')
        parser.add_argument('--weight_nce_loss', type=float, default=10.0, help='weight for nce loss')
        parser.add_argument('--weight_rebuild_loss', type=float, default=10.0, help='weight for rebuild loss')
        parser.add_argument('--dis_weight_schedule_epoch', type=int, default=3, help='')
        parser.add_argument('--shifter_pane', type=int, default=16, help='the size of shifter pane, (crop_size/4/shifter_pane)^2 = patch number')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer in nce loss')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            action = 'store_true',
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')


        parser.add_argument('--label_reverse', action='store_true', help='if true, reverse the label of real and fake')
        parser.add_argument('--label_smooth', type=float, default=0.2, help='label smoothing')
        parser.add_argument('--dropout_rate', type=float, default=0.0, help='dropout rate')
        parser.add_argument('--norm_input', type=util.str2bool, default=False, help='if normalize the input')
        parser.add_argument('--colorjitter', type=util.str2bool, default=False, help='if normalize the input')
        parser.add_argument('--norm_output', type=util.str2bool, default=True, help='if normalize the input')

        parser.add_argument('--he_rec', type=str, default='')
        parser.add_argument('--he_cls', type=str, default='')
        parser.add_argument('--ihc_rec', type=str, default='')
        parser.add_argument('--ihc_cls', type=str, default='')






        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, weight_nce_loss=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, weight_nce_loss=10.0, flip_equivariance=False,
                n_epochs=20, n_epochs_decay=10
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
      
        self.optimize_counter = 0
       
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_NCE', 'G_Dis','G_PY', 'D_real', 'D_fake', 'G_Std']
        self.loss_names += ['FG_GAN', 'FG_NCE', 'FG_PY','DFG_real', 
                            'DFG_fake','Shifter','fusion','Shifter_cp']

        if self.opt.isTrain:
            self.visual_names = ['real_A', 'real_B', 'fake_B']
        else:
            self.visual_names = ['fake_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'FD','FG1', 'FG2', 'DP', 'DP2']
        else:  
            self.model_names = ['G'] # during test time, only load G

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        print(self.netG)
        if self.isTrain:
            self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netFG1 = networks.define_FG(netG='FG',init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, shifter_pane=opt.shifter_pane, opt=opt)
            self.netFG2 = networks.define_FG(netG='FG',init_type=opt.init_type, init_gain=opt.init_gain, 
                                             gpu_ids=self.gpu_ids, shifter_pane=opt.shifter_pane, opt=opt, cp_train=True)    
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netFD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netDP = DisProjector().to(self.device)
            self.netDP2 = DisProjector().to(self.device)

            print(self.netFG1)
            print(self.netD)
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_FD = torch.optim.Adam(self.netFD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_FG = torch.optim.Adam([{'params': self.netFG1.parameters()},
                                                 {'params': self.netFG2.parameters()},
                                                 ], lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_DP = torch.optim.Adam(self.netDP.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_DP2 = torch.optim.Adam(self.netDP2.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))


            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_FG)

            self.netHeCls = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'classifier', 'instance',
                                      opt.dropout_rate, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)
            self.netIhcCls = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'classifier', 'instance',
                                      opt.dropout_rate, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)
            self.netHeRe = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'rebuilder', 'instance',
                                      opt.dropout_rate, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)
            self.netIhcRe = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'rebuilder', 'instance',
                                      opt.dropout_rate, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)
            
            self.netHeCls.load_state_dict(torch.load(self.opt.he_cls))
            self.netIhcCls.load_state_dict(torch.load(self.opt.ihc_cls))
            
            self.netHeRe.load_state_dict(torch.load(self.opt.he_rec))
            self.netIhcRe.load_state_dict(torch.load(self.opt.ihc_rec))
            print('Load task models')

            self.netHeCls.eval()
            self.netHeRe.eval()
            self.netIhcCls.eval()
            self.netIhcRe.eval()

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = PatchNCELoss(opt).to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            if self.opt.weight_gp > 0:
                self.P = Gauss_Pyramid_Conv(num_high=5)
                self.criterionGP = torch.nn.L1Loss().to(self.device)
                if self.opt.gp_weights == 'uniform':
                    self.gp_weights = [1.0] * 6
                else:
                    self.gp_weights = eval(self.opt.gp_weights)

            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.hed_trs = HEDTransform().to(self.device)
            self.std_loss = VarianceConstraint().to(self.device)
            self.shiftcp_loss = ShiftContrastiveLoss().to(self.device)


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss()
            self.loss_D.backward()                  # calculate gradients for D
            self.compute_FG_loss()
            self.loss_FG.backward()                   # calculate graidents for FG
            self.compute_G_loss()
            self.loss_G.backward()                   # calculate graidents for G
            if self.opt.weight_nce_loss > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)


    def optimize_parameters(self):
        # forward
        self.forward()
        
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        # update FD
        if hasattr(self, 'fake_BF'):
            self.set_requires_grad(self.netFD, True)
            self.optimizer_FD.zero_grad()
            self.loss_FD = self.compute_FD_loss()
            self.loss_FD.backward()
            self.optimizer_FD.step()
        # update FG
        self.set_requires_grad(self.netFD, False)
        self.optimizer_FG.zero_grad()
        self.optimizer_DP2.zero_grad()
        if self.opt.weight_nce_loss > 0.0:
            if self.opt.netF == 'mlp_sample':
                self.optimizer_F.zero_grad()
        self.loss_FG = self.compute_FG_loss()
        self.loss_FG.backward()
        self.optimizer_FG.step()
        self.optimizer_DP2.step()
        if self.opt.weight_nce_loss > 0.0:
            if self.opt.netF == 'mlp_sample':
                self.optimizer_F.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_DP.zero_grad()
        if self.opt.weight_nce_loss > 0.0:
            if self.opt.netF == 'mlp_sample':
                self.optimizer_F.zero_grad()
        self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_DP.step()
        if self.opt.weight_nce_loss > 0.0:
            if self.opt.netF == 'mlp_sample':
                self.optimizer_F.step()

        self.optimize_counter = self.optimize_counter + 1


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if 'current_epoch' in input:
            self.current_epoch = input['current_epoch']
        if 'current_iter' in input:
            self.current_iter = input['current_iter']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        if self.isTrain:
            self.norm_real_A = self.adjust_brightness(self.real_A, target_brightness=0, image_min=-1)[1]
            self.norm_real_B = self.adjust_brightness(self.real_B, target_brightness=0, image_min=-1)[1]
            self.comb_fea = self.forward_muti_task_nets(self.norm_real_A, self.norm_real_B)
            self.netG.train()
        self.fake, self.st_fea = self.netG(self.real_A, output_layers=[13,15,17]) 
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        self.netD.train()
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D
    
    def compute_FD_loss(self):
        """Calculate GAN loss for the discriminator"""
        self.netFD.train()
        if self.optimize_counter % 100 == 0:
            self.netFD.load_state_dict(self.netD.state_dict())
        fake = self.fake_BF.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netFD(fake)

        self.loss_DFG_fake = self.criterionGAN(pred_fake, False).mean()

        # Real
        self.pred_real = self.netFD(self.real_B)
        loss_DFG_real = self.criterionGAN(self.pred_real, True)
        self.loss_DFG_real = loss_DFG_real.mean()

        # combine loss and calculate gradients
        self.loss_D2 = (self.loss_DFG_fake  + self.loss_DFG_real) * 0.5

        return self.loss_D2
    
    def compute_FG_loss(self):

        self.netFG1.train()
        self.netFG2.train()
        self.netG.eval()

        with torch.no_grad():
           
            self.recon_B = self.netG(self.real_B)
            self.recon_B_norm = self.adjust_brightness(self.recon_B, target_brightness=0, image_min=-1)[1]
            self.fake_B_norm = self.adjust_brightness(self.fake_B, target_brightness=0, image_min=-1)[1]
            

            self.fea_re_B = self.netIhcRe(self.fake_B_norm, output_layers=[11,18])[1]
            self.fea_re_rB = self.netIhcRe(self.recon_B_norm, output_layers=[11,18])[1]
           
            self.space_sf_img, self.channel_sf_img = self.get_cp_samples()
            fea_pos = self.netIhcRe(self.channel_sf_img, output_layers=[18])[1][18]
            fea_neg = self.netIhcRe(self.space_sf_img, output_layers=[18])[1][18]

        te_f2, shift_recon2, self.shift_map, _, _ ,cp_fea4loss= self.netFG2(self.comb_fea[18], self.fea_re_B[18],
                                                    recon_input=self.fea_re_rB[18], shift_map=None, cp_fea=[fea_pos, fea_neg])
        te_f1, shift_recon1, _, _, _, _ = self.netFG1(self.comb_fea[11], self.fea_re_B[11], 
                                  recon_input=self.fea_re_rB[11], shift_map=self.shift_map)
        
    
        self.loss_FG = 0          
        #shift loss
        self.loss_Shifter = self.criterionL1(input=self.netDP2(shift_recon2),
                                              target=self.netDP2(self.fea_re_B[18]).detach()) + \
                            self.criterionL1(input=self.netDP2(shift_recon1),
                                              target=self.netDP2(self.fea_re_B[11]).detach()) 
        self.loss_FG += self.loss_Shifter

        self.loss_fusion = self.criterionL1(input=self.netDP2(te_f2), target=self.netDP2(shift_recon2).detach()) + \
                            self.criterionL1(input=self.netDP2(te_f1), target=self.netDP2(shift_recon1).detach())
        self.loss_FG += self.loss_fusion

        self.loss_Shifter_cp = self.shiftcp_loss(cp_fea4loss)* 0.1
        self.loss_FG += self.loss_Shifter_cp
       
        self.fake_BF, _ = self.netG(self.real_A, train_fea_gen=True, input_fea={13:te_f1,  17:te_f2})
        
        
        # gan loss
        pred_fake = self.netFD(self.fake_BF)
        self.loss_FG_GAN = self.criterionGAN(pred_fake, True) * (self.opt.weight_gan_loss_FG)
        self.loss_FG += self.loss_FG_GAN

        # nce loss
        _, feat_real_A = self.netG(self.real_A, self.nce_layers)
        _, feat_fake_BF = self.netG(self.fake_BF, self.nce_layers)
        self.loss_FG_NCE = self.calculate_NCE_loss(feat_real_A, feat_fake_BF, self.netF, self.nce_layers) * self.opt.weight_nce_loss
        self.loss_FG += self.loss_FG_NCE

        # pyramid loss
        if self.opt.weight_gp > 0:
            p_fake_BF = self.P(self.fake_BF)
            p_real_B = self.P(self.real_B)
            loss_pyramid = [self.criterionGP(pf, pr) for pf, pr in zip(p_fake_BF, p_real_B)]
            weights = self.gp_weights
            loss_pyramid = [l * w for l, w in zip(loss_pyramid, weights)]
            self.loss_FG_PY = torch.mean(torch.stack(loss_pyramid)) * self.opt.weight_gp
        else:
            self.loss_FG_PY = 0
        self.loss_FG += self.loss_FG_PY
      

        return self.loss_FG


    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        self.loss_G = 0
        self.netG.eval()

        # #NCE loss
        _, feat_real_A = self.netG(self.real_A, self.nce_layers)
        _, feat_fake_B = self.netG(self.fake_B, self.nce_layers)
        self.loss_G_NCE_A = self.calculate_NCE_loss(feat_real_A, feat_fake_B, self.netF, self.nce_layers) * self.opt.weight_nce_loss
        self.loss_G_NCE = self.loss_G_NCE_A 
        self.loss_G += self.loss_G_NCE


        # Gan loss
        if self.opt.weight_gan_loss_G > 0.0:
            # self.netD.eval()
            pred_fake = self.netD(self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.weight_gan_loss_G
        else:
            self.loss_G_GAN = 0.0
        self.loss_G += self.loss_G_GAN

        # Dis loss
        self.netFG1.eval()
        self.netFG2.eval()

        with torch.no_grad():
            te_f3, _, self.shift_map, _, _, _ = self.netFG2(self.comb_fea[18], self.fea_re_B[18],
                                      recon_input=self.fea_re_rB[18], shift_map=None)
            te_f1, _, _, _, _, _ = self.netFG1(self.comb_fea[11], self.fea_re_B[11],
                                      recon_input=self.fea_re_rB[11], shift_map=self.shift_map)
            
            self.final_gen, te_fea = self.netG(self.real_A, train_fea_gen=True, output_layers=[13,15,17], 
                            input_fea={13:te_f1, 17:te_f3})
        
        self.netG.train()
        dis_loss1 = self.criterionL1(input=self.netDP(self.st_fea[13]), target=self.netDP(te_fea[13]))
        dis_loss2 = self.criterionL1(input=self.netDP(self.st_fea[15]), target=self.netDP(te_fea[15]))
        dis_loss3 = self.criterionL1(input=self.netDP(self.st_fea[17]), target=self.netDP(te_fea[17]))
               
        self.loss_G_Dis_weight = self.dis_weight_schedule()
        self.loss_G_Dis = (dis_loss1 + dis_loss2 + dis_loss3) * self.loss_G_Dis_weight
        self.loss_G +=self.loss_G_Dis

        # Pyramid loss
        if self.opt.weight_gp > 0:
            p_fake_BF = self.P(self.fake_B)
            p_real_B = self.P(self.real_B)
            loss_pyramid = [self.criterionGP(pf, pr) for pf, pr in zip(p_fake_BF, p_real_B)]
            weights = self.gp_weights
            loss_pyramid = [l * w for l, w in zip(loss_pyramid, weights)]
            self.loss_G_PY = torch.mean(torch.stack(loss_pyramid)) * self.opt.weight_gp
        else:
            self.loss_G_PY = 0
        
        self.loss_G += self.loss_G_PY
       
        # std loss
        if self.opt.weight_std_loss > 0:
            self.loss_G_Std = self.std_loss(generated=self.fake_B, target=self.real_B.detach(), weight=self.opt.weight_std_loss) 
        else:
            self.loss_G_Std = 0
        self.loss_G += self.loss_G_Std        
        
        return self.loss_G 


    def calculate_NCE_loss(self, feat_src, feat_tgt, netF, nce_layers, paired=False, detach_tgt=True):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = feat_src
        feat_k_pool, sample_ids = netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            if paired:
                loss = self.criterionASP(f_q, f_k, self.current_epoch)
            else:
                loss = self.criterionNCE(f_q, f_k, detach_tgt)
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    

    def forward_muti_task_nets(self, inputA, inputB):
        
        with torch.no_grad():
            he_cls_fea = self.netHeCls(inputA, output_layers=[11,18])[1]
            he_re_fea = self.netHeRe(inputA, output_layers=[11,18])[1]

            ihc_cls_fea = self.netIhcCls(inputB, output_layers=[11,18])[1]
            ihc_re_fea = self.netIhcRe(inputB, output_layers=[11,18])[1]

        
        return {
                    11:{'he_fea': torch.cat([he_cls_fea[11], he_re_fea[11]], dim=1),
                        'ihc_fea': torch.cat([ihc_cls_fea[11], ihc_re_fea[11]],dim=1)},
                    18:{'he_fea': torch.cat([he_cls_fea[18], he_re_fea[18]],dim=1),
                        'ihc_fea': torch.cat([ihc_cls_fea[18], ihc_re_fea[18]],dim=1)},
                }
    

    def dis_weight_schedule(self):
        initial_weight = 0
        final_weight = self.opt.weight_dis_loss
        num_iterations = 4000 * self.opt.dis_weight_schedule_epoch // self.opt.batch_size
        power = 10
        if self.optimize_counter >= num_iterations:
            return final_weight
        else:
            progress = self.optimize_counter / num_iterations
            return initial_weight + (final_weight - initial_weight) * (progress ** power)
        

    def compute_brightness(self, image):
    
        gray_image = torch.mean(image, dim=1) 
        mean_brightness = torch.mean(gray_image, dim=(1, 2))
        
        return mean_brightness
    

    def adjust_brightness(self, image, target_brightness, image_min=0):
        if image_min == -1:
            image = (image + 1) / 2
            target_brightness = (target_brightness + 1) / 2
        current_brightness = self.compute_brightness(image)
        brightness_factor = target_brightness / (current_brightness+1e-6)
        adjusted_image = torchvision.transforms.functional.adjust_brightness(image, brightness_factor)
        if image_min == -1:
            adjusted_image = adjusted_image * 2 - 1
        
        return current_brightness, adjusted_image
    

    def get_cp_samples(self):

        if not hasattr(self, 'elastic_trs') or self.optimize_counter % 100 == 0:
            x_f =random.uniform(0.5, 1.2)
            y_f =random.uniform(0.5, 1.2)
            self.elastic_trs = K.RandomElasticTransform(alpha=(x_f, y_f), p=1, padding_mode='reflection')
            x_f =random.uniform(3, 6)
            y_f =random.uniform(3, 6)
            self.elastic_trs_neg = K.RandomElasticTransform(alpha=(x_f, y_f), p=1, padding_mode='reflection')

        if not hasattr(self, 'tr_seq'):
            self.tr_seq = ImageSequential(
                RandomRotationReflection(p=0.2, degrees=30),
                K.RandomRotation90(p=0.5, times=[1,3]),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
            )

        with torch.no_grad():
            neg_p = random.random()
            if neg_p < 0.3:
                patch_list = [8,8,16,16,32,64]
                space_sf_img = shuffle_pixels_inside_patches(self.fake_B_norm, patch_size=random.choice(patch_list))
            elif 0.3 <= neg_p < 0.5:
                space_sf_img = fft_lowpass(self.fake_B_norm, cutoff_ratio=[0.01, 0.04])
            else:
                space_sf_img = self.elastic_trs_neg(self.fake_B_norm)
            
            if random.random() < 0.75:
                channel_sf_img = self.fake_B_norm
                channel_sf_img = self.hed_trs.jitter_channels(channel_sf_img, p=1, jitter_d=[0.0,1.2], jitter_h=[0.8,1.5])
                # channel_sf_img = jitter_mean_variance_ch(channel_sf_img, p=1, mean_range=0.15, var_range=0.15, per_channel=True, clamp_output=True)
                if random.random() < 0.25:
                    channel_sf_img = self.elastic_trs(channel_sf_img)
                if random.random() < 0.75:
                    channel_sf_img = self.k_patchwise_augmentation(channel_sf_img, grid_size=(8,8))
            else:
                channel_sf_img = self.real_A
        return space_sf_img, channel_sf_img
    


    def k_patchwise_augmentation(self, input, grid_size):
        # into patches
        window_size = (input.size(-2) // grid_size[-2], input.size(-1) // grid_size[-1])
        stride = window_size
        input = extract_tensor_patches(input, window_size, stride)

        # apply augmentations
        in_shape = input.shape
        input = input.reshape(-1, *in_shape[-3:])
        input = self.tr_seq(input)
        input = input.reshape(in_shape)

        # restore from patches
        input = input.view(-1, grid_size[0], grid_size[1], *input.shape[-3:])
        input = torch.cat(torch.chunk(input, grid_size[0], 1), -2).squeeze(1)
        input = torch.cat(torch.chunk(input, grid_size[1], 1), -1).squeeze(1)

        return input
    

def rot_image_patches(image, patch_size):
    
    B, C, H, W = image.shape

    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)  # (B, C, Num_Patches, H, W)

    patches = torch.rot90(patches, k=random.choice([1,2,3]), dims=(-2, -1))
    if random.random() > 0.5:
        patches = patches.flip(random.choice([-2, -1]))

    num_patches_per_row = H // patch_size
    patches = patches.view(B, C, num_patches_per_row, num_patches_per_row, patch_size, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    transformed_image = patches.view(B, C, H, W)

    return transformed_image


def jitter_mean_variance_ch(
    image,
    mean_range = 0.10,
    var_range = 0.10,
    per_channel = True,
    clamp_output = True,
    p = 0.5
):

    
    if random.random() >= p:
        return image
    else:
        reduce_dims = (-2, -1)  
        if per_channel:
            mean = image.mean(dim=reduce_dims, keepdim=True)  # [..., C, 1, 1]
            std = image.std(dim=reduce_dims, keepdim=True)
        else:
            mean = torch.mean(image,dim=None, keepdim=True) 
            std = torch.std(image, dim=None, keepdim=True)
        
        if per_channel:
            shape = (1, image.size(1), 1, 1) if image.dim() == 4 else (image.size(0), 1, 1)
        else:
            shape = (1, 1, 1, 1) if image.dim() == 4 else (1, 1, 1)
        
        mean_offset = (torch.rand(shape, device=image.device) * 2 - 1) * mean_range
        jittered = image + mean_offset
        
        var_scale = 1 + (torch.rand(shape, device=image.device) * 2 - 1) * var_range
        jittered = (jittered - mean) / (std + 1e-6) * std * var_scale + mean
        
        if clamp_output:
            if torch.min(image) >= 0 and torch.max(image) <= 1:  
                jittered = jittered.clamp(0, 1)
            elif torch.min(image) >= -1 and torch.max(image) <= 1:  
                jittered = jittered.clamp(-1, 1)
        
        perm = torch.randperm(3)  
        jittered = jittered[:, perm, :, :]  

        return jittered
    

def shuffle_pixels_inside_patches(image, patch_size=10):
    
    batch, channels, height, width = image.shape
    
    assert height % patch_size == 0 and width % patch_size == 0, \
        "Image dimensions must be divisible by patch_size"
    
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    B, C, n_w, n_h, p_w, p_h= patches.shape
    patches = patches.contiguous().view(B, C, n_h * n_w, p_h * p_w)  # [B, C, n_patches, patch_pixels]

    perm = torch.randperm(p_h * p_w)  
    patches[:, :, :, :] = patches[:, :, :, perm]  
      
    patches = patches.contiguous().view(B, C, n_h, n_w, p_h, p_w)
    patches = patches.permute(0, 1, 4, 5, 2, 3)  # [B, C, p_h, p_w, n_h, n_w]
    patches = patches.contiguous().view(B, C * p_h * p_w, -1)
    patches = torch.nn.functional.fold(
        patches,
        output_size=(height, width),
        kernel_size=(p_h, p_w),
        stride=(p_h, p_w)
    )
    
    return patches


def fft_lowpass(img_tensor, cutoff_ratio):
  
    cutoff_ratio = random.uniform(cutoff_ratio[0], cutoff_ratio[1])

    fft = torch.fft.fft2(img_tensor, dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
    
    h, w = img_tensor.shape[-2:]
    cy, cx = h // 2, w // 2
    cutoff = int(min(h, w) * cutoff_ratio)
    y = torch.arange(h, device=img_tensor.device)
    x = torch.arange(w, device=img_tensor.device)
    mask = (y[:, None] - cy)**2 + (x[None, :] - cx)**2 <= cutoff**2
    
    filtered = fft_shift * mask
    ifft = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1))
    return torch.abs(ifft)  


class DisProjector(nn.Module):
    def __init__(self, in_channels=256, out_channels=128):
        super(DisProjector, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)
    

class VarianceConstraint(nn.Module):
   
    def __init__(self, image_size=512):
        super(VarianceConstraint, self).__init__()
        self.image_size = image_size
       
    def compute_variance(self, image):
    
        mean = torch.mean(image, dim=(2, 3), keepdim=True)
        variance = torch.mean((image - mean) ** 2, dim=(2, 3))

        return variance

    def forward(self, generated, target=None, weight=1.0):
      
        generated_variance = self.compute_variance(generated)  
        target_variance = self.compute_variance(target)  
        
        variance_loss = torch.mean((generated_variance - target_variance) ** 2)  

        loss = variance_loss * self.image_size * self.image_size * weight
        if loss > 1:
            loss = loss / loss.detach()
        return loss
    


