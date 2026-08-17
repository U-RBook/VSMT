from html import parser
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

class RECModel(BaseModel):


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
        parser.add_argument('--norm_input', type=util.str2bool, default=False)
        parser.add_argument('--colorjitter', type=util.str2bool, default=False)
        parser.add_argument('--norm_output', type=util.str2bool, default=True)
        parser.add_argument('--img_type', type=str, default='HE', choices=['HE', 'IHC'])





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
        self.loss_names = ['Re']
    

        if self.opt.isTrain:
            self.visual_names = ['recon_img', 'img']
        else:
            self.visual_names = ['recon_img']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['Re']
        else:  
            self.model_names = ['Re'] # during test time, only load G
        self.img_type = opt.img_type
        self.netRe = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'rebuilder', 'instance', opt.dropout_rate, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)
        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_Re = torch.optim.Adam(self.netRe.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_Re)

            

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.forward()                     # compute fake images: G(A)


    def optimize_parameters(self):
        # forward
        self.forward()
        # update G
        self.optimizer_Re.zero_grad()
        self.compute_Re_loss()
        self.loss_Re.backward()
        self.optimizer_Re.step()

        self.optimize_counter = self.optimize_counter + 1


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """

        if self.img_type == 'HE':
            self.img = input['A'].to(self.device)
        elif self.img_type == 'IHC':
            self.img = input['B'].to(self.device)
        else:
            raise ValueError('img_type should be HE or IHC')

        if 'current_epoch' in input:
            self.current_epoch = input['current_epoch']
        if 'current_iter' in input:
            self.current_iter = input['current_iter']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.netRe.train()
        self.recon_img = self.netRe(self.img)


    def compute_Re_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        self.loss_Re = 0
        self.loss_Re = self.criterionL2(self.recon_img, self.img.detach())
        
        return self.loss_Re


    