import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .base_model import BaseModel
from . import networks
import util.util as util

class CLSModel(BaseModel):


    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--netF_nc', type=int, default=256)
       
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.set_defaults(pool_size=0, dataset_mode='cls')  # no image pooling

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
        parser.add_argument('--distill_temperature', type=float, default=4.0)
        parser.add_argument('--distill_weight', type=float, default=0.99)
        




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
        self.loss_names = ['Cls', 'Cls_T', 'CE', 'KD']
        self.visual_names = ['img']
        self.model_names = ['Cls', 'Cls_T'] if self.isTrain else ['Cls']
        self.img_type = opt.img_type
        self.distill_temperature = opt.distill_temperature
        self.distill_weight = opt.distill_weight
        self.netCls_T = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.netCls_T.fc = nn.Linear(self.netCls_T.fc.in_features, 4)
        self.netCls_T = self.netCls_T.to(self.device)
        self.imagenet_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device
        ).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device
        ).view(1, 3, 1, 1)
        self.netCls = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, 'classifier', 'instance',
            opt.dropout_rate, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids
        )
        if self.isTrain:
            # class_weights = torch.tensor(
            #         [4.8218, 1.0598, 0.5740, 0.9035],
            #         dtype=torch.float32,
            #         device=self.device,
            #     )
            # self.criterionCls = nn.CrossEntropyLoss(weight=class_weights)
            self.criterionCls = nn.CrossEntropyLoss()
            self.optimizer_Cls_T = torch.optim.Adam(
                self.netCls_T.parameters(),
                lr=opt.lr/5,
                betas=(opt.beta1, opt.beta2),
            )
            self.optimizer_Cls = torch.optim.Adam(self.netCls.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_Cls_T)
            self.optimizers.append(self.optimizer_Cls)

            

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
        # update teacher and student
        self.optimizer_Cls_T.zero_grad()
        self.optimizer_Cls.zero_grad()
        self.compute_Cls_T_loss()
        self.compute_Cls_loss()
        self.loss_Cls_T.backward()
        self.loss_Cls.backward()
        self.optimizer_Cls_T.step()
        self.optimizer_Cls.step()

        self.optimize_counter = self.optimize_counter + 1


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """

        if self.img_type == 'HE':
            self.img = input['A'].to(self.device)
            self.cls_label = input['cls_label_A'].to(self.device).long().view(-1)
            self.image_paths = input['A_paths']
        elif self.img_type == 'IHC':
            self.img = input['B'].to(self.device)
            self.cls_label = input['cls_label_B'].to(self.device).long().view(-1)
            self.image_paths = input['B_paths']

        else:
            raise ValueError('img_type should be HE or IHC')

        if 'current_epoch' in input:
            self.current_epoch = input['current_epoch']
        if 'current_iter' in input:
            self.current_iter = input['current_iter']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.cls_logits = self.netCls(self.img)
        self.cls_pred = self.cls_logits.argmax(dim=1)
        if self.isTrain:
            teacher_img = self.img * 0.5 + 0.5
            teacher_img = (
                teacher_img - self.imagenet_mean
            ) / self.imagenet_std
            self.cls_logits_T = self.netCls_T(teacher_img)


    def compute_Cls_T_loss(self):
        """Calculate the teacher classification loss."""
        self.loss_Cls_T = self.criterionCls(
            self.cls_logits_T, self.cls_label
        )
        return self.loss_Cls_T


    def compute_Cls_loss(self):
        """Calculate the student classification and distillation losses."""
        temperature = self.distill_temperature
        self.loss_CE = self.criterionCls(self.cls_logits, self.cls_label)
        self.loss_KD = F.kl_div(
            F.log_softmax(self.cls_logits / temperature, dim=1),
            F.softmax(self.cls_logits_T.detach() / temperature, dim=1),
            reduction='batchmean',
        ) * (temperature ** 2)
        self.loss_Cls = (
            (1.0 - self.distill_weight) * self.loss_CE
            + self.distill_weight * self.loss_KD
        )
        
        return self.loss_Cls

