import torch
import itertools
from .base_model import BaseModel
from . import networks
from . import vgg
import torch.nn.functional as F
import numpy as np
import skimage.measure as measure
import code
import torchvision.transforms as transforms


class IBCLNModel(BaseModel, torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=False)  # default CycleGAN did not use dropout
        parser.add_argument('--blurKernel', type=int, default=5, help='maximum R for gaussian kernel')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        torch.nn.Module.__init__(self)
        self.loss_names = ['pixel', 'res', 'adv', 'Smooth', 'MP']

        if self.isTrain:
            self.visual_names = ['fake_Ts', 'fake_Rs', 'real_T', 'real_I', 'real_R', 'mix_AB', 'fake_I']
        else:
            self.visual_names = ['fake_Ts', 'real_T', 'real_I', 'fake_Rs']

        if self.isTrain:
            self.model_names = ['G_T', 'G_R', 'G_I', 'D_T', 'D_R', 'D_I']
        else:  # during test time, only load Gs
            self.model_names = ['G_T', 'G_R', 'G_I']

        self.vgg = vgg.Vgg19(requires_grad=False).to(self.device)
        # Define generator of synthesis net
        self.netG_T = networks.define_G(opt.input_nc * 3, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG_R = networks.define_G(opt.input_nc * 3, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_T = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netD_R = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG_I = networks.define_G_I(opt.input_nc * 2, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netD_I = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        # CNN
        if self.isTrain:
            torch.nn.utils.clip_grad_norm_(self.netG_T.parameters(), 0.25)
            torch.nn.utils.clip_grad_norm_(self.netG_R.parameters(), 0.25)
            torch.nn.utils.clip_grad_norm_(self.netG_I.parameters(), 0.25)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionGradient = torch.nn.L1Loss()

            self.criterionVgg = networks.VGGLoss1(self.device, vgg=self.vgg, normalize=False)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_T.parameters(), self.netG_R.parameters(),
                                                                self.netG_I.parameters()), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_T.parameters(), self.netD_I.parameters(),
                                                                self.netD_R.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.criterionIdt = torch.nn.MSELoss()

        resSize = 64
        self.k_sz = np.linspace(opt.batch_size, self.opt.blurKernel, 80)  # for synthetic images
        self.t_h = torch.zeros(opt.batch_size, opt.ngf * 4, resSize, resSize).to(self.device)
        self.t_c = torch.zeros(opt.batch_size, opt.ngf * 4, resSize, resSize).to(self.device)
        self.r_h = torch.zeros(opt.batch_size, opt.ngf * 4, resSize, resSize).to(self.device)
        self.r_c = torch.zeros(opt.batch_size, opt.ngf * 4, resSize, resSize).to(self.device)

        self.fake_R = torch.zeros(self.opt.batch_size, 3, 256, 256).to(self.device)
        self.fake_Rs = [self.fake_R]

        self.fake_T = torch.zeros(self.opt.batch_size, 3, 256, 256).to(self.device)
        self.fake_Ts = [self.fake_T]

        self.fake_I = torch.zeros(self.opt.batch_size, 3, 256, 256).to(self.device)
        self.fake_Is = [self.fake_I]

        # Pass invalid data
        self.trainFlag = True

        self.real_R = None
        self.real_I = None
        self.real_T = None
        self.real_T2 = None
        self.real_T4 = None
        self.alpha = None
        self.One = None

    def set_input(self, input):
        with torch.no_grad():
            if self.isTrain:
                self.real_T2 = input['T2'].to(self.device)
                self.real_T4 = input['T4'].to(self.device)
                I = input['I']
                T = input['T']
                R = input['R']
                self.real_R = R.to(self.device)
            else:  # Test
                self.image_paths = input['B_paths']
                I = input['I']
                T = input['T']

        self.real_T = T.to(self.device)
        self.real_I = I.to(self.device)
        self.One = torch.ones(self.real_I.shape).to(self.device)

    def get_c(self):
        b, c, w, h = self.real_I.shape
        return torch.zeros((b, self.opt.ngf * 4, w // 4, h // 4))

    def init(self):
        self.t_h = self.get_c()
        self.t_c = self.get_c()
        self.r_h = self.get_c()
        self.r_c = self.get_c()
        self.fake_T = torch.tensor(self.real_I)
        self.fake_Ts = [self.fake_T]
        self.fake_R = torch.ones_like(self.real_I) * 0.1
        self.fake_Rs = [self.fake_R]

    def forward(self):
        self.init()
        i = 0
        while i <= 2:
            self.fake_T, self.t_h, self.t_c, self.fake_T2, self.fake_T4 = self.netG_T(
                torch.cat((self.real_I, self.fake_Ts[-1], self.fake_Rs[-1]), 1), self.t_h, self.t_c)
            self.fake_Ts.append(self.fake_T)
            self.fake_R, self.r_h, self.r_c, self.fake_R2, self.fake_R4 = self.netG_R(
                torch.cat((self.real_I, self.fake_Ts[-1], self.fake_Rs[-1]), 1), self.r_h, self.r_c)
            self.fake_Rs.append(self.fake_R)
            i += 1

        self.fake_I = self.netG_I(torch.cat((self.fake_T, self.fake_R), 1))
        self.fake_I_revise = self.One - self.fake_I
        self.mix_AB = self.fake_I_revise * self.fake_T + self.fake_I * self.fake_R

        # clip operation in test
        if not self.isTrain:
            for i in range(len(self.fake_Ts)):
                self.fake_Ts[i] = torch.clamp(self.fake_Ts[i], min=0, max=1)
            for i in range(len(self.fake_Rs)):
                self.fake_Rs[i] = torch.clamp(self.fake_Rs[i], min=0, max=1)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.loss_D_I = self.backward_D_basic(self.netD_I, self.real_I, self.mix_AB)
        self.loss_D_T = self.backward_D_basic(self.netD_T, self.real_T, self.fake_T)
        self.loss_D_R = self.backward_D_basic(self.netD_R, self.real_R, self.fake_R)

    def backward_G(self):
        self.loss_idt_T = 0.0  # L_pixel on T
        self.loss_idt_R = 0.0  # L_pixel on R
        self.loss_MP = 0.0  # L_MP: multi-scale perceptual loss
        self.loss_idt_I = 0.0  # L_residual: residual reconstruction loss
        self.loss_Smooth = 0.0
        iter_num = len(self.fake_Ts)

        sigma = 1
        self.loss_I = self.criterionIdt(self.fake_I, self.real_I)

        for i in range(iter_num):
            if i > 0:
                self.loss_idt_T += self.criterionIdt(self.fake_Ts[i], self.real_T) * np.power(sigma, iter_num - i)
                self.loss_idt_R += self.criterionIdt(self.fake_Rs[i], self.real_R) * np.power(sigma, iter_num - i) * 10

        self.loss_MP = self.criterionVgg(self.fake_T, self.real_T) \
                       + 0.8 * self.criterionVgg(self.fake_T2, self.real_T2) \
                       + 0.6 * self.criterionVgg(self.fake_T4, self.real_T4)

        self.loss_G_T = self.criterionGAN(self.netD_T(self.fake_T), True) * 0.01  # L_adv: adversarial loss
        self.loss_G_R = self.criterionGAN(self.netD_R(self.fake_R), True) * 0.01
        self.loss_G_I = self.criterionGAN(self.netD_I(self.mix_AB), True) * 0.01

        self.loss_pixel = self.loss_idt_T + self.loss_idt_R
        self.loss_adv = self.loss_G_I + self.loss_G_T + self.loss_G_R
        self.loss_res = self.loss_I

        # for smoothness loss
        smooth_y = self.criterionIdt(self.fake_I[:, :, 1:, :], self.fake_I.detach()[:, :, :-1, :])
        smooth_x = self.criterionIdt(self.fake_I[:, :, :, 1:], self.fake_I.detach()[:, :, :, :-1])
        self.loss_Smooth = smooth_x + smooth_y

        self.loss = 2*self.loss_pixel + 2*self.loss_res + self.loss_adv + 0.1*self.loss_Smooth + self.loss_MP

        self.loss.backward()

    def optimize_parameters(self):
        # Pass invalid data
        if not self.trainFlag:
            self.trainFlag = True
            return

        self.optimizer_G.zero_grad()
        self.set_requires_grad([self.netD_T, self.netD_I, self.netD_R], False)  # Ds require no gradients when optimizing Gs
        self.forward()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD_T, self.netD_I, self.netD_R], True)
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()
