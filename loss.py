import torch.nn as nn
import vgg
from training.loss import *


class ProxyLoss:
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G

        # l1 loss
        self.l1_criterion = nn.L1Loss()

        # VGG loss
        self.vgg19 = vgg.get_vgg19()
        self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.style_weights = [1 / n ** 2 for n in [64, 128, 256, 512, 512]]
        self.feature_layers = ['r22', 'r32', 'r42']
        self.feature_weights = [1e-3, 1e-3, 1e-3]
        self.criterion_feature = vgg.WeightedLoss(self.feature_weights, metric='l1')
        self.criterion_style = vgg.WeightedLoss(self.style_weights, metric='l1')
        self.feat_w = 10.0
        self.style_w = 1.0

        print(f"Loss Config: l1_w = 1, feat_w = {self.feat_w}, style_w = {self.style_w}")

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def accumulate_gradients(self, real_img, real_c, gen_z):
        with torch.autograd.profiler.record_function('Gmain_forward'):
            gen_img, _gen_ws = self.run_G(gen_z, real_c)

            # l1 loss
            loss_l1 = self.l1_criterion(gen_img, real_img)
            training_stats.report('Loss/L1/loss', loss_l1)

            # VGG loss
            real_feat, real_style = self.vgg19.extract_features(real_img, self.feature_layers, self.style_layers,
                                                                detach_features=True, detach_styles=True)
            recon_feat, recon_style = self.vgg19.extract_features(gen_img, self.feature_layers, self.style_layers)
            feature_loss = self.criterion_feature(real_feat, recon_feat) * self.feat_w
            style_loss = self.criterion_style(real_style, recon_style) * self.style_w
            training_stats.report('Loss/Feat/loss', feature_loss)
            training_stats.report('Loss/Style/loss', style_loss)

            loss = loss_l1 + feature_loss + style_loss
            training_stats.report('Loss/G/loss', loss)

        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss.backward()


class ProxyGANLoss:
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.r1_gamma           = r1_gamma
        self.pl_weight          = 0  # do not include path length regularization
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        # l1 loss
        self.l1_criterion = nn.L1Loss()

        # VGG loss
        self.vgg19 = vgg.get_vgg19()
        self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.style_weights = [1 / n ** 2 for n in [64, 128, 256, 512, 512]]
        self.feature_layers = ['r22', 'r32', 'r42']
        self.feature_weights = [1e-3, 1e-3, 1e-3]
        self.criterion_feature = vgg.WeightedLoss(self.feature_weights, metric='l1')
        self.criterion_style = vgg.WeightedLoss(self.style_weights, metric='l1')

        self.l1_w = 1.0
        self.feat_w = 1.0
        self.style_w = 1.0
        self.gan_w = 0.1

        print(f"Loss Config: l1_w = {self.l1_w}, feat_w = {self.feat_w}, style_w = {self.style_w}, gan_w = {self.gan_w}")

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, real_c)
                gen_logits = self.run_D(gen_img, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                # l1 loss
                loss_l1 = self.l1_criterion(gen_img, real_img) * self.l1_w
                training_stats.report('Loss/L1/loss', loss_l1)

                # VGG loss
                real_feat, real_style = self.vgg19.extract_features(real_img, self.feature_layers, self.style_layers,
                                                                    detach_features=True, detach_styles=True)
                recon_feat, recon_style = self.vgg19.extract_features(gen_img, self.feature_layers, self.style_layers)
                feature_loss = self.criterion_feature(real_feat, recon_feat) * self.feat_w
                style_loss = self.criterion_style(real_style, recon_style) * self.style_w
                training_stats.report('Loss/Feat/loss', feature_loss)
                training_stats.report('Loss/Style/loss', style_loss)

                # GAN loss
                loss_G = torch.nn.functional.softplus(-gen_logits).mean()*self.gan_w # -log(sigmoid(gen_logits))
                training_stats.report('Loss/GAN/loss', loss_G)

                loss_Gmain = loss_G + loss_l1 + feature_loss + style_loss
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mul(gain).backward()
                # loss_Gmain.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, real_c, update_emas=True)
                gen_logits = self.run_D(gen_img, real_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()
