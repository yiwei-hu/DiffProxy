import sys
import os.path as pth
import torch.nn as nn
import torch
import pickle
import cv2
from utils import read_image
from sbs_generators import generator_lookup_table, SBSGenerators
sys.path.append('stylegan')

sat_dir = 'C:/Program Files/Allegorithmic/Substance Automation Toolkit'
sbs_lib = {'arc_pavement': './data/sbs/arc_pavement.sbs'}


def get_real_generator(generator_name):
    if generator_name in sbs_lib:
        graph_filename = sbs_lib[generator_name]
        return generator_lookup_table[generator_name](graph_filename, 'generator', sat_dir, 256)
    else:
        raise RuntimeError(f'Cannot find {generator_name}\'s real signature')


class DifferentiableGenerator(nn.Module):
    def output(self, *arg):
        return self.forward()

    def sample_params(self):
        return None

    def get_params(self):
        return None

    def set_params(self, params):
        pass

    def regularize(self):
        pass

    def get_optimizable(self):
        raise NotImplementedError


class StyleGANCond(DifferentiableGenerator):
    @staticmethod
    def load_pretrained_model(model_path):
        with open(model_path, 'rb') as f:
            dat = pickle.load(f)
            if 'G_ema' in dat:
                G = dat['G_ema']
            else:
                G = dat['G']
            G = G.eval().requires_grad_(False).cuda()  # torch.nn.Module
        return G

    def get_normalizer(self):
        params = generator_lookup_table[self.generator_name].get_params()
        normalizer = SBSGenerators.get_normalizer(params, batch_size=1, device=self.device)
        return normalizer

    def get_standarizer(self):
        params = generator_lookup_table[self.generator_name].get_params()
        normalizer = SBSGenerators.get_standarizer(params, batch_size=1, device=self.device)
        return normalizer

    def get_reguluarizer(self):
        params = generator_lookup_table[self.generator_name].get_params()
        regularizer  = SBSGenerators.get_regularizer(params, batch_size=1, device=self.device)
        return regularizer

    def init_avg(self):
        assert self.normalization is not None
        if self.normalization == 'norm':
            self.set_params((self.normalizer.min_ + self.normalizer.max_)*0.5)
        else:
            self.set_params(self.normalizer.mean)

    def init_rand(self):
        params = self.sample_params()
        self.set_params(params)

    def sample_params(self):
        if self.normalization == 'norm':
            params = torch.rand_like(self.normalizer.min_) * (self.normalizer.max_ - self.normalizer.min_) + self.normalizer.min_
        else:
            params = torch.randn_like(self.normalizer.mean) * self.normalizer.std + self.normalizer.mean
            self.regularizer.regularize_(params)
        return params

    def init_direct(self, init_params):
        params = torch.as_tensor(init_params, device=self.device, dtype=torch.float64).unsqueeze(0)
        assert params.ndim == 2
        self.set_params(params)

    def __init__(self, generator_name, model_path, init, model_type='none'):
        super(StyleGANCond, self).__init__()

        self.model_path = model_path
        self.generator_name = generator_name
        self.init = init
        self.G = self.load_pretrained_model(model_path)
        self.img_resolution = self.G.img_resolution

        self.real_G = None
        self.params = None
        self.optimizable = None
        self.device = torch.device('cuda')

        self.model_type = model_type

        assert model_type in ['norm', 'std']
        self.normalization = model_type
        if self.normalization == 'norm':
            self.normalizer = self.get_normalizer()
        else:
            self.normalizer = self.get_standarizer()
        self.regularizer = self.get_reguluarizer()

        init_method = init['method']
        if init_method == 'avg':
            self.init_avg()
        elif init_method == 'rand':
            self.init_rand()
        elif init_method == 'direct':
            self.init_direct(init['init_params'])
        else:
            raise RuntimeError("Unknown initialization method")

    def get_optimizable(self):
        return self.optimizable

    def forward(self):
        ws = self.G.mapping(None, self.params)
        im = self.G.synthesis(ws)
        im = (im + 1.0) / 2.0  # (-1, 1) to (0, 1)
        return im.clamp(0, 1)

    def regularize(self):
        params = self.get_params()
        params = self.regularizer.regularize(params)
        params = self.normalizer.normalize(params)
        self.params.data = params.data

    def get_params(self):
        return self.normalizer.denormalize(self.params)

    def set_params(self, params, validate=True):
        if validate:
            # params are unnormalized
            self.regularizer.check_valid(params)

        self.params = self.normalizer.normalize(params).detach().clone()
        self.params.requires_grad = True
        self.optimizable = [self.params]

    # get real generator map
    def output(self, tmp_dir, tmp_image_filename):
        if self.real_G is None:
            self.real_G = get_real_generator(self.generator_name)
        params = self.get_params()[0].detach().cpu().tolist()

        json_file = [(tmp_image_filename, params)]
        self.real_G.sample_with_json(tmp_dir, json_file)

        # reload image from disk
        image_filename = pth.join(tmp_dir, tmp_image_filename)
        image_np = read_image(image_filename)
        image_np = cv2.resize(image_np, (self.G.img_resolution, self.G.img_resolution))
        assert image_np.ndim == 2
        image = torch.as_tensor(image_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        return image