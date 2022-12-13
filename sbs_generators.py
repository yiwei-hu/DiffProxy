import os
import os.path as pth
import numbers
import shutil
import glob
import random
import torch
import numpy as np
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
import json
import subprocess
from collections import OrderedDict
from utils import Timer, read_image, write_image


class SimpleSBSGraph:
    def __init__(self, sbs_file_name, params):
        self.sbs_file_name = sbs_file_name
        self.params = params
        self.xml_tree, self.n_nodes, self.node_params_dict = self.parse()

    @staticmethod
    def set_param(param_val_xml, val, is_int):
        if isinstance(val, (list, tuple)):
            if not all(isinstance(x, numbers.Number) for x in val):
                raise RuntimeError('Unknown parameter type.')
            param_val_str = ' '.join(str(x) for x in val)
        else:
            raise RuntimeError(f'Unknown parameter type: {type(val)}')

        param_tag = f'constantValueInt{len(val)}' if is_int else f'constantValueFloat{len(val)}'
        if len(val) == 1 and is_int:
            param_tag = 'constantValueInt32'
        param_val_xml_ = param_val_xml.find(param_tag)
        if param_val_xml_ is None:
            ET.SubElement(param_val_xml, param_tag).set('v', param_val_str)
        else:
            param_val_xml_.set('v', param_val_str)

    def parse(self):
        # Parse XML file
        doc_xml = ET.parse(self.sbs_file_name)
        graph_xml = doc_xml.getroot().find('content/graph')

        # find graph outputs
        graph_outputs_by_uid = {}
        for output_ in graph_xml.iter('graphoutput'):
            output_name = output_.find('identifier').get('v')
            output_uid = output_.find('uid').get('v')
            graph_outputs_by_uid[output_uid] = output_name

        n_nodes = 0
        node_params_dict = OrderedDict()
        output_name_list = []

        # check generator nodes
        for node_xml in graph_xml.iter('compNode'):
            node_uid = int(node_xml.find('uid').get('v'))
            node_imp = node_xml.find('compImplementation')[0]

            if node_imp.tag == 'compInstance':
                node_params = {}
                for param_xml in node_imp.iter('parameter'):
                    param_name = param_xml.find('name').get('v')
                    if param_name in self.params:
                        node_params[param_name] = param_xml.find('paramValue')

                # add unregistered params to nodes
                unregistered_param_names = set(self.params) - set(node_params)
                print(f"In Node {n_nodes}, found registered params:{set(node_params)}")
                print(f"In Node {n_nodes}, found unregistered params: {unregistered_param_names}")

                params = node_imp.find('parameters')
                for param_name in unregistered_param_names:
                    param_xml = ET.SubElement(params, 'parameter')
                    ET.SubElement(param_xml, 'name').set('v', param_name)
                    param_val_xml = ET.SubElement(param_xml, 'paramValue')
                    self.set_param(param_val_xml,
                                   val=self.params[param_name].default_val,
                                   is_int=self.params[param_name].is_discrete)
                    node_params[param_name] = param_val_xml

                n_nodes += 1
                node_params_dict[node_uid] = {'params': node_params, 'name': None}

            elif node_imp.tag == 'compOutputBridge':
                pass

            else:
                raise NotImplementedError(f'This simple sbs parse cannot recognize this types of node: {node_imp.tag}')

        for node_xml in graph_xml.iter('compNode'):
            node_imp = node_xml.find('compImplementation')[0]
            if node_imp.tag == 'compInstance':
                pass
            elif node_imp.tag == 'compOutputBridge':
                output_uid = node_imp.find('output').get('v')
                output_name_list.append(graph_outputs_by_uid[output_uid])
                connections = node_xml.findall('connections/connection')
                if len(connections) != 1:
                    raise RuntimeError('A output node is not connected.')
                gen_uid = int(connections[0].find('connRef').get('v'))
                if gen_uid not in node_params_dict:
                    raise RuntimeError('Cannot find input generator node for this output node.')
                node_params_dict[gen_uid]['name'] = graph_outputs_by_uid[output_uid]
            else:
                raise NotImplementedError(f'This simple sbs parse cannot recognize this types of node: {node_imp.tag}')

        return doc_xml, n_nodes, node_params_dict


class Sampler(ABC):
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def size(self):
        pass


class RandomSampler(Sampler):
    def __init__(self, min_val, max_val, default_val=None, is_discrete=False):
        self.min_val = min_val
        self.max_val = max_val
        self.default_val = default_val if default_val is not None else self.min_val
        self.is_discrete = is_discrete
        if is_discrete:
            self.func = random.randint
        else:
            self.func = random.uniform

    def sample(self):
        val = []
        for min_v, max_v in zip(self.min_val, self.max_val):
            val.append(self.func(min_v, max_v))
        return val

    def size(self):
        return len(self.min_val)


class GaussianRandomSampler(Sampler):
    def __init__(self, min_val, max_val, mean_val, std_val, default_val=None, is_discrete=False):
        self.min_val = min_val
        self.max_val = max_val
        self.mean_val = mean_val
        self.std_val = std_val
        self.default_val = default_val if default_val is not None else mean_val
        self.is_discrete = is_discrete

    def get_sample_np(self):
        val = np.random.normal(self.mean_val, self.std_val)
        val = np.clip(val, self.min_val, self.max_val)
        return val

    def sample(self):
        val = self.get_sample_np()
        if self.is_discrete:
            val = np.rint(val).astype(np.int)
        return val.tolist()

    def size(self):
        return len(self.mean_val)


class ParameterNormalizer:
    def __init__(self, min_, max_):
        self.min_ = min_.clone()
        self.max_ = max_.clone()
        self.range = self.max_ - self.min_

    def normalize(self, x):
        return torch.nan_to_num((x - self.min_) / self.range)

    def denormalize(self, x):
        return x * self.range + self.min_

    def __str__(self):
        return 'Parameter Normalizer'


class ParameterStandarizer:
    def __init__(self, mean, std):
        self.mean = mean.clone()
        self.std = std.clone()

    def normalize(self, x):
        return torch.nan_to_num((x - self.mean) / self.std)

    def denormalize(self, x):
        return x * self.std + self.mean

    def __str__(self):
        return 'Parameter Standarizer'

class ParameterRegularizer:
    def __init__(self, min_, max_):
        self.min_ = min_.clone()
        self.max_ = max_.clone()

    def regularize(self, x):
        return torch.clamp(x, self.min_, self.max_)

    def regularize_(self, x):
        x.clamp_(self.min_, self.max_)

    def check_valid(self, x):
        all_min = x >= self.min_
        if not torch.all(all_min):
            l = x.shape[1]
            for k in range(l):
                i, j = x[0, k], self.min_[0, k]
                if i < j:
                    print(f'For {k}th params: {i} < {j}')
            raise RuntimeError('Invalid parameters')

        all_max = x <= self.max_
        if not torch.all(all_max):
            l = x.shape[1]
            for k in range(l):
                i, j = x[0, k], self.max_[0, k]
                if i > j:
                    print(f'For {k}th params: {i} > {j}')
            raise RuntimeError('Invalid parameters')


def get_normalizer(generator_name, normalization_type, batch_size, device):
    if normalization_type == 'norm':
        params = generator_lookup_table[generator_name].get_params()
        normalizer = SBSGenerators.get_normalizer(params, batch_size, device)
    elif normalization_type == 'std':
        params = generator_lookup_table[generator_name].get_params()
        normalizer = SBSGenerators.get_standarizer(params, batch_size, device)
    else:
        normalizer = None
    return normalizer


class SBSGenerators:
    @staticmethod
    def get_params():
        pass

    @staticmethod
    def get_normalizer(params, batch_size, device):
        min_, max_ = [], []
        for param_name, param_sampler in params.items():
            min_val = param_sampler.min_val
            max_val = param_sampler.max_val
            min_.extend(min_val)
            max_.extend(max_val)

        min_tensor = torch.as_tensor(min_, dtype=torch.float64, device=device)
        max_tensor = torch.as_tensor(max_, dtype=torch.float64, device=device)
        min_tensor = min_tensor.expand((batch_size, -1))
        max_tensor = max_tensor.expand((batch_size, -1))

        return ParameterNormalizer(min_tensor, max_tensor)

    @staticmethod
    def get_standarizer(params, batch_size, device):
        mean, std = [], []
        for param_name, param_sampler in params.items():
            mean_v = param_sampler.mean_val
            std_v = param_sampler.std_val
            mean.extend(mean_v)
            std.extend(std_v)

        mean_tensor = torch.as_tensor(mean, dtype=torch.float64, device=device)
        std_tensor = torch.as_tensor(std, dtype=torch.float64, device=device)
        mean_tensor = mean_tensor.expand((batch_size, -1))
        std_tensor = std_tensor.expand((batch_size, -1))
        return ParameterStandarizer(mean_tensor, std_tensor)

    @staticmethod
    def get_regularizer(params, batch_size, device):
        min_, max_ = [], []
        for param_name, param_sampler in params.items():
            min_val = param_sampler.min_val
            max_val = param_sampler.max_val
            min_.extend(min_val)
            max_.extend(max_val)

        min_tensor = torch.as_tensor(min_, dtype=torch.float64, device=device)
        max_tensor = torch.as_tensor(max_, dtype=torch.float64, device=device)
        min_tensor = min_tensor.expand((batch_size, -1))
        max_tensor = max_tensor.expand((batch_size, -1))

        return ParameterRegularizer(min_tensor, max_tensor)

    def __init__(self, graph_filename, graph_name, sat_dir, image_res):
        self.grah_filename = graph_filename
        self.graph_name = graph_name
        self.sat_dir = sat_dir
        self.image_res = image_res

        # load parameters
        self.params = self.get_params()

        # load graph
        self.graph = SimpleSBSGraph(graph_filename, self.params)

    @staticmethod
    def save_params(all_params, all_image_names, output_dir, i_batch):
        assert len(all_params) == len(all_image_names)
        data = dict()
        data['labels'] = []

        for params, image_name in zip(all_params, all_image_names):
            data['labels'].append([image_name, params])

        with open(pth.join(output_dir, f'dataset{i_batch}.json'), 'w') as outfile:
            json.dump(data, outfile, indent=4)

    @staticmethod
    def combine_params(input_path, n_batch, move_to_folder=None):
        assert (n_batch >= 1)
        if move_to_folder is not None:
            os.makedirs(move_to_folder, exist_ok=True)

        data_path = pth.join(input_path, 'dataset0.json')
        with open(data_path) as f:
            data = json.load(f)

        if move_to_folder is not None:
            shutil.move(data_path, pth.join(move_to_folder, f'dataset0.json'))

        for i in range(1, n_batch):
            data_path = pth.join(input_path, f'dataset{i}.json')
            with open(data_path) as f:
                data_i = json.load(f)
            data['labels'].extend(data_i['labels'])

            if move_to_folder is not None:
                shutil.move(data_path, pth.join(move_to_folder, f'dataset{i}.json'))

        output_path = pth.join(input_path, 'dataset.json')
        with open(output_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def sample(self, output_dir, n_samples, vis_every):
        n_samples = n_samples // self.graph.n_nodes * self.graph.n_nodes
        n_batch = n_samples // self.graph.n_nodes

        timer = Timer()
        timer.begin("Begin Sampling")

        for i in range(n_batch):
            params_list = []

            # sample parameters
            for nodes in self.graph.node_params_dict.values():
                params = []
                for param_name, param_sampler in self.params.items():
                    val = param_sampler.sample()
                    self.graph.set_param(nodes['params'][param_name], val, param_sampler.is_discrete)
                    params.extend(val)

                params_list.append(params)

            # save sbs
            output_graph_filename = pth.join(output_dir, f'tmp{i}.sbs')
            self.save_graph(output_graph_filename)
            image_names_list = self.save_sample(output_graph_filename, output_dir, i)
            self.save_params(params_list, image_names_list, output_dir, i)

            if i % vis_every == 0 or i == n_batch - 1:
                timer.end(f'Generated {(i+1)*self.graph.n_nodes}/{n_samples} samples')
                timer.begin()

        # combine parameter json into one file
        self.combine_params(output_dir, n_batch, move_to_folder=pth.join(output_dir, 'params'))

        # move generated sbs and sbsar files to an sbs folder
        sbs_files = glob.glob(pth.join(output_dir, '*.sbs')) + glob.glob(pth.join(output_dir, '*.sbsar'))
        sbs_out_dir = pth.join(output_dir, 'sbs')
        if pth.exists(sbs_out_dir):
            shutil.rmtree(sbs_out_dir)
        os.makedirs(sbs_out_dir)
        for sbs_file in sbs_files:
            shutil.move(sbs_file, sbs_out_dir)

    def sample_with_json(self, output_dir, json_file):
        if isinstance(json_file, str):
            with open(json_file) as f:
                params = json.load(f)['labels']
        else:
            params = json_file

        n_samples = len(params)
        n_batch = n_samples // self.graph.n_nodes

        timer = Timer()
        timer.begin("Begin Sampling")

        for i in range(n_batch):
            image_names = []
            for k, node in enumerate(self.graph.node_params_dict.values()):
                idx = i * self.graph.n_nodes + k

                # set parameters
                s = 0
                for param_name, param_sampler in self.params.items():
                    r = param_sampler.size()
                    val = [int(np.rint(x)) if param_sampler.is_discrete else x for x in params[idx][1][s:s + r]]
                    self.graph.set_param(node['params'][param_name], val, param_sampler.is_discrete)
                    s += r

                # record image name
                image_name = pth.join(output_dir, pth.basename(params[idx][0]))
                image_names.append(image_name)

            # save sbs
            output_graph_filename = pth.join(output_dir, f'tmp{i}.sbs')
            self.save_graph(output_graph_filename)
            self.save_sample(output_graph_filename, output_dir, i, image_names)

        # move generated sbs and sbsar files to an sbs folder
        sbs_files = glob.glob(pth.join(output_dir, '*.sbs')) + glob.glob(pth.join(output_dir, '*.sbsar'))
        sbs_out_dir = pth.join(output_dir, 'sbs')
        if pth.exists(sbs_out_dir):
            shutil.rmtree(sbs_out_dir)
        os.makedirs(sbs_out_dir)
        for sbs_file in sbs_files:
            shutil.move(sbs_file, sbs_out_dir)

    # save sbs graph back to an sbs file
    def save_graph(self, output_graph_filename):
        self.graph.xml_tree.write(output_graph_filename)

    # cook and output images
    def save_sample(self, input_graph_filename, output_dir, i_batch, image_names=None):
        tmp_output_dir = pth.join(output_dir, 'tmp')
        os.makedirs(tmp_output_dir, exist_ok=True)

        command_cooker = (
            f'"{os.path.join(self.sat_dir, "sbscooker")}" '
            f'--inputs "{input_graph_filename}" '
            f'--alias "sbs://{os.path.join(self.sat_dir, "resources", "packages")}" '
            f'--output-path {{inputPath}}')

        completed_process = subprocess.run(command_cooker, shell=True, capture_output=True, text=True)
        if completed_process.returncode != 0:
            raise RuntimeError(f'Error while running sbs cooker:\n{completed_process.stderr}')
        # import pdb; pdb.set_trace()

        cooked_input_graph_filename = pth.splitext(input_graph_filename)[0] + '.sbsar'
        image_format = 'png'
        command_render = (
            f'"{os.path.join(self.sat_dir, "sbsrender")}" render '
            f'--inputs "{cooked_input_graph_filename}" '
            f'--input-graph "{self.graph_name}" '
            f'--output-format "{image_format}" '
            f'--output-path "{tmp_output_dir}" '
            f'--output-name "{{outputNodeName}}"')

        completed_process = subprocess.run(command_render, shell=True, capture_output=True, text=True)
        if completed_process.returncode != 0:
            raise RuntimeError(f'Error while running sbs render:\n{completed_process.stderr}')

        image_list = [pth.join(tmp_output_dir, f'{node["name"]}.png') for node in self.graph.node_params_dict.values()]

        assert len(image_list) == self.graph.n_nodes

        image_names_list = []
        dir_name = pth.basename(output_dir)

        for i, image_filename in enumerate(image_list):
            if image_names is None:
                image_name = pth.join(output_dir, '{:08d}.png'.format(i_batch*self.graph.n_nodes + i))
            else:
                image_name = image_names[i]
            shutil.move(image_filename, image_name)

            # convert to 8bit
            self.convert(image_name)

            image_name = f'{dir_name}/{pth.basename(image_name)}'
            image_names_list.append(image_name)

        os.rmdir(tmp_output_dir)

        return image_names_list

    @staticmethod
    def convert(image_file):
        im = read_image(image_file)
        write_image(image_file, im)


class ArcPavement(SBSGenerators):
    @staticmethod
    def get_params():
        params = OrderedDict([('pattern_amount', RandomSampler((4,), (32,), (12,), True)),
                              ('arcs_amount', RandomSampler((4,), (20,), (14,), True)),
                              ('pattern_scale', RandomSampler((0.9,), (1.0,), (1.0,), False)),
                              ('pattern_width', RandomSampler((0.7,), (0.9,), (0.8,), False)),
                              ('pattern_height', RandomSampler((0.8,), (1.0,), (0.9,), False)),
                              ('pattern_width_random', RandomSampler((0.0,), (0.2,), (0.0,), False)),
                              ('pattern_height_random', RandomSampler((0.0,), (0.2,), (0.0,), False)),
                              ('global_pattern_width_random', RandomSampler((0.0,), (0.2,), (0.0,), False)),
                              ('pattern_height_decrease', RandomSampler((0.0,), (0.5,), (0.25,), False)),
                              ('color_random', RandomSampler((0.0,), (1.0,), (0.0,), False)),
                              ])
        return params


generator_lookup_table = {'arc_pavement': ArcPavement,
                          }
