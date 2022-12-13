import os
import os.path as pth
import shutil
import argparse
import platform
from sbs_generators import generator_lookup_table

sat_dir = 'C:/Program Files/Allegorithmic/Substance Automation Toolkit'


def synthesize_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--generator_name', type=str, required=True)
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--image_res', type=int, default=256)
    parser.add_argument('--verify', default=False, action='store_true')
    args = parser.parse_args()

    # graphviz may have a potential problem of resource leak
    if platform.system() == 'Windows':
        import win32file
        win32file._setmaxstdio(4096)

    data_path = args.data_path
    generator_name = args.generator_name
    graph_name = 'generator'
    n_samples = args.n_samples
    vis_every = 50

    output_dir = pth.join(data_path, generator_name, generator_name)
    os.makedirs(output_dir, exist_ok=True)
    graph_filename = pth.join(data_path, f'{generator_name}.sbs')
    sampler = generator_lookup_table[generator_name](graph_filename, graph_name, sat_dir, args.image_res)
    sampler.sample(output_dir, n_samples=n_samples, vis_every=vis_every)
    shutil.copyfile(pth.join(output_dir, 'dataset.json'), pth.join(data_path, generator_name, f'dataset.json'))

    # verify
    if args.verify:
        verify_output_dir = pth.join(data_path, f'{generator_name}_verify')
        os.makedirs(verify_output_dir, exist_ok=True)
        sampler.sample_with_json(verify_output_dir, pth.join(output_dir, 'dataset.json'))


if __name__ == '__main__':
    synthesize_data()