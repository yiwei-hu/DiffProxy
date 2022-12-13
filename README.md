# Node Graph Optimization Using Differentiable Proxies
An improved node graph optimization method for inverse procedural material modeling.
![teaser](data/teaser.png)

Yiwei Hu, Paul Guerrero, Miloš Hašan, Holly Rushmeier, Valentin Deschaintre

In SIGGRAPH '22 Conference Processing, Vancouver, BC, Canada, 2022. [[Project page]](https://yiweihu.netlify.app/project/hu2022diff/)

## Requirements
See requirements for StyleGAN3: https://github.com/NVlabs/stylegan3.

## List of Change for StyleGAN
We modify the training setup by making the inputs as normalized parameters only, and implement our customized loss functions in `loss.py`.
Training code of StyleGAN is included in `stylegan` folder. (Updated on 12/10/2022)

## Generate Training Data (ArcPavement Generator as Example)
All commands are listed in `cmd.py` for convenience. 

**Step 1**: Please specify a correct path `sat_dir` to substance automatic toolkit in `synthesis.py` and then synthesize training data:
```bash 
python synthesis.py --data_path=./data/sbs --generator_name=arc_pavement -n_samples=307200
```
307200 images will be synthesized into a folder `./data/sbs/arc_pavement`

**Step 2**: Build the training dataset:
```bash 
python stylegan/dataset_tool.py --source=./data/sbs/arc_pavement --dest=./data/train/arc_pavement_300k.zip
```
A dataset for training StyleGAN will be generated at the location `./data/train/arc_pavement_300k.zip`.

## Train a Differentiable Proxy
If you want to train your own data, please make sure you have a json file that stores parameters for each image, 
and use `stylegan/dataset_tool.py` (Step 2) to build a dataset for training. We include an example of such json file in `./data/example`. 

Please check `cmd.py` for the training command. Generally, there are two versions of networks: with GAN loss and without GAN loss, which is specified by an option `--no_gan`. The training log and intermediate results can be found in  `training-runs` subfolder. 
In order to run the code, please put the pretrained VGG19 model into `./pretrained` folder. The VGG19 model can be downloaded from here:  https://drive.google.com/file/d/13TTR61wQ3OVJUKk6P39HegoWBQ-_MUQq/view?usp=sharing.

Loss weights can be adjusted in the `loss.py` . 

## Validate a Differentiable Proxy
`generate.py` is a simple script describing how to use the trained differentiable proxy to generate images and compare to the real generator.

##  Train a New Substance Generator
To train a new generator, you need to implement an appropriate parameter sampling method for that generator. The code can be found in `./sbs/sbs_generator.py`. Define you own generator class by overwriting method `get_params()`. Two basic samplers are defined to sample parameters: `RandomSampler` (uniform sampling based on min and max values) and `GaussianRandomSampler` (Gaussian-like sampling based on mean and std, and the sampled values are clipped between min and max). What you need to do is to create your own class and then register that class in the variable `generator_lookup_table`. 

The samplers can be arbitrary for each parameter. `GaussianRandomSampler` is recommended if you have the reasonable statistics for that parameter. But note that during training phase, there is an option `--norm_type` specifies the normalization method applied on the parameters during the training phase. Currently, there are two options `norm` and `std`. The `norm` method rescales parameters between `0` and `1` based on min and max values, while `std` works by normalizing the parameters to `mean=0` and `std=1` so `std` option only works when all the samplers are `GaussianRandomSampler`.

After defining the sampling method, you should prepare a .sbs file similar to `./data/sbs/arc_pavement.sbs`. You can find a template file in `./data/sbs` named as `basic.sbs`. All you need to do is to open that file and replace the placeholder generator by your own generator and rename the sbs file properly. However, only 1 generator can be very slow when generating data. We recommend you copy-paste that "two-node" node graph 512 times or 1024 times. This will make the program synthesize 512 mask maps or 1024 mask maps for each step.

You should now be able to generate the data and train the networks. To test if the data is correctly generated, we recommend first generating a couple of mask maps (e.g. 128 or 256) to see if their general appearances looks reasonable. To see if your sampled parameters are correctly saved as a json file, you can specify `--verify=True` when you call `synthesis.py`. The program will re-generate all the mask maps to another folder based on the parameters your just sampled. This can help you examine whether the parameter you sampled and the generated mask map is a correct one-on-one mapping. Note `verify=True` should be only used for verification and prevent using it when you generating the whole dataset.

## Citation
```
@inproceedings{hu2022diff,
author = {Hu, Yiwei and Guerrero, Paul and Hasan, Milos and Rushmeier, Holly and Deschaintre, Valentin},
title = {Node Graph Optimization Using Differentiable Proxies},
year = {2022},
isbn = {9781450393379},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3528233.3530733},
doi = {10.1145/3528233.3530733},
booktitle = {ACM SIGGRAPH 2022 Conference Proceedings},
articleno = {5},
numpages = {9},
keywords = {inverse material modeling, procedural materials},
location = {Vancouver, BC, Canada},
series = {SIGGRAPH '22}
}
```

## Contact
If you have any question, feel free to contact yiwei.hu@yale.edu
