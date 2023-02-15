# DualQSELD-TCN
Official PyTorch repository for Dual Quaternion Ambisonics Array for Six-Degree-of-Freedom Acoustic Representation, published in 
[Elsevier Pattern Recognition Letters](https://www.sciencedirect.com/science/article/pii/S0167865522003749), ([ArXiv preprint](https://arxiv.org/pdf/2204.01851.pdf)).

Eleonora Grassucci, Gioia Mancini, Christian Brignone, Aurelio Uncini, and Danilo Comminiello

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dual-quaternion-ambisonics-array-for-six/sound-event-localization-and-detection-on)](https://paperswithcode.com/sota/sound-event-localization-and-detection-on?p=dual-quaternion-ambisonics-array-for-six)


**This repository is under construction, please open an issue if you find a bug!**

## Usage

* Create a new conda environment and then install requirements with `pip install -r requirements.txt`.
* Download and preprocess the L3DAS21 dataset with
```python
python download_dataset.py --task Task2 --set_type train --output_path DATASETS/Task2
python download_dataset.py --task Task2 --set_type dev --output_path DATASETS/Task2
python preprocessing.py --task 2 --input_path DATASETS/Task2 --num_mics 2 --frame_len 100
```
For detailed instructions and more information on the dataset, please refer to the official GitHub repository [L3DAS21](https://github.com/l3das/L3DAS21).

* Choose the configuration in `configs`.
* Run the experiment: `python train_model.py --TextArgs=chosen_config.txt`.


## Cite
Please cite our work if you found it useful.

```
@article{GRASSUCCI202324,
title = {Dual quaternion ambisonics array for six-degree-of-freedom acoustic representation},
journal = {Pattern Recognition Letters},
volume = {166},
pages = {24-30},
year = {2023},
issn = {0167-8655},
doi = {https://doi.org/10.1016/j.patrec.2022.12.006},
url = {https://www.sciencedirect.com/science/article/pii/S0167865522003749},
author = {Eleonora Grassucci and Gioia Mancini and Christian Brignone and Aurelio Uncini and Danilo Comminiello}
}
```
