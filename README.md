# Learning to Classify Images without Labels


This repo contains the Pytorch implementation of our paper:
> [**FUSC: Fetal Ultrasound Semantic Clustering of Second Trimester Scans Using Deep Self-supervised Learning**](https://arxiv.org/abs/2310.12600)
>
> Hussain Alasmawi, Leanne Bricker, and Mohammad Yaqub.

## Installation
The code runs with recent Pytorch versions, e.g. 1.4. 
Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/), the most important packages can be installed as:
```shell
conda install pytorch torchvision cudatoolkit -c pytorch
conda install matplotlib scipy scikit-learn   # For evaluation and confusion matrix visualization
conda install faiss-gpu                       # For efficient nearest neighbors search 
conda install pyyaml easydict                 # For using config files
conda install termcolor                       # For colored print statements
```
We refer to the `requirements.txt` file for an overview of the packages in the environment we used to produce our results.

## Training

### Setup
The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the datasets in `utils/mypath.py`, e.g. `/path/to/cifar10`.
- Specify the output directory in `configs/env.yml`. All results will be stored under this directory. 


### Train model
The configuration files can be found in the `configs/` directory. The training procedure consists of the following steps:
- __STEP 1__: Solve the pretext task i.e. `simclr.py`
- __STEP 2__: Perform the clustering step i.e. `fusc.py`
- __STEP 3__: Perform the self-labeling step i.e. `selflabel.py`

For example, run the following commands sequentially to perform our method on US:
```shell
python simclr.py --config_env configs/your_env.yml --config_exp configs/pretext/simclr_USyml
python scan.py --config_env configs/your_env.yml --config_exp configs/fusc/fusc_US.yml
python selflabel.py --config_env configs/your_env.yml --config_exp configs/selflabel/selflabel_US.yml
```







## Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@misc{alasmawi2023fusc,
      title={FUSC: Fetal Ultrasound Semantic Clustering of Second Trimester Scans Using Deep Self-supervised Learning}, 
      author={Hussain Alasmawi and Leanne Bricker and Mohammad Yaqub},
      year={2023},
      eprint={2310.12600},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
For any enquiries, please contact the main authors.

## License


This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).

## Acknowledgements
This code based on [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification/tree/master)
