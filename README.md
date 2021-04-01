# README 

This is for releasing the source code of the paper "Februus: Input Purification Defense Against Trojan Attacks on Deep Neural Network Systems" 

Archived Version: [Februus](https://arxiv.org/abs/1908.03369)

The project is published as part of the following paper and if you re-use our work, please cite the following paper:


```
@inproceedings{doan2020februus,
title={Februus: Input Purification Defense Against Trojan Attacks on Deep Neural Network Systems},
author={Bao Gia Doan and Ehsan Abbasnejad and Damith C. Ranasinghe},
year = {2020},
booktitle = {Proceedings of the 36th Annual Computer Security Applications Conference (ACSAC)},
location = {Austin, TX, USA},
series = {ACSAC 2020}
}
```


The source code is written mostly on *Python 3* and *Pytorch*, so please help to download and install Python3 and Pytorch beforehand.


There are some large files in the repo, so it is recommended to install git LFS: 
```
sudo apt-get install git-lfs
git lfs install
```

Install *cmake* (optionally):
```
sudo apt-get install cmake
```

# Requirements

To install the requirements for this repo, run the following command: 
```
git clone https://github.com/AdelaideAuto-IDLab/Februus.git
cd Februus
pip3 install -r requirements.txt
pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```


# Large Files and Datasets

For the convenience of the users, we put the GTSRB and VGGFace2-val set that we used in [datasets](./datasets) folder. 
There are some large files but with the limited storage of Github LFS, we put large files on a separate link, please help to follow the below instructions: 


- Download the *largefile* file (https://universityofadelaide.box.com/s/u7di8pcirjc1flvnzwcu16l2ddujorb8)

- Download the *BTSR* file for Belgium Traffic Sign Recognition (https://universityofadelaide.box.com/s/wo567ru7tuxrcfjz2ypbkarqat87eabw) 
  
- Put the *largefile.tar.gz*  and *BTSR.tar.gz* to the root of Februus repo folder (i.e. Februus/largefile.tar.gz)

- Untar the files
```
tar -xzvf largefile.tar.gz
tar -xzvf BTSR.tar.gz
```
- Untar the dataset within the datasets folder (please don't change the name of the folder, or you'll need to adjust the location of the dataset later in the source code).
```
cd datasets
tar -xzvf gtsrb-dataset.tar.gz
tar -xzvf vggface2-val.tar.gz
```

# Run the Februus

There are 4 sub-repo, each for a different visual classification task:
- [face](./face): Face Recognition Task (VGGFace2)
- [scene](./scene): Scene Classification Task (CIFAR-10)
- [traffic_sign](./traffic_sign): Traffic Sign Recognition Task (GTSRB)
- [BTSR](./BTSR): Belgium Traffic Sign Recognition (BTSR)

There are two ways to run the method:

- The first way is to run step-by-step with the Jupyter Notebook file *Februus.ipynb* inside of each repo. 


- The second way is to run the *Februus.py* file within each sub-repo. This is to run the Februus on the whole test set for that task.: 

```python
# For example, to run Februus on Scene Classification task
cd scene
python3 Februus.py
```
  
## TODO 
- [ ] add the training code


