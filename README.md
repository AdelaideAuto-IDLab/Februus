# README 

This is for releasing the source code of the paper "Februus: Input Purification Defense Against Trojan Attacks on Deep Neural Network Systems" 

The source code is written mostly on *Python 3* and *Pytorch*, so please help to download and install Python3 and Pytorch beforehand.


There are some large files in the repo, so it is recommended to install git LFS: 
```
sudo apt-get install git-lfs
git lfs install
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

For the convenience of the reviewers, we put the GTSRB and VGGFace2-val set that we used in [datasets](./datasets) folder. 
There are some large files but with the limited storage of Github LFS, we put large files on a separate link, please help to follow the below instructions: 


- Download the *largefile* file: 

```
https://universityofadelaide.box.com/s/u7di8pcirjc1flvnzwcu16l2ddujorb8
```

- Put the *largefile.tar.gz* to the root of Februus repo folder (i.e. Februus/largefile.tar.gz)

- Untar the largefile
```
untar -xzvf largefile.tar.gz
```
- Untar the dataset within the datasets folder (please don't change the name of the folder, or you'll need to adjust the location of the dataset later in the source code).
```
cd datasets
tar -xzvf gtsrb-dataset.tar.gz
tar -xzvf vggface2-val.tar.gz
```

# Run the Februus

There are 3 sub-repo, each for a different visual classification task:
- [face](./face): Face Recognition Task (VGGFace2)
- [scene](./scene): Scene Classification Task (CIFAR-10)
- [traffic_sign](./traffic_sign): Traffic Sign Recognition Task (GTSRB)

There are two ways to run the method, the first on is to run the *Februus.py* file within each sub-repo: 

```python
# For example, to run Februus on Scene Classification task
cd scene
python3 Februus.py
```
This is to run the Februus on the whole test set for that task.

The second ways is to run step-by-step with the Jupyter Notebook file *Februus.ipynb* inside of each repo. 

