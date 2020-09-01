# README 

This is for releasing the source code of the paper "Februus: Input Purification Defense Against Trojan Attacks on Deep Neural Network Systems" 

The source code is written mostly on *Python 3* and *Pytorch*, so please help to download and install Python3 and Pytorch beforehand.


# Requirements

To install the requirements for this repo, run the following command: 
```
git clone https://github.com/AdelaideAuto-IDLab/Februus.git
cd Februus
pip3 install -r requirements.txt
```


# Dataset

For the convenience of the reviewers, we put the GTSRB and VGGFace2-val set that we used in [datasets](./datasets) folder. 
Please *untar* files within the datasets folder (please don't change the name of the folder, or you'll need to adjust the location of the dataset later in the source code).

# Run the Februus

There are 3 sub-repo, each for a different visual classification task:
- [face](./face): Face Recognition Task (VGGFace2)
- [scene](./scene): Scene Classification Task (CIFAR-10)
- [traffic_sign](./traffic_sign): Traffic Sign Recognition Task (GTSRB)

There are two ways to run the method, the first on is to run: 

```python
python3 Februus.py
```

within each of the sub-repo, which is to run the Februus method on the whole test set.

The second ways is to run step-by-step with the Jupyter Notebook file *Februus.ipynb* inside of each repo. 

