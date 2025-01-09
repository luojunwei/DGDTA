# DGDTA

Dynamic graph attention network for predicting drug-target binding affinity. It is
built with **Pytorch* and **Python 3**.


## Installation

### Requirements
  * python 3.9, numpy, scipy, pandas, pytorch, pyg

#### 1. Create a virtual environment

```bash
# create
conda create -n DGDTA python=3.9
# activate
conda activate DGDTA
# deactivate
conda deactivate
```

#### 2. clone DGDTA
- After creating and activating the DGDTA virtual environment, download DGDTA from github:
```bash
git clone https://github.com/luojunwei/DGDTA.git
cd DGDTA
```
#### 3. Install

```bash
conda activate DGDTA
conda install numpy, scipy, pandas, Pytorch, pyg

```



## Tested data
The example data can be downloaded from 
#### Davis and KIBA
https://github.com/thinng/GraphDTA/tree/master/data



## Usage

### Train Model

#### 1. Create Dataset

```bash
python data_creation.py

```
First, divide the data into training and test sets and create data files in pytorch format.
#### 2. Train model

Run the following script to train the model.
```bash
python main.py

```
The default values of the parameter parser are the DGDTA-CL version and the KIBA dataset.


#### 3. Validate the training prediction model

Run the following script to test the model.
```bash
python validation.py

```
This returns the best MSE model for the validation dataset during the training process.


