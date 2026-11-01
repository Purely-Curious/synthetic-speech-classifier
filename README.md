# synthetic-speech-classifier
ML-assignment repo for audio classification


# How to use:

## Loading the dataset.
* Download the FoR audio dataset from kaggle: https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset/data
* Unzip the dataset
* Move the `for-norm` dataset into this folder.
> Note: this folder has an odd quirk as it's packaged as the following:
```bash
 for-norm/for-norm/...
```
> As a result, you need to ensure that the folder is not nested, aka:
```bash
for-norm/...
```

## Running the program
* Create a python virtual environment.
* Install the requirements file.
```bash
pip install -r requirements.txt
```
* To run the baseline model:
```bash
python baseline-model.py
```
