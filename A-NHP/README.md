# A-NHP
Public Repo for Attentive-NeuralHawkesProcess (A-NHP). 

Author: Chenghao Yang (yangalan1996@gmail.com)

## Instructions
Here are the instructions to use the code base.

### Dependencies and Installation
This code is written in Python 3, and I recommend you to install:
* [Anaconda](https://www.continuum.io/) that provides almost all the Python-related dependencies;

Run the command line below to install the package (add `-e` option if you need an editable installation):
```
pip install .
```

### Dataset Preparation
Download datasets from [here](https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U).

Organize your domain datasets as follow:
```
domains/YOUR_DOMAIN/YOUR_PROGRAMS_AND_DATA
```

### Train Models
To train the model specified by your Datalog probram, try the command line below for detailed guide:
```
python train.py --help
```

The training log and model parameters are stored in this directory: 
```
domains/YOUR_DOMAIN/YOUR_PROGRAMS_AND_DATA/ContKVLogs
```

Example and default parameters for training:
```
python train.py -d YOUR_DOMAIN -ps ../../ -bs BATCH_SIZE -me 50 -lr 1e-4 -d_model 32 -teDim 10 -sd 1111 -layer 1
```

### Test Models
To test the trained model, use the command line below for detailed guide: 
```
python test.py --help
```

Example command line for testing:

```
python test.py -d YOUR_DOMAIN -fn FOLDER_NAME -s test -sd 12345 -pred
```

To evaluate the model predictions, use the command line below for detailed guide: 
```
python eval.py --help
```

Example command line for testing:

```
python eval.py -d YOUR_DOMAIN -fn FOLDER_NAME -s test
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
1. The transformer component implementation used in this repo is based on widely-recognized [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html
). 
1. The code structure is inspired by Prof. Hongyuan Mei's [Neural Datalog Through Time](https://github.com/HMEIatJHU/neural-datalog-through-time.git)

