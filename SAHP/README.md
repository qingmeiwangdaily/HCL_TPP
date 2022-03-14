# SAHP

This is the repository for the [Self-Attentive Hawkes Processes](https://proceedings.icml.cc/static/paper_files/icml/2020/1421-Paper.pdf) paper where self-attention is used to adapt the intensity function of Hawkes process.

## Dataset
The realword datasets are available on this [Google drive] (https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U) while the synthetic dataset is at this [link] (https://drive.google.com/file/d/1lRUIJx5UIPMx4TMwKy6GiAiP-k2vwvDc/view?usp=sharing). To run the model, you should download them to
the parent directory of the source code, with the folder name `data`.

 To make the data format consistent, it is necessary to run the script [convert_realdata_syntheform.py](utils/convert_realdata_syntheform.py) first. 


## Package
The Python version should be at least 3.5 and the torch version can be 0.4.1

## Scripts
`models` defines the self-attentive Hawkes model, multi-head attention and the related.

`main_func.py` is the main function to run the experiments, hyper-parameters are provided here.

`utils` contains utility functions

To run the model: python main_func.py

## Citation
```
@article{zhang2019self,
  title={Self-attentive Hawkes processes},
  author={Zhang, Qiang and Lipani, Aldo and Kirnap, Omer and Yilmaz, Emine},
  journal={arXiv preprint arXiv:1907.07561},
  year={2019}
}
```
