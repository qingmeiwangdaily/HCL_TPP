## Dependencies

* [PyTorch] - Version: 1.10.0
* [Python] - Version: 3.9.0


## Training & Evaluation(take dataset'Hawkes' for example)


### MLE+Reg.

```
python test_learning.py -batch_size 4 -epoch 50 -model 'MLE' -save_label 'MLE + Reg' -data_folder 'tpp-data/data_hawkes' -w_mle 1 -w_dis 1 -w_cl1 0 -w_cl2 0 -seed 12

```


### MLE+DA

```
python test_learning.py -batch_size 4 -superpose -epoch 50 -model 'MLE' -save_label 'MLE + DA' -data_folder 'tpp-data/data_hawkes' -w_mle 1 -w_dis 1 -w_cl1 0 -w_cl2 0 -seed 12

```


### Dis

```
python test_learning.py  -batch_size 4  -model 'MLE' -save_label 'Dis' -data_folder 'tpp-data/data_hawkes' -epoch 50 -w_mle 0 -w_dis 1 -w_cl1 0 -w_cl2 0 -seed 12

```


### HCL+MLE

```
python test_learning.py  -batch_size 4  -num_neg 20 -ratio_remove 0.4 -model 'HCL' -save_label 'HCL+MLE ' -data_folder 'tpp-data/data_hawkes' -epoch 50 -w_mle 1 -w_dis 1 -w_cl1 1 -w_cl2 1 -seed 12

```


### HCLeve+MLE

```
python test_learning.py  -batch_size 4  -num_neg 20 -ratio_remove 0.4 -model 'HCL' -save_label 'HCL+MLE ' -data_folder 'tpp-data/data_hawkes' -epoch 50 -w_mle 1 -w_dis 1 -w_cl1 1 -w_cl2 0 -seed 12
```


### HCLseq+MLE

```
python test_learning.py  -batch_size 4  -num_neg 20 -ratio_remove 0.4 -model 'HCL' -save_label 'HCL+MLE ' -data_folder 'tpp-data/data_hawkes' -epoch 50 -w_mle 1 -w_dis 1 -w_cl1 0 -w_cl2 1 -seed 12

```


## parameters


```tpp-data``` is the dataset.


```Learning``` is the learning methods chosen for the training, including mle, hcl.


```TPPS```is the model chosen for the backbone of training.


```num_neg``` is the number of negative sequence for contrastive learning. The default value of Hawkes dataset is 20.


```wcl1``` corresponds to the weight of event level contrastive learning loss. The default value is 1.


```wcl2``` corresponds to the weight of event level contrastive learning loss. The default value is 1.


```ratio_remove ``` corresponds to the ration of removing events of per sequence when generate negative and positive sequence . The default value is 0.4.


