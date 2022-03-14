# A pytorch implementation of ERPP and RMTPP
This is a course project for SJTU, CS488 Temporal Point Process Modeling.  

ERPP reference paper is [Modeling the intensity function of point process via recurrent neural networks. (AAAI 2017)](https://arxiv.org/pdf/1705.08982.pdf)

RMTPP reference paper is [Recurrent marked temporal point processes: Embedding event history to vectore.(KDD 2016)](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf)

## Dataset
In maintenance support services, when a device fails, the equipment owner raises a maintenance service ticket and technician will be assigned to repair the failure. The studied dataset is comprised of the event logs involving error reporting and failure tickets, which is originally collected from 1,554 ATMs. The event log of error records includes device identity, timestamp, message content, priority, code, and action.

Dataset has been splited into train and test set.

```
data/train_day.csv
data/test_day.csv
```

Each csv file contains 3 columns, the first column indicates current ATM machine id. The second column refers to the time sequence where an event happens. The last column indicates the type of events.
```
id,time,event
g1548,16344.394270833332,0
g1548,16367.035381944444,4
g1548,16367.036377314815,4
g1548,16367.037650462962,4
g1548,16442.100289351853,2
g1548,16490.032743055555,1
g1548,16490.032743055555,1
g1548,16514.03287037037,4
g1548,16514.033252314814,3
g1548,16514.041932870372,3
```

The task is to predict the next event time and to classify the event's category. The metric for time prediction is **mean relative error(MAE)**. And for multi-class classification, the metric is traditional **Precision, Recall and F1-score**.


## Requirements

```
pytorch = 0.4.1
numpy = 1.14.2
tqdm = 4.28.1
pandas = 0.23.4
```

## Run ERPP (Event Recurrent Point Process)
With default setting:
```
python main.py --model=erpp
```
You may get the following result:
MAE=4.9, Precision=0.77, Recall=0.90, F1=0.83

## Run RMTPP (Recurrent Marked Temporal Point Process)
With default setting:
```
python main.py --model=rmtpp
```
You may get the following result:
MAE=4.8, Precision=0.76, Recall=0.89, F1=0.825

## Other Parameters

```
python main.py --name=EXPERIMENT_NAME \
               --model= \   # "erpp" or "rmtpp"
               --seq_len=10 \
               --emb_dim=10 \
               --hid_dim=32 \
               --mlp_dim=16 \
               --alpha=0.05 \  # weight on time loss
               --dropout=0.1 \
               --batch_size= 1024 \
               --lr=1e-3 \
               --epochs=30 \
               --importance_weight \  # if use importance loss weight
               --verbose_step 
```
