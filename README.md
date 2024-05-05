# PETReLM

This is the source code of PETReLM for dataset pre-processing and fine-tuning for classifying encrypted traffic.

In the pre-training phase we improve the code of [UER-py](https://github.com/dbiir/UER-py) for pre-training model.Many thanks to the authors.

We transfer the parameters of the trained model to the `pt_model`.
<br/>

## Using PETReLM

### Content

```
dataprepocess.py
├─checkpoint
├─ft_datasets       # store fine-tuning datasets
├─ft_model          # store fine-tuning model parapmeters
├─pt_model          # store pretrained model parameters
│
├─dataprepocess.py
├─finetuning.py       
└─evaluate.py      
```

1. Run `dataprepocess.py` to generate the encrypted traffic fine-tuning datasets. The datasets will be stored in `ft_datasets`. Note you'll need to change the file paths and some configures at the top of the file.

2. Run `finetuning.py` to load the pretrained model in `ft_model` with the fine-tuning datasets in `ft_datasets` for fine-tuning. The parameters of the trained model will be saved in `ft_model`.

3. Run `evaluate.py` to load the fine-tuned model for further viewing of the fine-tuned training results.

