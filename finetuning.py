from peft import LoraConfig, TaskType, get_peft_model, IA3Config
import time
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, AutoModelForSequenceClassification
import os
import random
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import pandas as pd
import gc


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def compute_metrics(eval_preds):
    metric = evaluate.load('accuracy')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    seed = 1617181
    batch_size = 32
    num_train_epochs = 10
    # LoRA config
    r = 4
    lora_alpha = 32

    learning_rate = 5e-4
    # num of dataset labels
    num_labels = 20
    # path to load fine tuning dataset
    from_path = 'ft_datasets/U20_5000'
    # path to load pretraining model
    from_model = "pt_model"
    # path to save fine-tuning model
    save_path = "lora_saved_U20"

    gc.collect()
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checksave_path = "checkpoint"

    t_start = time.time()

    model = AutoModelForSequenceClassification.from_pretrained(from_model, num_labels=num_labels,
                                                               ignore_mismatched_sizes=True)

    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=r, lora_alpha=lora_alpha,
                             lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    for n, p in model.named_parameters():
        if n == 'base_model.model.bert.pooler.dense.weight' or n == 'base_model.model.bert.pooler.dense.bias':
            p.requires_grad = True
    # check model matrix
    # for n, p in model.named_parameters():
    #     print(n, p.requires_grad)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(from_model, tokenizer_class=AutoTokenizer)

    data_files = {
        "train": from_path + "/train.jsonl",
        "validation": from_path + "/valid.jsonl",
        "test": from_path + "/test.jsonl"
    }
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.shuffle(seed=seed)

    # tokenized_datasets features: ['sentence', 'labels', 'input_ids', 'token_type_ids', 'attention_mask']
    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(checksave_path, per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      num_train_epochs=num_train_epochs, label_names=["labels"],
                                      evaluation_strategy="epoch", learning_rate=learning_rate, save_steps=10000)
    print('using', device)
    model.to(device)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    print('start training')
    trainer.train()

    predictions = trainer.predict(tokenized_datasets["test"])
    predicts = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    avg_w = accuracy_score(labels, predicts)
    pr_w = precision_score(labels, predicts, average='macro')
    rc_w = recall_score(labels, predicts, average='macro')
    f1_w = f1_score(labels, predicts, average='macro')
    predictions_m = trainer.predict(tokenized_datasets["validation"])
    predicts_m = np.argmax(predictions_m.predictions, axis=-1)
    labels_m = predictions_m.label_ids
    avg_m = accuracy_score(labels_m, predicts_m)
    pr_m = precision_score(labels_m, predicts_m, average='macro')
    rc_m = recall_score(labels_m, predicts_m, average='macro')
    f1_m = f1_score(labels_m, predicts_m, average='macro')
    print('------------lorafinetune_withpooler--------------')
    print(seed, batch_size, num_train_epochs, r, lora_alpha, learning_rate,
          num_labels, from_path)
    score_dict_w = {'accuracy': avg_w, 'precision_w': pr_w, 'recall_w': rc_w, 'f1_w': f1_w}
    print("test: " + str(score_dict_w))
    score_dict_m = {'accuracy': avg_m, 'precision_m': pr_m, 'recall_m': rc_m, 'f1_m': f1_m}
    print("valid: " + str(score_dict_m))

    model.save_pretrained(save_path)
    lora_params_addicted = {n: p for n, p in model.named_parameters() if "pooler" in n}
    torch.save(lora_params_addicted, from_path + "\withpoolerlora_params.pth")

    print("running time:", time.time() - t_start)


