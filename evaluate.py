import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import evaluate
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, AutoModelForSequenceClassification
import os
import random
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import pandas as pd
import json


# # C33
# ID_TO_APP = {0: 'Backdoor_Malware', 1: 'BrowserHijacking', 2: 'CommandInjection', 3: 'DDoS-ACK_Fragmentation', 4: 'DDoS-HTTP_Flood', 5: 'DDoS-ICMP_Flood', 6: 'DDoS-ICMP_Fragmentation', 7: 'DDoS-PSHACK_Flood', 8: 'DDoS-RSTFINFlood', 9: 'DDoS-SlowLoris', 10: 'DDoS-SynonymousIP_Flood', 11: 'DDoS-SYN_Flood', 12: 'DDoS-TCP_Flood', 13: 'DDoS-UDP_Flood', 14: 'DDoS-UDP_Fragmentation', 15: 'DictionaryBruteForce', 16: 'DNS_Spoofing', 17: 'DoS-HTTP_Flood', 18: 'DoS-SYN_Flood', 19: 'DoS-TCP_Flood', 20: 'DoS-UDP_Flood', 21: 'Mirai-greeth_flood', 22: 'Mirai-greip_flood', 23: 'Mirai-udpplain', 24: 'MITM-ArpSpoofing', 25: 'Recon-HostDiscovery', 26: 'Recon-OSScan', 27: 'Recon-PingSweep', 28: 'Recon-PortScan', 29: 'SqlInjection', 30: 'Uploading_Attack', 31: 'VulnerabilityScan', 32: 'XSS'}
# # IA17
# ID_TO_APP = {0: 'aim', 1: 'email', 2: 'facebook', 3: 'ftp', 4: 'gmail', 5: 'hangouts', 6: 'icq', 7: 'netflix', 8: 'p2p', 9: 'scp', 10: 'sftp', 11: 'skype', 12: 'spotify', 13: 'tor', 14: 'vimeo', 15: 'voipbuster', 16: 'youtube'}
# # IS12
# ID_TO_APP = {0: 'Chat', 1: 'Email', 2: 'FileTransfer', 3: 'P2P', 4: 'Streaming', 5: 'VoIP', 6: 'VPN_Chat', 7: 'VPN_Email', 8: 'VPN_FileTransfer', 9: 'VPN_P2P', 10: 'VPN_Streaming', 11: 'VPN_VoIP'}
# # IT8
# ID_TO_APP = {0: 'tor_AUDIO', 1: 'tor_BROWSING', 2: 'tor_CHAT', 3: 'tor_FTP', 4: 'tor_MAIL', 5: 'tor_P2P', 6: 'tor_VEDIO', 7: 'tor_VOIP'}
# U20
ID_TO_APP = {0: 'Benign_BitTorrent', 1: 'Benign_Facetime', 2: 'Benign_FTP', 3: 'Benign_Gmail', 4: 'Benign_MySQL', 5: 'Benign_Outlook', 6: 'Benign_Skype', 7: 'Benign_SMB', 8: 'Benign_Weibo', 9: 'Benign_WorldOfWarcraft', 10: 'Malware_Cridex', 11: 'Malware_Geodo', 12: 'Malware_Htbot', 13: 'Malware_Miuref', 14: 'Malware_Neris', 15: 'Malware_Nsis-ay', 16: 'Malware_Shifu', 17: 'Malware_Tinba', 18: 'Malware_Virut', 19: 'Malware_Zeus'}


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)


def normalise_cm(cm):
    with np.errstate(all="ignore"):
        normalised_cm = cm / cm.sum(axis=1, keepdims=True)
        normalised_cm = np.nan_to_num(normalised_cm)
        return normalised_cm


def plot_confusion_matrix(cm, labels):
    normalised_cm = normalise_cm(cm)
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.set(font_scale=1.5)
    sns.heatmap(
        data=normalised_cm, cmap='YlGnBu',
        xticklabels=labels, yticklabels=labels,
        ax=ax, aspect="equal"
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xlabel('Predict Labels', fontsize=23)
    ax.set_ylabel('True Labels', fontsize=23)
    plt.tight_layout()
    fig.show()


if __name__ == '__main__':
    seed = 1617181
    learning_rate = 5e-4
    num_labels = 33
    batch_size = 32
    num_train_epochs = 10
    from_path = 'ft_datasets/U20'
    checksave_path = "checkpoint"
    peft_model_id = "lora_saved_U20"
    seed_everything(seed)

    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=num_labels,
                                                               ignore_mismatched_sizes=True)
    model = PeftModel.from_pretrained(model, peft_model_id)
    pooler = torch.load(peft_model_id+'/withpoolerlora_params.pth')
    for name, p in model.named_parameters():
        if name == 'base_model.model.bert.pooler.dense.weight':
            p.data = pooler['base_model.model.bert.pooler.dense.weight'].data.cpu()
        if name == 'base_model.model.bert.pooler.dense.bias':
            p.data = pooler['base_model.model.bert.pooler.dense.bias'].data.cpu()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_files = {
        "validation": from_path+"/valid.jsonl",
        "test": from_path+"/test.jsonl"
    }
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.shuffle(seed=seed)
    # tokenized_datasets features: ['sentence', 'labels', 'input_ids', 'token_type_ids', 'attention_mask']
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(checksave_path, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                      num_train_epochs=num_train_epochs, label_names=["labels"], evaluation_strategy="epoch",
                                      save_steps=5000, learning_rate=learning_rate)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["test"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

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
    score_dict_w = {'accuracy': avg_w, 'precision_m': pr_w, 'recall_m': rc_w, 'f1_m': f1_w}
    print("test: " + str(score_dict_w))
    score_dict_m = {'accuracy': avg_m, 'precision_m': pr_m, 'recall_m': rc_m, 'f1_m': f1_m}
    print("valid: " + str(score_dict_m))

    cm = confusion_matrix(labels, predicts)
    app_labels = []
    for i in sorted(list(ID_TO_APP.keys())):
        app_labels.append(ID_TO_APP[i])

    plot_confusion_matrix(cm, app_labels)





















