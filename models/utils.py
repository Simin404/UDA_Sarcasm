import torch
import torch.nn as nn
import time
import datetime
import matplotlib.pyplot as plt
from torcheval.metrics import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall,BinaryConfusionMatrix
import json



def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def compute_metrics_ce(logits, labels, device):
    predicted_label = logits.max(dim = 1)[1]
    acc_matrix = MulticlassAccuracy(average="macro", num_classes=2).to(device)
    f1_matrix = MulticlassF1Score(average="macro", num_classes=2).to(device)
    precision_matrix = MulticlassPrecision(average="macro", num_classes=2).to(device)
    recall_matrix = MulticlassRecall(average="macro", num_classes=2).to(device)
    acc = acc_matrix.update(predicted_label, labels).compute()
    f1 = f1_matrix.update(predicted_label, labels).compute()
    precision = precision_matrix.update(predicted_label, labels).compute()
    recall = recall_matrix.update(predicted_label, labels).compute()
    # cm = BinaryConfusionMatrix().update(predicted_label, labels).compute()
    return acc.item(), f1.item(), precision.item(), recall.item()


def compute_accuracy(logits, labels):
    
    predicted_labels_dict = {
      0: 0,
      1: 0,
    }
    
    predicted_label = logits.max(dim = 1)[1]
    
    for pred in predicted_label:
        predicted_labels_dict[pred.item()] += 1
    acc = (predicted_label == labels).float().mean()
    
    return acc, predicted_labels_dict


def compute_accuracy_v3(logits, labels, device):
    predicted_label = logits.max(dim = 1)[1]
    acc_matrix = BinaryAccuracy().to(device)
    f1_matrix = BinaryF1Score().to(device)
    precision_matrix = BinaryPrecision().to(device)
    recall_matrix = BinaryRecall().to(device)
    acc = acc_matrix.update(predicted_label, labels).compute()
    f1 = f1_matrix.update(predicted_label, labels).compute()
    precision = precision_matrix.update(predicted_label, labels).compute()
    recall = recall_matrix.update(predicted_label, labels).compute()
    # cm = BinaryConfusionMatrix().update(predicted_label, labels).compute()
    return acc.item(), f1.item(), precision.item(),recall.item()

def compute_accuracy_v2(logits, labels):
    acc = torch.sum((logits == labels).float())/len(labels)
    return acc.item()

def compute_metrics_bce(pred, labels, device):
    acc_matrix =  MulticlassAccuracy(average="macro", num_classes=2).to(device)
    f1_matrix = MulticlassF1Score(average="macro", num_classes=2).to(device)
    precision_matrix = MulticlassPrecision(average="macro", num_classes=2).to(device)
    recall_matrix = MulticlassRecall(average="macro", num_classes=2).to(device)
    acc = acc_matrix.update(pred, labels).compute()
    f1 = f1_matrix.update(pred, labels).compute()
    precision = precision_matrix.update(pred, labels).compute()
    recall = recall_matrix.update(pred, labels).compute()
    return acc.item(), f1.item(), precision.item(),recall.item()


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def plotting(caption, dic_to_print):
    for i in range(len(dic_to_print)):
        plt.plot(list(dic_to_print.values())[i],label=list(dic_to_print.keys())[i])
    plt.title(caption)
    plt.legend()
    plt.show()

def plot_two(caption, dic_one, dic_two):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(caption)
    for i in range(len(dic_one)):
        ax1.plot(list(dic_one.values())[i],label=list(dic_one.keys())[i])
        ax1.legend()
    for j in range(len(dic_two)):
        ax2.plot(list(dic_two.values())[j],label=list(dic_two.keys())[j])
        ax2.legend()
    fig.show()

def get_cfg(model_name, IN_COLAB = False):
  if IN_COLAB:
     config = json.load(open('/content/drive/MyDrive/UDA_Sarcasm/cfgs/' + model_name + '.json', 'r'))
  else:
     config = json.load(open('../cfgs/'+ model_name +'.json', 'r'))
  return config