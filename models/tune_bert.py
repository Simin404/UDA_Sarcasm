# freeze all the parameters
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from utils import compute_accuracy_v3
from T_BERT import T_BERT
import os

def train_bert(epochs, lr, source_dl, val_dl, output_Bert, output_file, device):
    best_valid_loss = float('inf')
    train_losses=[]
    valid_losses=[]
    loss_function  = nn.NLLLoss() 
    mymodel = T_BERT().to(device)
    optimizer = optimizer = optim.Adam(mymodel.parameters(), lr = lr)

    for epoch in range(epochs):
        
        print('Epoch {:} / {:}'.format(epoch + 1, epochs))
        train_loss, _ = train(mymodel, source_dl, loss_function, optimizer, device)
        val_loss, avg_acc, avg_f1, avg_p, avg_recall = evaluate(mymodel, val_dl, loss_function, device)
        print('Training Loss: {:.3f}'.format(train_loss))
        print('Validation Loss:{:.3f}, Acc:{:.3f}, F1:{:.3f}, Precision:{:.3f}, Recall:{:.3f}'.format(val_loss, avg_acc, avg_f1, avg_p, avg_recall))

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            model_path = os.path.join(output_Bert, "epoch_" + str(epoch+1) + "_" + output_file)
            torch.save(mymodel.state_dict(), model_path)
            print('Saved model:', "epoch_" + str(epoch+1) + "_" + output_file)

        train_losses.append(train_loss)
        valid_losses.append(val_loss)
    return model_path


def train(model, train_dl, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0
    total_preds=[]
    
    for step, batch in enumerate(train_dl):
        if step % 1000 == 0 and not step == 0:
          print('Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dl)))

        batch = [r.to(device) for r in batch]
        sent_id, mask, token, labels = batch
        model.zero_grad()        
        preds, _ = model(sent_id, mask, token)
        loss = loss_function(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()
        preds=preds.detach().cpu().numpy()
        total_preds.append(preds)
    avg_loss = total_loss / len(train_dl)
    total_preds  = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


def evaluate(model, data_dl, loss_function, device):
    # print("\nEvaluating...")
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    total_p = 0.0
    total_recall = 0.0
    total_preds = []
    for step,batch in enumerate(data_dl):
        if step % 200 == 0 and not step == 0:
            # elapsed = format_time(time.time() - t0)
            print('Batch {:>5,}  of  {:>5,}: '.format(step, len(data_dl)))
        batch = [t.to(device) for t in batch]
        sent_id, mask, token, labels = batch
        with torch.no_grad():
            preds, _ = model(sent_id, mask, token)
            loss = loss_function(preds,labels)
            acc, f1,precision,recall = compute_accuracy_v3(preds, labels, device)
            total_loss = total_loss + loss.item()
            total_accuracy += acc
            total_f1 += f1
            total_p += precision
            total_recall += recall
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
    avg_loss = total_loss / len(data_dl) 
    avg_acc = total_accuracy / len(data_dl) 
    avg_f1 = total_f1 / len(data_dl) 
    avg_p = total_p / len(data_dl) 
    avg_recall = total_recall / len(data_dl) 
    
    return avg_loss, avg_acc, avg_f1, avg_p, avg_recall
     
        