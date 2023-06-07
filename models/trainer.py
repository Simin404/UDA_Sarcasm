
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import time
from utils import format_time, compute_metrics_ce, compute_metrics_bce
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, ConcatDataset

class BERT_Trainer(object):
    def __init__(self, cfg, model, source, source_val, device, task):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.task = task
        self.source_dl = DataLoader(dataset = source, batch_size = cfg['batch_size'], shuffle = True, drop_last=True, num_workers = 2)
        self.source_val_dl = DataLoader(dataset = source_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True, num_workers = 2)

    def train(self):

        self.model = self.model.to(self.device)
        n_epochs = self.cfg['n_epochs']
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['lr'])
        batch_num = len(self.source_dl)

        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl)
            self.model.train()
            t0 = time.time()
            train_loss = 0
            best_loss = float('inf')
            for batch_index in range(batch_num):           
                
                src_id, src_mask, src_token, src_label = [ t.to(self.device) for t in next(source_iter)]
                optimizer.zero_grad()
                outputs = self.model(src_id, attention_mask = src_mask, token_type_ids = src_token, labels=src_label)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                train_loss += loss
                if((batch_index + 1)%self.cfg['print_after_steps'] == 0 ):
                    trainning_time = format_time(time.time() - t0)
                    print("Epoch: {}/{}, Step: {}/{}, time:{}, loss:{:.4f}"
                        .format(epoch+1, n_epochs, batch_index+1, batch_num, trainning_time, loss))
                
            avg_loss = train_loss / batch_num
            print("Epoch: {}/{}, Avg Loss:{:.4f}".format(epoch+1, n_epochs, avg_loss))
            avg_loss = self.eval_train() 
            if avg_loss < best_loss:
                self.save()
                best_loss = avg_loss
            print('--------------------------------------------------------------')
            t0 = time.time()

    def eval_train(self):
        """ evaluation function """
        self.model.eval()
        total_loss, total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0, 0
        data_len = len(self.source_val_dl)
        for val_id, val_mask, val_token , val_y_batch in self.source_val_dl:
            with torch.no_grad():
                val_y_batch = val_y_batch.to(self.device)
                val_id = val_id.to(self.device)
                val_mask = val_mask.to(self.device)
                val_token = val_token.to(self.device)
                outputs = self.model(val_id, attention_mask = val_mask, token_type_ids = val_token, labels=val_y_batch)
                loss = outputs["loss"]
                total_loss += loss
                logits = outputs["logits"]
                acc, f1, precision,recall = compute_metrics_ce(logits, val_y_batch, device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall
        print("Source: Loss:{:.4f}, Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                    .format(total_loss/data_len, total_acc/data_len, total_f1/data_len, total_p/data_len, total_recall/data_len))
        return  total_loss/data_len

    def eval_model(self, eval_set):
        """ evaluation function """
        self.model.eval()
        eval_dl = DataLoader(dataset = eval_set, batch_size = self.cfg['batch_size'], shuffle = True, drop_last=True, num_workers = 2)

        total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0
        data_len = len(eval_dl)
        for val_id, val_mask, val_token , val_y_batch in eval_dl:
            with torch.no_grad():
                val_y_batch = val_y_batch.to(self.device)
                val_id = val_id.to(self.device)
                val_mask = val_mask.to(self.device)
                val_token = val_token.to(self.device)
                outputs = self.model(val_id, attention_mask = val_mask, token_type_ids = val_token, labels=val_y_batch)
                logits = outputs["logits"]
                acc, f1, precision,recall = compute_metrics_ce(logits, val_y_batch, device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall
        print("Target: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
            .format(total_acc/data_len, total_f1/data_len, total_p/data_len, total_recall/data_len))
    
    def save(self):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'], self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.model.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'model_bert.pt'))



class DANN_Trainer(object):
    def __init__(self, cfg, model, source, source_val, target, device, task):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.task = task
        self.source_dl = DataLoader(dataset = source, batch_size = cfg['batch_size'], shuffle = True, drop_last=True, num_workers = 2)
        self.source_val_dl = DataLoader(dataset = source_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True, num_workers = 2)
        self.target_dl = DataLoader(dataset = target, batch_size = cfg['batch_size'], shuffle = True, drop_last=True, num_workers = 2)
        self.loss_fn_sentiment_classifier = nn.CrossEntropyLoss()
        self.loss_fn_domain_classifier = nn.CrossEntropyLoss()
    def train(self):

        self.model = self.model.to(self.device)
        n_epochs = self.cfg['n_epochs']
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['lr'])
        batch_num = min(len(self.source_dl),len(self.target_dl))

        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl)
            target_iter = iter(self.target_dl)

            self.model.train()
            t0 = time.time()
            train_loss = 0
            best_loss = float('inf')
            for batch_index in range(batch_num):           
                p = float(batch_index + epoch * batch_num) / n_epochs * batch_num
                grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
                grl_lambda = torch.tensor(grl_lambda)

                src_id, src_mask, src_token, src_label = [ t.to(self.device) for t in next(source_iter)]
                

                optimizer.zero_grad()
                inputs = {
                "input_ids": src_id,
                "attention_mask": src_mask,
                "token_type_ids" : src_token,
                "grl_lambda" : grl_lambda,
                }
        
                sentiment_pred, domain_pred = self.model(**inputs)
                loss_s_sentiment = self.loss_fn_sentiment_classifier(sentiment_pred, src_label)
                y_s_domain = torch.zeros_like(src_label).to(self.device)
                loss_s_domain = self.loss_fn_domain_classifier(domain_pred, y_s_domain)

                tgt_id, tgt_mask, tgt_token, _ = [ t.to(self.device) for t in next(target_iter)]
                inputs = {
                        "input_ids": tgt_id,
                        "attention_mask": tgt_mask,
                        "token_type_ids" : tgt_token,
                        "grl_lambda" : grl_lambda,
                    }  
                _, domain_pred = self.model(**inputs)

                y_t_domain = torch.ones_like(src_label).to(self.device)
                loss_t_domain = self.loss_fn_domain_classifier(domain_pred, y_t_domain)

                # Combining the loss
                loss = loss_s_sentiment + loss_s_domain + loss_t_domain
                loss.backward()
                optimizer.step()

                train_loss += loss
                if((batch_index + 1)%self.cfg['print_after_steps'] == 0 ):
                    trainning_time = format_time(time.time() - t0)
                    print("Epoch: {}/{}, Step: {}/{}, time:{}, loss:{:.4f}"
                        .format(epoch+1, n_epochs, batch_index+1, batch_num, trainning_time, loss))
                
            avg_loss = train_loss / batch_num
    
            print("Epoch: {}/{}, Avg Loss:{:.4f}".format(epoch+1, n_epochs, avg_loss))
            
            avg_loss = self.eval_model(domain = 'Source') 
            if avg_loss < best_loss:
                self.save()
                best_loss = avg_loss
            print('--------------------------------------------------------------')
            t0 = time.time()
        avg_loss = self.eval_model(domain = 'Target') 

    def eval_model(self, domain = 'Target'):
        """ evaluation function """
        self.model.eval()
        if domain == 'Source':
            eval_dl = self.source_val_dl
        else:
            eval_dl = self.target_dl
        total_loss, total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0, 0
        data_len = len(eval_dl)
        for val_id, val_mask, val_token , val_y_batch in eval_dl:
            with torch.no_grad():
                inputs = {
                    "input_ids": val_id,
                    "attention_mask": val_mask,
                    "token_type_ids" : val_token,
                }
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)

                sentiment_pred, _ = self.model(**inputs)
                if domain == 'Source':
                    loss = self.loss_fn_sentiment_classifier(sentiment_pred, val_y_batch.to(self.device))
                    total_loss += loss
                acc, f1, precision,recall = compute_metrics_ce(sentiment_pred, val_y_batch.to(self.device), device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall
        if domain == 'Source':
            print("{}: Loss:{:.4f}, Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                            .format(domain, total_loss/data_len, total_acc/data_len, total_f1/data_len, total_p/data_len, total_recall/data_len))
        else:
            print("{}: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                .format(domain, total_acc/data_len, total_f1/data_len, total_p/data_len, total_recall/data_len))
        return  total_loss/data_len
    
    def save(self):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'], self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.model.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'dann.pt'))




class ADDN_Trainer(object):
    def __init__(self, cfg, src_encoder, tgt_encoder, src_classifier, discriminator, source, source_val, target, device, task):
        self.cfg = cfg
        self.device = device
        self.task = task
        self.src_encoder = src_encoder.to(device)
        self.tgt_encoder = tgt_encoder.to(device)
        self.src_classifier = src_classifier.to(device)
        self.discriminator = discriminator.to(device)
        self.path_classifier = ''
        self.path_source_encoder = ''
        self.source_dl = DataLoader(dataset = source, batch_size = cfg['batch_size'], shuffle = True, drop_last=True, num_workers = 2)
        self.source_val_dl = DataLoader(dataset = source_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True, num_workers = 2)
        self.target_dl = DataLoader(dataset = target, batch_size = cfg['batch_size'], shuffle = True, drop_last=True, num_workers = 2)
        self.criterion = nn.CrossEntropyLoss()

    def train_src(self):
        print('Train ADDA Source encoder and classifier...')
        self.src_encoder = self.src_encoder.to(self.device)
        self.src_classifier = self.src_classifier.to(self.device)
        n_epochs = self.cfg["n_epochs"]
        optimizer = optim.Adam(
                            list(self.src_encoder.parameters()) + list(self.src_classifier.parameters()),
                            lr=self.cfg['lr'],
                            betas=(self.cfg['beta1'], self.cfg['beta2']))
        t0 = time.time()
        num_batches = len(self.source_dl)
        best_loss = float('inf')
        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl) 
            train_loss = 0
            for batch_idx in range(num_batches):
                self.src_encoder.train()    
                self.src_classifier.train()    
                optimizer.zero_grad()
                
                src_id, src_mask, src_token, src_label = [ t.to(self.device) for t in next(source_iter)]
                inputs = {
                "input_ids": src_id,
                "attention_mask": src_mask,
                "token_type_ids" : src_token,
                }

                encode_pred = self.src_encoder(**inputs)
                sentiment_pred = self.src_classifier(encode_pred)
                loss = self.criterion(sentiment_pred.to(self.device), src_label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                if((batch_idx + 1)%self.cfg["print_after_steps"] == 0 ):
                    training_time = format_time(time.time() - t0)
                    print("Epoch [{}/{}] Step [{}/{}]: Time={}, Loss: {:.5f} ".format(epoch + 1, n_epochs, batch_idx + 1, num_batches, training_time, loss))
                    t0=time.time()

            print("Epoch: {}/{}, Avg Loss:{:.4f}"
            .format(epoch+1, n_epochs, train_loss/num_batches))
            
            
            val_loss = self.eval_src() 
            if val_loss < best_loss:
                self.save_source_encoder()
                self.save_classifier()
                best_loss = val_loss
            print('--------------------------------------------------------------')
            t0 = time.time()
        
    def train_tgt(self):
        print('Train ADDA Target encoder and discriminator...')
        # self.tgt_encoder.load_state_dict(torch.load(self.path_source_encoder))
        # self.src_classifier.load_state_dict(torch.load(self.path_classifier))
        # self.src_encoder = self.src_encoder.to(self.device)
        # self.tgt_encoder = self.tgt_encoder.to(self.device)
        # self.discriminator = self.discriminator.to(self.device)

        self.tgt_encoder.train()
        self.discriminator.train()

        optimizer_tgt = optim.Adam(self.tgt_encoder.parameters(),
                                lr=self.cfg['lr_tgt'],
                                betas=(self.cfg['beta1'], self.cfg['beta2']))
        optimizer_discriminator = optim.Adam(self.discriminator.parameters(),
                                    lr=self.cfg['lr_tgt'],
                                    betas=(self.cfg['beta1'], self.cfg['beta2']))
        batch_num = min(len(self.source_dl), len(self.target_dl))
        n_epochs = self.cfg["n_epochs"]
        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl)
            target_iter = iter(self.target_dl)
            t0 = time.time()
            epoch_d_loss = 0
            epoch_g_loss = 0
            for batch in range(batch_num):

                optimizer_discriminator.zero_grad()
                src_id, src_mask, src_token, src_label = [ t.to(self.device) for t in next(source_iter)]
                tgt_id, tgt_mask, tgt_token, _ = [ t.to(self.device) for t in next(target_iter)]

                feat_src = self.src_encoder(src_id, attention_mask=src_mask, token_type_ids=src_token)
                feat_tgt = self.tgt_encoder(tgt_id, attention_mask=tgt_mask, token_type_ids=tgt_token)
                feat_concat = torch.cat((feat_src, feat_tgt), 0)
                pred_concat = self.discriminator(feat_concat)

                label_src = torch.ones_like(src_label)
                label_tgt = torch.zeros_like(src_label)
                label_concat = torch.cat((label_src, label_tgt), 0).to(self.device)
                loss_discriminator = self.criterion(pred_concat, label_concat)
                epoch_d_loss += loss_discriminator.item()
                loss_discriminator.backward()

                optimizer_discriminator.step()

                optimizer_discriminator.zero_grad()
                optimizer_tgt.zero_grad()

                feat_tgt = self.tgt_encoder(tgt_id, attention_mask=tgt_mask, token_type_ids=tgt_token)
                pred_tgt = self.discriminator(feat_tgt)
                label_tgt = torch.ones_like(src_label).to(self.device)

                loss_tgt = self.criterion(pred_tgt, label_tgt)
                epoch_g_loss += loss_tgt.item()
                loss_tgt.backward()

                optimizer_tgt.step()

                if ((batch + 1) % self.cfg['print_after_steps'] == 0):
                    training_time = format_time(time.time() - t0)
                    print("Epoch [{}/{}] Step [{}/{}]: Time:{}, d_loss={:.5f}, g_loss={:.5f} "
                        .format(epoch + 1,
                                self.cfg['n_epochs'],
                                batch + 1,
                                batch_num,
                                training_time,
                                loss_discriminator.item(),
                                loss_tgt.item()))
        self.eval_tgt()
        self.save_target_encoder()

  

    def eval_src(self):
        """Evaluate classifier for source domain."""
        self.src_encoder.eval()
        self.src_classifier.eval()

        total_loss, total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0, 0
        data_len = len(self.source_val_dl)
        for val_id, val_mask, val_token , val_y_batch in self.source_val_dl:
            with torch.no_grad():
                inputs = {
                    "input_ids": val_id,
                    "attention_mask": val_mask,
                    "token_type_ids" : val_token,
                }
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)

                encode_pred= self.src_encoder(**inputs)
                sentiment_pred = self.src_classifier(encode_pred)
                loss = self.criterion(sentiment_pred, val_y_batch.to(self.device)).to(self.device)
                total_loss += loss
                acc, f1, precision,recall = compute_metrics_ce(sentiment_pred, val_y_batch.to(self.device), device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall
        print("Source: Loss:{:.4f}, Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_loss/data_len, total_acc/data_len, total_f1/data_len, total_p/data_len, total_recall/data_len))
        return  total_loss/data_len
    

    def eval_tgt(self):
        """Evaluation for target encoder by source classifier on target dataset."""
        self.tgt_encoder.eval()
        self.src_classifier.eval()

        total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0

        target_iterator = iter(self.target_dl)
        data_len = len(self.target_dl)
        for batch in range(data_len):
            with torch.no_grad():
                input_ids, attention_mask, token_type_ids, labels = next(target_iterator)
                encode_pred = self.tgt_encoder(input_ids.to(self.device),attention_mask=attention_mask.to(self.device),token_type_ids=token_type_ids.to(self.device))
                sentiment_pred = self.src_classifier(encode_pred)
                acc, f1, precision,recall  = compute_metrics_ce(sentiment_pred, labels.to(self.device), device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall

        print("Target: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_acc/data_len, total_f1/data_len, total_p/data_len, total_recall/data_len))


    def save_classifier(self):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'], self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.src_classifier.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'classifier.pt'))
        self.path_classifier = os.path.join(self.cfg['results_dir'], self.task, 'classifier.pt')
    
    def save_source_encoder(self):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'], self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.src_encoder.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'source_encoder.pt'))
        self.path_source_encoder = os.path.join(self.cfg['results_dir'], self.task, 'source_encoder.pt')

    def save_target_encoder(self):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'],  self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.tgt_encoder.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'target_encoder.pt'))






