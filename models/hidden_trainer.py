
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import time
from utils import format_time, compute_metrics_ce, compute_metrics_bce, compute_accuracy_v3
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, ConcatDataset




class Hidden_Trainer_One(object):
    def __init__(self, cfg, tgt_encoder, src_classifier, discriminator, source, source_val, target, target_val, device, task):
        self.cfg = cfg
        self.device = device
        self.task = task
        self.tgt_encoder = tgt_encoder.to(device)
        self.src_classifier = src_classifier.to(device)
        self.discriminator = discriminator.to(device)
        self.source_dl = DataLoader(dataset = source, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.source_val_dl = DataLoader(dataset = source_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.target_dl = DataLoader(dataset = target, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.target_val_dl = DataLoader(dataset = target_val, batch_size = cfg['batch_size'], shuffle = True)
        self.criterion = nn.CrossEntropyLoss()

    def train_classifier(self):
        print('Train Source classifier...')
        self.src_classifier = self.src_classifier
        n_epochs = self.cfg["classifier_epochs"]
        optimizer = optim.Adam(self.src_classifier.parameters(), lr=self.cfg['lr_c'])
        t0 = time.time()
        num_batches = len(self.source_dl)
        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl) 
            train_loss = 0
            self.src_classifier.train()  
            for batch_idx in range(num_batches):  
                optimizer.zero_grad()
                source, src_label = [t.to(self.device) for t in next(source_iter)]
                y_pred = self.src_classifier(source)
                loss = self.criterion(y_pred.to(self.device), src_label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                if((batch_idx + 1)%self.cfg["print_after_steps"] == 0 ):
                    training_time = format_time(time.time() - t0)
                    print("Epoch [{}/{}] Step [{}/{}]: Time={}, Loss: {:.5f} "
                        .format(epoch + 1, 
                                n_epochs, 
                                batch_idx + 1, 
                                num_batches, 
                                training_time, 
                                loss))

            print("Epoch: {}/{}, Time:{}, Avg Loss:{:.4f}"
                .format(epoch+1, 
                        n_epochs, 
                        format_time(time.time() - t0), 
                        train_loss/num_batches))
            
            val_loss = self.eval_classifier() 
            print('--------------------------------------------------------------')
            t0 = time.time()
            self.save_classifier(epoch)
        
    def train_tgt(self):
        print('Train Target encoder and discriminator...')
        self.tgt_encoder = self.tgt_encoder
        self.discriminator = self.discriminator

        optimizer_tgt = optim.Adam(self.tgt_encoder.parameters(), lr=self.cfg['lr_encoder'])
        optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.cfg['lr_dis'])
        batch_num = min(len(self.source_dl), len(self.target_dl))
        n_epochs = self.cfg["n_epochs"]
        best_f = 0.00
        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl)
            target_iter = iter(self.target_dl)

            epoch_d_loss = 0
            epoch_g_loss = 0
            t0 = time.time()
            self.tgt_encoder.train()
            self.discriminator.train()
            for batch in range(batch_num):
                p = float(batch + epoch * batch_num) / (n_epochs * batch_num)
                grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1              
                
                try:
                    source, src_label = [ t.to(self.device) for t in next(source_iter)]
                except StopIteration:
                    source_iter = iter(self.source_dl)
                    source, src_label = [ t.to(self.device) for t in next(source_iter)]
                
                try:
                    target, _  = [ t.to(self.device) for t in next(target_iter)]
                except StopIteration:
                    target_iter = iter(self.target_dl)
                    target, _  = [ t.to(self.device) for t in next(target_iter)]

    
                source_label = torch.ones_like(src_label).to(self.device)
                target_label = torch.zeros_like(src_label).to(self.device)
                genrated_target = self.tgt_encoder(target).to(self.device)

                optimizer_discriminator.zero_grad()
                pred_source = self.discriminator(source, grl_lambda)
                pred_generated = self.discriminator(genrated_target.detach(), grl_lambda)
                loss_generated = self.criterion(pred_generated, target_label)
                loss_source = self.criterion(pred_source, source_label)
                loss_dis = (loss_generated + loss_source)/2
                epoch_d_loss += loss_dis.item()
                loss_dis.backward()
                optimizer_discriminator.step()

                optimizer_tgt.zero_grad()
                pre_target = self.discriminator(genrated_target)
                loss_gen = self.criterion(pre_target, source_label.to(self.device))

                epoch_g_loss += loss_gen.item()
                loss_gen.backward()
                optimizer_tgt.step()


                if ((batch + 1) % self.cfg['print_after_steps'] == 0):
                    training_time = format_time(time.time() - t0)
                    print("Epoch [{}/{}] Step [{}/{}]: Time:{}, d_loss={:.5f}, g_loss={:.5f} "
                        .format(epoch + 1,
                                n_epochs,
                                batch + 1,
                                batch_num,
                                training_time,
                                loss_dis,
                                loss_gen))

            print("Epoch: {}/{}, Time:{}, Loss:{:.4f}, discriminator loss:{:.4f}, generator loss:{:.4f}"
                .format(epoch+1, 
                    n_epochs, 
                    format_time(time.time() - t0), 
                    epoch_d_loss/batch_num + epoch_g_loss/batch_num, 
                    epoch_d_loss/batch_num, 
                    epoch_g_loss/batch_num))

            # f1 = self.eval_model()
            self.save_target_encoder(epoch)

    def eval_classifier(self):
        """Evaluate classifier for source domain."""
        self.src_classifier.eval()
        total_loss, total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0, 0
        data_len = len(self.source_val_dl)
        for eval, labels in self.source_val_dl:
            with torch.no_grad():
                eval = eval.to(self.device)
                labels = labels.to(self.device)
                y_pred = self.src_classifier(eval)
                loss = self.criterion(y_pred, labels)
                total_loss += loss
                acc, f1, precision,recall = compute_metrics_ce(y_pred, labels, device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall
        print("Source: Loss:{:.4f}, Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_loss/data_len, 
                                total_acc/data_len, 
                                total_f1/data_len, 
                                total_p/data_len, 
                                total_recall/data_len))
        return  total_loss/data_len
    
    def eval_model(self):
        """Evaluation for target encoder by source classifier on target dataset."""
        self.tgt_encoder.eval()
        self.src_classifier.eval()

        total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0
        total_acc_2, total_f1_2, total_p_2, total_recall_2 = 0, 0, 0, 0

        target_iterator = iter(self.target_val_dl)
        data_len = len(self.target_val_dl)
        for batch in range(data_len):
            with torch.no_grad():
                eval, labels = next(target_iterator)
                encode_pred = self.tgt_encoder(eval.to(self.device))
                y_pred = self.src_classifier(encode_pred)
                acc, f1, precision,recall  = compute_metrics_ce(y_pred, labels.to(self.device), device=self.device)
                acc_2, f1_2, precision_2,recall_2  = compute_accuracy_v3(y_pred, labels.to(self.device), device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall

                total_acc_2 += acc_2
                total_f1_2 += f1_2
                total_p_2 += precision_2
                total_recall_2 += recall_2

        print("Target: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_acc/data_len, 
                                total_f1/data_len, 
                                total_p/data_len, 
                                total_recall/data_len))
        print("Target_2: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_acc_2/data_len, 
                                total_f1_2/data_len, 
                                total_p_2/data_len, 
                                total_recall_2/data_len))
        return total_f1/data_len
    
    def predict(self):
        """Evaluation for target encoder by source classifier on target dataset."""
        y_pred = np.array([])
        y_true = np.array([])
        self.tgt_encoder.eval()
        self.src_classifier.eval()
        target_iterator = iter(self.target_val_dl)
        data_len = len(self.target_val_dl)
        for batch in range(data_len):
            with torch.no_grad():
                eval, labels = next(target_iterator)
                encode_pred = self.tgt_encoder(eval.to(self.device))
                pred = self.src_classifier(encode_pred)
                pred = pred.max(dim = 1)[1]
                y_pred = np.append(y_pred, pred.cpu().numpy())
                y_true = np.append(y_true, labels.cpu().numpy())
        return y_pred, y_true

    def save_classifier(self, epoch):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'], self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.src_classifier.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'classifier_'+str(epoch)+'.pt'))

    def save_target_encoder(self, epoch):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'],  self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.tgt_encoder.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'target_encoder'+str(epoch)+'.pt'))


class Hidden_Trainer_Two(object):
    def __init__(self, cfg, tgt_encoder, src_classifier, discriminator, source, source_val, target, target_val, device, task):
        self.cfg = cfg
        self.device = device
        self.task = task
        self.tgt_encoder = tgt_encoder.to(device)
        self.src_classifier = src_classifier.to(device)
        self.discriminator = discriminator.to(device)
        self.source_dl = DataLoader(dataset = source, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.source_val_dl = DataLoader(dataset = source_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.target_dl = DataLoader(dataset = target, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.target_val_dl = DataLoader(dataset = target_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.criterion = nn.CrossEntropyLoss()

    def train_classifier(self):
        print('Train Source classifier...')
        self.src_classifier = self.src_classifier
        n_epochs = self.cfg["classifier_epochs"]
        optimizer = optim.Adam(self.src_classifier.parameters(), lr=self.cfg['lr_c'])
        t0 = time.time()
        num_batches = len(self.source_dl)
        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl) 
            train_loss = 0
            self.src_classifier.train()  
            for batch_idx in range(num_batches):  
                optimizer.zero_grad()
                source, src_label = [t.to(self.device) for t in next(source_iter)]
                y_pred = self.src_classifier(source)
                loss = self.criterion(y_pred.to(self.device), src_label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                if((batch_idx + 1)%self.cfg["print_after_steps"] == 0 ):
                    training_time = format_time(time.time() - t0)
                    print("Epoch [{}/{}] Step [{}/{}]: Time={}, Loss: {:.5f} "
                        .format(epoch + 1, 
                                n_epochs, 
                                batch_idx + 1, 
                                num_batches, 
                                training_time, 
                                loss))

            print("Epoch: {}/{}, Time:{}, Avg Loss:{:.4f}"
                .format(epoch+1, 
                        n_epochs, 
                        format_time(time.time() - t0), 
                        train_loss/num_batches))
            
            val_loss = self.eval_classifier() 
            self.eval_model()
            print('--------------------------------------------------------------')
            t0 = time.time()
            self.save_classifier()
        
    def train_tgt(self):
        print('Train Target encoder and discriminator...')
        self.tgt_encoder = self.tgt_encoder
        self.discriminator = self.discriminator

        optimizer_tgt = optim.Adam(self.tgt_encoder.parameters(), lr=self.cfg['lr_encoder'])
        optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.cfg['lr_dis'])
        batch_num = min(len(self.source_dl), len(self.target_dl))
        n_epochs = self.cfg["n_epochs"]
        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl)
            target_iter = iter(self.target_dl)

            epoch_d_loss = 0
            epoch_g_loss = 0
            t0 = time.time()
            self.tgt_encoder.train()
            self.discriminator.train()
            for batch in range(batch_num):
                p = float(batch + epoch * batch_num) / (n_epochs * batch_num)
                grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1              
                
                try:
                    source, src_label = [ t.to(self.device) for t in next(source_iter)]
                except StopIteration:
                    source_iter = iter(self.source_dl)
                    source, src_label = [ t.to(self.device) for t in next(source_iter)]
                
                try:
                    target, _  = [ t.to(self.device) for t in next(target_iter)]
                except StopIteration:
                    target_iter = iter(self.target_dl)
                    target, _  = [ t.to(self.device) for t in next(target_iter)]

    
                source_label = torch.ones_like(src_label).to(self.device)
                target_label = torch.zeros_like(src_label).to(self.device)
                genrated_target = self.tgt_encoder(target).to(self.device)

                optimizer_discriminator.zero_grad()
                pred_source = self.discriminator(source, grl_lambda)
                pred_generated = self.discriminator(genrated_target.detach(), grl_lambda)
                loss_generated = self.criterion(pred_generated, target_label)
                loss_source = self.criterion(pred_source, source_label)
                loss_dis = (loss_generated + loss_source)/2
                epoch_d_loss += loss_dis.item()
                loss_dis.backward()
                optimizer_discriminator.step()

                optimizer_tgt.zero_grad()
                pre_target = self.discriminator(genrated_target)
                loss_gen = self.criterion(pre_target, source_label.to(self.device))

                epoch_g_loss += loss_gen.item()
                loss_gen.backward()
                optimizer_tgt.step()


                if ((batch + 1) % self.cfg['print_after_steps'] == 0):
                    training_time = format_time(time.time() - t0)
                    print("Epoch [{}/{}] Step [{}/{}]: Time:{}, d_loss={:.5f}, g_loss={:.5f} "
                        .format(epoch + 1,
                                n_epochs,
                                batch + 1,
                                batch_num,
                                training_time,
                                loss_dis,
                                loss_gen))

            print("Epoch: {}/{}, Time:{}, Loss:{:.4f}, discriminator loss:{:.4f}, generator loss:{:.4f}"
                .format(epoch+1, 
                    n_epochs, 
                    format_time(time.time() - t0), 
                    epoch_d_loss/batch_num + epoch_g_loss/batch_num, 
                    epoch_d_loss/batch_num, 
                    epoch_g_loss/batch_num))

        self.save_target_encoder()

    def eval_classifier(self):
        """Evaluate classifier for source domain."""
        self.src_classifier.eval()
        total_loss, total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0, 0
        data_len = len(self.source_val_dl)
        for eval, labels in self.source_val_dl:
            with torch.no_grad():
                eval = eval.to(self.device)
                labels = labels.to(self.device)
                y_pred = self.src_classifier(eval)
                loss = self.criterion(y_pred, labels)
                total_loss += loss
                acc, f1, precision,recall = compute_metrics_ce(y_pred, labels, device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall
        print("Source: Loss:{:.4f}, Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_loss/data_len, 
                                total_acc/data_len, 
                                total_f1/data_len, 
                                total_p/data_len, 
                                total_recall/data_len))
        return  total_loss/data_len
    
    def eval_model(self):
        """Evaluation for target encoder by source classifier on target dataset."""
        self.tgt_encoder.eval()
        self.src_classifier.eval()

        total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0
        total_acc_2, total_f1_2, total_p_2, total_recall_2 = 0, 0, 0, 0

        target_iterator = iter(self.target_val_dl)
        data_len = len(self.target_val_dl)
        for batch in range(data_len):
            with torch.no_grad():
                eval, labels = next(target_iterator)
                encode_pred = self.tgt_encoder(eval.to(self.device))
                y_pred = self.src_classifier(encode_pred)
                acc, f1, precision,recall  = compute_metrics_ce(y_pred, labels.to(self.device), device=self.device)
                acc_2, f1_2, precision_2,recall_2  = compute_accuracy_v3(y_pred, labels.to(self.device), device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall

                total_acc_2 += acc_2
                total_f1_2 += f1_2
                total_p_2 += precision_2
                total_recall_2 += recall_2

        print("Target: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_acc/data_len, 
                                total_f1/data_len, 
                                total_p/data_len, 
                                total_recall/data_len))
        print("Target_2: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_acc_2/data_len, 
                                total_f1_2/data_len, 
                                total_p_2/data_len, 
                                total_recall_2/data_len))

    def save_classifier(self):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'], self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.src_classifier.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'classifier.pt'))

    def save_target_encoder(self):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'],  self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.tgt_encoder.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'target_encoder.pt'))


class Hidden_Trainer_Three(object):
    def __init__(self, cfg, src_encoder, tgt_encoder, src_classifier, discriminator, source, source_val, target, target_val, device, task):
        self.cfg = cfg
        self.device = device
        self.task = task
        self.src_encoder = src_encoder.to(device)
        self.tgt_encoder = tgt_encoder.to(device)
        self.src_classifier = src_classifier.to(device)
        self.discriminator = discriminator.to(device)
        self.source_dl = DataLoader(dataset = source, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.source_val_dl = DataLoader(dataset = source_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.target_dl = DataLoader(dataset = target, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.target_val_dl = DataLoader(dataset = target_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.criterion = nn.CrossEntropyLoss()

    def train_classifier(self):
        print('Train Source classifier...')
        n_epochs = self.cfg["classifier_epochs"]
        optimizer_cls = optim.Adam(self.src_classifier.parameters(), lr=self.cfg['lr_c'])
        optimizer_src_en = optim.Adam(self.src_encoder.parameters(), lr=self.cfg['lr_c'])
        t0 = time.time()
        num_batches = len(self.source_dl)
        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl) 
            train_loss = 0
            self.src_classifier.train()  
            for batch_idx in range(num_batches):  
                optimizer_cls.zero_grad()
                optimizer_src_en.zero_grad()
                source, src_label = [t.to(self.device) for t in next(source_iter)]

                genrated_source = self.src_encoder(source).to(self.device)
                y_pred = self.src_classifier(genrated_source)
                loss = self.criterion(y_pred.to(self.device), src_label)
                train_loss += loss.item()
                loss.backward()
                optimizer_cls.step()
                optimizer_src_en.step()

                if((batch_idx + 1)%self.cfg["print_after_steps"] == 0 ):
                    training_time = format_time(time.time() - t0)
                    print("Epoch [{}/{}] Step [{}/{}]: Time={}, Loss: {:.5f} "
                        .format(epoch + 1, 
                                n_epochs, 
                                batch_idx + 1, 
                                num_batches, 
                                training_time, 
                                loss))

            print("Epoch: {}/{}, Time:{}, Avg Loss:{:.4f}"
                .format(epoch+1, 
                        n_epochs, 
                        format_time(time.time() - t0), 
                        train_loss/num_batches))
            
            val_loss = self.eval_classifier() 
            print('--------------------------------------------------------------')
            t0 = time.time()
            self.save_classifier()
            self.save_source_encoder()

        
    def train_tgt(self):
        print('Train Target encoder and discriminator...')
        # self.tgt_encoder = self.src_encoder

        optimizer_tgt = optim.Adam(self.tgt_encoder.parameters(), lr=self.cfg['lr_encoder'])
        optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.cfg['lr_dis'])
        batch_num = min(len(self.source_dl), len(self.target_dl))
        n_epochs = self.cfg["n_epochs"]
        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl)
            target_iter = iter(self.target_dl)

            epoch_d_loss = 0
            epoch_g_loss = 0
            t0 = time.time()
            self.tgt_encoder.train()
            self.discriminator.train()
            for batch in range(batch_num):
                p = float(batch + epoch * batch_num) / (n_epochs * batch_num)
                grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1              
                
                try:
                    source, src_label = [ t.to(self.device) for t in next(source_iter)]
                except StopIteration:
                    source_iter = iter(self.source_dl)
                    source, src_label = [ t.to(self.device) for t in next(source_iter)]
                
                try:
                    target, _  = [ t.to(self.device) for t in next(target_iter)]
                except StopIteration:
                    target_iter = iter(self.target_dl)
                    target, _  = [ t.to(self.device) for t in next(target_iter)]

    
                source_label = torch.ones_like(src_label).to(self.device)
                target_label = torch.zeros_like(src_label).to(self.device)
                genrated_source = self.src_encoder(source).to(self.device)
                genrated_target = self.tgt_encoder(target).to(self.device)

                optimizer_discriminator.zero_grad()
                pred_source = self.discriminator(genrated_source.detach(), grl_lambda)
                pred_generated = self.discriminator(genrated_target.detach(), grl_lambda)
                loss_generated = self.criterion(pred_generated, target_label)
                loss_source = self.criterion(pred_source, source_label)
                loss_dis = (loss_generated + loss_source)/2
                epoch_d_loss += loss_dis.item()
                loss_dis.backward()
                optimizer_discriminator.step()

                optimizer_tgt.zero_grad()
                pre_target = self.discriminator(genrated_target)
                loss_gen = self.criterion(pre_target, source_label.to(self.device))

                epoch_g_loss += loss_gen.item()
                loss_gen.backward()
                optimizer_tgt.step()


                if ((batch + 1) % self.cfg['print_after_steps'] == 0):
                    training_time = format_time(time.time() - t0)
                    print("Epoch [{}/{}] Step [{}/{}]: Time:{}, d_loss={:.5f}, g_loss={:.5f} "
                        .format(epoch + 1,
                                n_epochs,
                                batch + 1,
                                batch_num,
                                training_time,
                                loss_dis,
                                loss_gen))

            print("Epoch: {}/{}, Time:{}, Loss:{:.4f}, discriminator loss:{:.4f}, generator loss:{:.4f}"
                .format(epoch+1, 
                    n_epochs, 
                    format_time(time.time() - t0), 
                    epoch_d_loss/batch_num + epoch_g_loss/batch_num, 
                    epoch_d_loss/batch_num, 
                    epoch_g_loss/batch_num))

            self.eval_model()
            self.save_target_encoder()

  

    def eval_classifier(self):
        """Evaluate classifier for source domain."""
        self.src_classifier.eval()

        total_loss, total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0, 0
        data_len = len(self.source_val_dl)
        for eval, labels in self.source_val_dl:
            with torch.no_grad():
                eval = eval.to(self.device)
                labels = labels.to(self.device)
                y_pred = self.src_classifier(eval)
                loss = self.criterion(y_pred, labels)
                total_loss += loss
                acc, f1, precision,recall = compute_metrics_ce(y_pred, labels, device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall
        print("Source: Loss:{:.4f}, Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_loss/data_len, 
                                total_acc/data_len, 
                                total_f1/data_len, 
                                total_p/data_len, 
                                total_recall/data_len))
        return  total_loss/data_len
    

    def eval_model(self):
        """Evaluation for target encoder by source classifier on target dataset."""
        self.tgt_encoder.eval()
        self.src_classifier.eval()

        total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0
        total_acc_2, total_f1_2, total_p_2, total_recall_2 = 0, 0, 0, 0

        target_iterator = iter(self.target_val_dl)
        data_len = len(self.target_val_dl)
        for batch in range(data_len):
            with torch.no_grad():
                eval, labels = next(target_iterator)
                encode_pred = self.tgt_encoder(eval.to(self.device))
                y_pred = self.src_classifier(encode_pred)
                acc, f1, precision,recall  = compute_metrics_ce(y_pred, labels.to(self.device), device=self.device)
                acc_2, f1_2, precision_2,recall_2  = compute_accuracy_v3(y_pred, labels.to(self.device), device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall

                total_acc_2 += acc_2
                total_f1_2 += f1_2
                total_p_2 += precision_2
                total_recall_2 += recall_2

        print("Target: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_acc/data_len, 
                                total_f1/data_len, 
                                total_p/data_len, 
                                total_recall/data_len))
        print("Target_2: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_acc_2/data_len, 
                                total_f1_2/data_len, 
                                total_p_2/data_len, 
                                total_recall_2/data_len))

    def save_classifier(self):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'], self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.src_classifier.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'classifier.pt'))

    def save_target_encoder(self):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'],  self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.tgt_encoder.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'target_encoder.pt'))


    def save_source_encoder(self):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'],  self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.src_encoder.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'source_encoder.pt'))


class Hidden_Trainer_Four(object):
    def __init__(self, cfg, model, source, source_val, target, target_val, device, task):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.task = task
        self.source_dl = DataLoader(dataset = source, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.source_val_dl = DataLoader(dataset = source_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.target_dl = DataLoader(dataset = target, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.target_val_dl = DataLoader(dataset = target_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
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




class Hidden_Trainer_repeat(object):
    def __init__(self, cfg, tgt_encoder, src_classifier, discriminator, source, source_val, target, target_val, device, task):
        self.cfg = cfg
        self.device = device
        self.task = task
        self.tgt_encoder = tgt_encoder.to(device)
        self.src_classifier = src_classifier.to(device)
        self.discriminator = discriminator.to(device)
        self.source_dl = DataLoader(dataset = source, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.source_val_dl = DataLoader(dataset = source_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.target_dl = DataLoader(dataset = target, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.target_val_dl = DataLoader(dataset = target_val, batch_size = cfg['batch_size'], shuffle = True)
        self.criterion = nn.CrossEntropyLoss()

    def train_classifier(self):
        print('Train Source classifier...')
        self.src_classifier = self.src_classifier
        n_epochs = self.cfg["classifier_epochs"]
        optimizer = optim.Adam(self.src_classifier.parameters(), lr=self.cfg['lr_c'])
        t0 = time.time()
        num_batches = len(self.source_dl)
        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl) 
            train_loss = 0
            self.src_classifier.train()  
            for batch_idx in range(num_batches):  
                optimizer.zero_grad()
                source, src_label = [t.to(self.device) for t in next(source_iter)]
                y_pred = self.src_classifier(source)
                loss = self.criterion(y_pred.to(self.device), src_label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                if((batch_idx + 1)%self.cfg["print_after_steps"] == 0 ):
                    training_time = format_time(time.time() - t0)
                    print("Epoch [{}/{}] Step [{}/{}]: Time={}, Loss: {:.5f} "
                        .format(epoch + 1, 
                                n_epochs, 
                                batch_idx + 1, 
                                num_batches, 
                                training_time, 
                                loss))

            print("Epoch: {}/{}, Time:{}, Avg Loss:{:.4f}"
                .format(epoch+1, 
                        n_epochs, 
                        format_time(time.time() - t0), 
                        train_loss/num_batches))
            
            val_loss = self.eval_classifier() 
            print('--------------------------------------------------------------')
            t0 = time.time()
            self.save_classifier(epoch)
        
    def train_tgt(self):
        print('Train Target encoder and discriminator...')
        self.tgt_encoder = self.tgt_encoder
        self.discriminator = self.discriminator

        optimizer_tgt = optim.Adam(self.tgt_encoder.parameters(), lr=self.cfg['lr_encoder'])
        optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.cfg['lr_dis'])
        batch_num = max(len(self.source_dl), len(self.target_dl))
        n_epochs = self.cfg["n_epochs"]
        best_f = 0.00
        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl)
            target_iter = iter(self.target_dl)

            epoch_d_loss = 0
            epoch_g_loss = 0
            t0 = time.time()
            self.tgt_encoder.train()
            self.discriminator.train()
            for batch in range(batch_num):
                p = float(batch + epoch * batch_num) / (n_epochs * batch_num)
                grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1              
                
                try:
                    source, src_label = [ t.to(self.device) for t in next(source_iter)]
                except StopIteration:
                    source_iter = iter(self.source_dl)
                    source, src_label = [ t.to(self.device) for t in next(source_iter)]
                
                try:
                    target, _  = [ t.to(self.device) for t in next(target_iter)]
                except StopIteration:
                    target_iter = iter(self.target_dl)
                    target, _  = [ t.to(self.device) for t in next(target_iter)]

    
                source_label = torch.ones_like(src_label).to(self.device)
                target_label = torch.zeros_like(src_label).to(self.device)
                genrated_target = self.tgt_encoder(target).to(self.device)

                optimizer_discriminator.zero_grad()
                pred_source = self.discriminator(source, grl_lambda)
                pred_generated = self.discriminator(genrated_target.detach(), grl_lambda)
                loss_generated = self.criterion(pred_generated, target_label)
                loss_source = self.criterion(pred_source, source_label)
                loss_dis = (loss_generated + loss_source)/2
                epoch_d_loss += loss_dis.item()
                loss_dis.backward()
                optimizer_discriminator.step()

                optimizer_tgt.zero_grad()
                pre_target = self.discriminator(genrated_target)
                loss_gen = self.criterion(pre_target, source_label.to(self.device))

                epoch_g_loss += loss_gen.item()
                loss_gen.backward()
                optimizer_tgt.step()


                if ((batch + 1) % self.cfg['print_after_steps'] == 0):
                    training_time = format_time(time.time() - t0)
                    print("Epoch [{}/{}] Step [{}/{}]: Time:{}, d_loss={:.5f}, g_loss={:.5f} "
                        .format(epoch + 1,
                                n_epochs,
                                batch + 1,
                                batch_num,
                                training_time,
                                loss_dis,
                                loss_gen))

            print("Epoch: {}/{}, Time:{}, Loss:{:.4f}, discriminator loss:{:.4f}, generator loss:{:.4f}"
                .format(epoch+1, 
                    n_epochs, 
                    format_time(time.time() - t0), 
                    epoch_d_loss/batch_num + epoch_g_loss/batch_num, 
                    epoch_d_loss/batch_num, 
                    epoch_g_loss/batch_num))

            # f1 = self.eval_model()
            self.save_target_encoder(epoch)

    def eval_classifier(self):
        """Evaluate classifier for source domain."""
        self.src_classifier.eval()
        total_loss, total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0, 0
        data_len = len(self.source_val_dl)
        for eval, labels in self.source_val_dl:
            with torch.no_grad():
                eval = eval.to(self.device)
                labels = labels.to(self.device)
                y_pred = self.src_classifier(eval)
                loss = self.criterion(y_pred, labels)
                total_loss += loss
                acc, f1, precision,recall = compute_metrics_ce(y_pred, labels, device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall
        print("Source: Loss:{:.4f}, Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_loss/data_len, 
                                total_acc/data_len, 
                                total_f1/data_len, 
                                total_p/data_len, 
                                total_recall/data_len))
        return  total_loss/data_len
    
    def eval_model(self):
        """Evaluation for target encoder by source classifier on target dataset."""
        self.tgt_encoder.eval()
        self.src_classifier.eval()

        total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0
        total_acc_2, total_f1_2, total_p_2, total_recall_2 = 0, 0, 0, 0

        target_iterator = iter(self.target_val_dl)
        data_len = len(self.target_val_dl)
        for batch in range(data_len):
            with torch.no_grad():
                eval, labels = next(target_iterator)
                encode_pred = self.tgt_encoder(eval.to(self.device))
                y_pred = self.src_classifier(encode_pred)
                acc, f1, precision,recall  = compute_metrics_ce(y_pred, labels.to(self.device), device=self.device)
                acc_2, f1_2, precision_2,recall_2  = compute_accuracy_v3(y_pred, labels.to(self.device), device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall

                total_acc_2 += acc_2
                total_f1_2 += f1_2
                total_p_2 += precision_2
                total_recall_2 += recall_2

        print("Target: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_acc/data_len, 
                                total_f1/data_len, 
                                total_p/data_len, 
                                total_recall/data_len))
        print("Target_2: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_acc_2/data_len, 
                                total_f1_2/data_len, 
                                total_p_2/data_len, 
                                total_recall_2/data_len))
        return total_f1/data_len
    
    def predict(self):
        """Evaluation for target encoder by source classifier on target dataset."""
        self.tgt_encoder.eval()
        self.src_classifier.eval()

        target_iterator = iter(self.target_val_dl)
        data_len = len(self.target_val_dl)
        for batch in range(data_len):
            with torch.no_grad():
                eval, labels = next(target_iterator)
                encode_pred = self.tgt_encoder(eval.to(self.device))
                y_pred = self.src_classifier(encode_pred)
        return y_pred.max(dim = 1)[1], labels

    def save_classifier(self, epoch):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'], self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.src_classifier.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'classifier_'+str(epoch)+'.pt'))

    def save_target_encoder(self, epoch):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'],  self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.tgt_encoder.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'target_encoder'+str(epoch)+'.pt'))



class Hidden_Trainer_decay(object):
    def __init__(self, cfg, tgt_encoder, src_classifier, discriminator, source, source_val, target, target_val, device, task):
        self.cfg = cfg
        self.device = device
        self.task = task
        self.tgt_encoder = tgt_encoder.to(device)
        self.src_classifier = src_classifier.to(device)
        self.discriminator = discriminator.to(device)
        self.source_dl = DataLoader(dataset = source, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.source_val_dl = DataLoader(dataset = source_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.target_dl = DataLoader(dataset = target, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.target_val_dl = DataLoader(dataset = target_val, batch_size = cfg['batch_size'], shuffle = True, drop_last=True)
        self.criterion = nn.CrossEntropyLoss()

    def train_classifier(self):
        print('Train Source classifier...')
        self.src_classifier = self.src_classifier
        n_epochs = self.cfg["classifier_epochs"]
        optimizer = optim.Adam(self.src_classifier.parameters(), lr=self.cfg['lr_c'])
        t0 = time.time()
        num_batches = len(self.source_dl)
        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl) 
            train_loss = 0
            self.src_classifier.train()  
            for batch_idx in range(num_batches):  
                optimizer.zero_grad()
                source, src_label = [t.to(self.device) for t in next(source_iter)]
                y_pred = self.src_classifier(source)
                loss = self.criterion(y_pred.to(self.device), src_label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                if((batch_idx + 1)%self.cfg["print_after_steps"] == 0 ):
                    training_time = format_time(time.time() - t0)
                    print("Epoch [{}/{}] Step [{}/{}]: Time={}, Loss: {:.5f} "
                        .format(epoch + 1, 
                                n_epochs, 
                                batch_idx + 1, 
                                num_batches, 
                                training_time, 
                                loss))

            print("Epoch: {}/{}, Time:{}, Avg Loss:{:.4f}"
                .format(epoch+1, 
                        n_epochs, 
                        format_time(time.time() - t0), 
                        train_loss/num_batches))
            
            val_loss = self.eval_classifier() 
            print('--------------------------------------------------------------')
            t0 = time.time()
            self.save_classifier(epoch)
        
    def train_tgt(self):
        print('Train Target encoder and discriminator...')
        self.tgt_encoder = self.tgt_encoder
        self.discriminator = self.discriminator

        optimizer_tgt = optim.Adam(self.tgt_encoder.parameters(), lr=self.cfg['lr_encoder'])
        optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.cfg['lr_dis'])
        batch_num = min(len(self.source_dl), len(self.target_dl))
        n_epochs = self.cfg["n_epochs"]
        best_f = 0.00
        for epoch in range(n_epochs):
            source_iter = iter(self.source_dl)
            target_iter = iter(self.target_dl)

            epoch_d_loss = 0
            epoch_g_loss = 0
            t0 = time.time()
            self.tgt_encoder.train()
            self.discriminator.train()
            for batch in range(batch_num):
                p = float(batch + epoch * batch_num) / (n_epochs * batch_num)
                grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1              
                
                try:
                    source, src_label = [ t.to(self.device) for t in next(source_iter)]
                except StopIteration:
                    source_iter = iter(self.source_dl)
                    source, src_label = [ t.to(self.device) for t in next(source_iter)]
                
                try:
                    target, _  = [ t.to(self.device) for t in next(target_iter)]
                except StopIteration:
                    target_iter = iter(self.target_dl)
                    target, _  = [ t.to(self.device) for t in next(target_iter)]

    
                source_label = torch.ones_like(src_label).to(self.device)
                target_label = torch.zeros_like(src_label).to(self.device)
                genrated_target = self.tgt_encoder(target).to(self.device)

                optimizer_discriminator.zero_grad()
                pred_source = self.discriminator(source, grl_lambda)
                pred_generated = self.discriminator(genrated_target.detach(), grl_lambda)
                loss_generated = self.criterion(pred_generated, target_label)
                loss_source = self.criterion(pred_source, source_label)
                loss_dis = (loss_generated + loss_source)/2
                epoch_d_loss += loss_dis.item()
                loss_dis.backward()
                optimizer_discriminator.step()

                optimizer_tgt.zero_grad()
                pre_target = self.discriminator(genrated_target)
                loss_gen = self.criterion(pre_target, source_label.to(self.device))

                epoch_g_loss += loss_gen.item()
                loss_gen.backward()
                optimizer_tgt.step()


                if ((batch + 1) % self.cfg['print_after_steps'] == 0):
                    training_time = format_time(time.time() - t0)
                    print("Epoch [{}/{}] Step [{}/{}]: Time:{}, d_loss={:.5f}, g_loss={:.5f} "
                        .format(epoch + 1,
                                n_epochs,
                                batch + 1,
                                batch_num,
                                training_time,
                                loss_dis,
                                loss_gen))

            print("Epoch: {}/{}, Time:{}, Loss:{:.4f}, discriminator loss:{:.4f}, generator loss:{:.4f}"
                .format(epoch+1, 
                    n_epochs, 
                    format_time(time.time() - t0), 
                    epoch_d_loss/batch_num + epoch_g_loss/batch_num, 
                    epoch_d_loss/batch_num, 
                    epoch_g_loss/batch_num))

            # f1 = self.eval_model()
            self.save_target_encoder(epoch)

    def eval_classifier(self):
        """Evaluate classifier for source domain."""
        self.src_classifier.eval()
        total_loss, total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0, 0
        data_len = len(self.source_val_dl)
        for eval, labels in self.source_val_dl:
            with torch.no_grad():
                eval = eval.to(self.device)
                labels = labels.to(self.device)
                y_pred = self.src_classifier(eval)
                loss = self.criterion(y_pred, labels)
                total_loss += loss
                acc, f1, precision,recall = compute_metrics_ce(y_pred, labels, device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall
        print("Source: Loss:{:.4f}, Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_loss/data_len, 
                                total_acc/data_len, 
                                total_f1/data_len, 
                                total_p/data_len, 
                                total_recall/data_len))
        return  total_loss/data_len
    
    def eval_model(self):
        """Evaluation for target encoder by source classifier on target dataset."""
        self.tgt_encoder.eval()
        self.src_classifier.eval()

        total_acc, total_f1, total_p, total_recall = 0, 0, 0, 0
        total_acc_2, total_f1_2, total_p_2, total_recall_2 = 0, 0, 0, 0

        target_iterator = iter(self.target_val_dl)
        data_len = len(self.target_val_dl)
        for batch in range(data_len):
            with torch.no_grad():
                eval, labels = next(target_iterator)
                encode_pred = self.tgt_encoder(eval.to(self.device))
                y_pred = self.src_classifier(encode_pred)
                acc, f1, precision,recall  = compute_metrics_ce(y_pred, labels.to(self.device), device=self.device)
                acc_2, f1_2, precision_2,recall_2  = compute_accuracy_v3(y_pred, labels.to(self.device), device=self.device)
                total_acc += acc
                total_f1 += f1
                total_p += precision
                total_recall += recall

                total_acc_2 += acc_2
                total_f1_2 += f1_2
                total_p_2 += precision_2
                total_recall_2 += recall_2

        print("Target: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_acc/data_len, 
                                total_f1/data_len, 
                                total_p/data_len, 
                                total_recall/data_len))
        print("Target_2: Acc:{:.4f}, F1:{:.4f}, Precision:{:.4f}, recall:{:.4f}"
                        .format(total_acc_2/data_len, 
                                total_f1_2/data_len, 
                                total_p_2/data_len, 
                                total_recall_2/data_len))
        return total_f1/data_len
    
    def predict(self):
        """Evaluation for target encoder by source classifier on target dataset."""
        self.tgt_encoder.eval()
        self.src_classifier.eval()

        target_iterator = iter(self.target_val_dl)
        data_len = len(self.target_val_dl)
        for batch in range(data_len):
            with torch.no_grad():
                eval, labels = next(target_iterator)
                encode_pred = self.tgt_encoder(eval.to(self.device))
                y_pred = self.src_classifier(encode_pred)
        return y_pred.max(dim = 1)[1], labels

    def save_classifier(self, epoch):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'], self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.src_classifier.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'classifier_'+str(epoch)+'.pt'))

    def save_target_encoder(self, epoch):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg['results_dir'],  self.task)):
            os.makedirs(os.path.join(self.cfg['results_dir'], self.task))
        torch.save(self.tgt_encoder.state_dict(),
                        os.path.join(self.cfg['results_dir'], self.task, 'target_encoder'+str(epoch)+'.pt'))