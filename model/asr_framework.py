import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from lib.logger import (
    get_logger, 
    PD_Stats, 
)
from lib.utils import (
    get_log_dir, 
    get_model_params,   
)
from lib.metrics import test_metrics

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    i = 0
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            i += 1
            m.train()
    print('**found {} Dropout layers for random sampling'.format(i))

class PGASR(object):
    def __init__(self, models, optimizers, dataloader, graph, args):
        super(PGASR, self).__init__()
        self.pre_model1, self.pre_model2, self.re_model = models
        self.optimizer1, self.optimizer2, self.optimizer3 = optimizers
        self.train_loader = dataloader['train']
        self.split_point = len(self.train_loader)//2
        self.val_loader = dataloader['val']
        self.test_loader = dataloader['test']
        self.scaler = dataloader['scaler']
        self.graph = graph
        self.args = args

        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)
        
        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir)

        self.best_path1 = os.path.join(self.args.log_dir, 'best_pre_model1.pth')
        self.best_path2 = os.path.join(self.args.log_dir, 'best_pre_model2.pth')
        self.best_path3 = os.path.join(self.args.log_dir, 'best_re_model.pth')
        
        # create a panda object to log loss and acc
        self.pretraining_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'), 
            ['epoch', 'train_loss', 'val_loss'],
        )

        self.retraining_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'), 
            ['epoch', 'train_loss', 'val_loss'],
        )
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

    def train_epoch(self, epoch, mode, pe_weights_list=None):

        if mode == 'pre-train1':
            start = 0
            end = self.split_point
            self.pre_model1.train()
        elif mode == 'pre-train2':
            start = self.split_point
            end = len(self.train_loader)
            self.pre_model2.train()
        elif mode == 're-train':
            start = 0
            end = len(self.train_loader)
            self.re_model.train()
        
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if not (batch_idx >= start and batch_idx < end):
                continue

            if mode == 'pre-train1':
                self.optimizer1.zero_grad()
            elif mode == 'pre-train2':
                self.optimizer2.zero_grad()
            elif mode == 're-train':
                self.optimizer3.zero_grad()
            
            # input shape: n,l,v,c; graph shape: v,v;
            if mode == 'pre-train1':
                pe_embedd = self.pre_model1(data, self.graph)
                loss = self.pre_model1.loss(pe_embedd, target, self.scaler, mode)

                assert not torch.isnan(loss)
                loss.backward()

                # gradient clipping
                if self.args.grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        get_model_params([self.pre_model1]), 
                        self.args.max_grad_norm)

            elif mode =='pre-train2':
                pe_embedd = self.pre_model2(data, self.graph) 
                loss = self.pre_model2.loss(pe_embedd, target, self.scaler, mode)

                assert not torch.isnan(loss)
                loss.backward()

                # gradient clipping
                if self.args.grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        get_model_params([self.pre_model2]), 
                        self.args.max_grad_norm)
                    
            elif mode == 're-train':
                pe_weights = pe_weights_list[batch_idx]

                pe_embedd = self.re_model(data, self.graph)

                loss = self.re_model.loss(pe_embedd, target, self.scaler, mode, pe_weights)
                assert not torch.isnan(loss)
                loss.backward()

                # gradient clipping
                if self.args.grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        get_model_params([self.re_model]), 
                        self.args.max_grad_norm)
                
            if mode == 'pre-train1':
                self.optimizer1.step()
            elif mode == 'pre-train2':
                self.optimizer2.step()
            elif mode == 're-train':
                self.optimizer3.step()

            total_loss += loss.item()

        if mode == 'pre-train1':
            train_epoch_loss = total_loss/(end - start)
        elif mode == 'pre-train2':
            train_epoch_loss = total_loss/(end - start)
        elif mode == 're-train':
            train_epoch_loss = total_loss/self.train_per_epoch

        self.logger.info('*******Train Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss
    
    def val_epoch(self, epoch, val_dataloader, mode):
        if mode == 'pre-train1':
            self.pre_model1.eval()
        elif mode == 'pre-train2':
            self.pre_model2.eval()
        elif mode == 're-train':
            self.re_model.eval()
        
        
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                if mode == 'pre-train1':
                    pe_embedd = self.pre_model1(data, self.graph)
                    loss = self.pre_model1.loss(pe_embedd, target, self.scaler, mode)
                elif mode == 'pre-train2':
                    pe_embedd = self.pre_model2(data, self.graph)
                    loss = self.pre_model2.loss(pe_embedd, target, self.scaler, mode)
                elif mode == 're-train':
                    pe_embedd = self.re_model(data, self.graph)
                    loss = self.re_model.loss(pe_embedd, target, self.scaler, mode='infer')

                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f}'.format(epoch, val_loss))
        return val_loss

    def pre_train(self, mode):
        best_loss = float('inf')
        best_epoch = 0
        not_improved_count = 0
        start_time = time.time()

        for epoch in range(1, self.args.pretrain_epochs + 1):
            
            train_epoch_loss = self.train_epoch(epoch, mode=mode)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            
            val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader, mode=mode)       
            
            self.pretraining_stats.update((epoch, train_epoch_loss, val_epoch_loss))

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_epoch = epoch
                not_improved_count = 0
                # save the best state
                if mode == 'pre-train1':
                    save_dict = {
                        "epoch": epoch, 
                        "model": self.pre_model1.state_dict(), 
                        "optimizer": self.optimizer1.state_dict(),
                    }
                    self.logger.info('**************Current best model saved to {}'.format(self.best_path1))
                    torch.save(save_dict, self.best_path1)

                    if self.args.pretrain == True:
                        current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                        pretrain_path1 = os.path.join(current_dir, 'experiments', self.args.dataset, 'pretrain', 'best_pre_model1_seed-'+ str(self.args.seed) +'_wd_'+ str(self.args.weight_decay1) +'.pth')
                        torch.save(save_dict, pretrain_path1)

                elif mode == 'pre-train2':
                    save_dict = {
                        "epoch": epoch, 
                        "model": self.pre_model2.state_dict(), 
                        "optimizer": self.optimizer2.state_dict(),
                    }
                
                    self.logger.info('**************Current best model saved to {}'.format(self.best_path2))
                    torch.save(save_dict, self.best_path2)

                    if self.args.pretrain == True:
                        current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                        pretrain_path2 = os.path.join(current_dir, 'experiments', self.args.dataset, 'pretrain', 'best_pre_model2_seed-'+ str(self.args.seed) +'_wd_'+ str(self.args.weight_decay2) +'.pth')
                        torch.save(save_dict, pretrain_path2)
            else:
                not_improved_count += 1

            # early stopping
            if self.args.early_stop and not_improved_count == self.args.pretrain_early_stop_patience:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                "Training stops.".format(self.args.pretrain_early_stop_patience))
                break   

        training_time = time.time() - start_time
        self.logger.info("== Pre-training finished.\n"
                    "Total Pre-training time: {:.2f} min\t"
                    "best loss: {:.4f}\t"
                    "best epoch: {}\t".format(
                        (training_time / 60), 
                        best_loss, 
                        best_epoch))

    def re_train(self, pe_weights_list):
        best_loss = float('inf')
        best_epoch = 0
        not_improved_count = 0
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            
            train_epoch_loss = self.train_epoch(epoch, mode='re-train', pe_weights_list=pe_weights_list)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            
            val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader, mode='re-train')       
            
            self.retraining_stats.update((epoch, train_epoch_loss, val_epoch_loss))

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_epoch = epoch
                not_improved_count = 0
                # save the best state
                save_dict = {
                    "epoch": epoch, 
                    "model": self.re_model.state_dict(), 
                    "optimizer": self.optimizer3.state_dict(),
                }
                
                self.logger.info('**************Current best model saved to {}'.format(self.best_path3))
                torch.save(save_dict, self.best_path3)
            else:
                not_improved_count += 1

            # early stopping
            if self.args.early_stop and not_improved_count == self.args.retrain_early_stop_patience:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                "Training stops.".format(self.args.retrain_early_stop_patience))
                break   

        training_time = time.time() - start_time
        self.logger.info("== Re-training finished.\n"
                    "Total Re-training time: {:.2f} min\t"
                    "best loss: {:.4f}\t"
                    "best epoch: {}\t".format(
                        (training_time / 60), 
                        best_loss, 
                        best_epoch))
        

        # test
        state_dict = torch.load(self.best_path3, map_location=torch.device(self.args.device))
        self.re_model.load_state_dict(state_dict['model'])
        self.logger.info("== Test results.")
        test_results = self.test(self.re_model, self.test_loader, self.scaler, 
                                self.graph, self.logger)
        results = {
            'best_val_loss': best_loss, 
            'best_val_epoch': best_epoch, 
            'test_results': test_results,
        }

        return results

    def normalize(self, x, x_min, x_max):

        x_normalized = (x - x_min) / (x_max - x_min)

        return x_normalized

    def infer_policy(self, mode):
        """"""

        if mode == 'pre-train1':
            current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            pretrain_path1 = os.path.join(current_dir, 'experiments', self.args.dataset, 'pretrain', 'best_pre_model1_seed-'+ str(self.args.seed) +'_wd_'+ str(self.args.weight_decay1) +'.pth')
            if self.args.pretrain == False:
                state_dict = torch.load(pretrain_path1, map_location=torch.device(self.args.device))
                self.pre_model1.load_state_dict(state_dict['model'])

            self.pre_model1.eval()
            enable_dropout(self.pre_model1)

        elif mode == 'pre-train2':
            current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            pretrain_path2 = os.path.join(current_dir, 'experiments', self.args.dataset, 'pretrain', 'best_pre_model2_seed-'+ str(self.args.seed) +'_wd_'+ str(self.args.weight_decay2) +'.pth')
            if self.args.pretrain == False:
                state_dict = torch.load(pretrain_path2, map_location=torch.device(self.args.device))
                self.pre_model2.load_state_dict(state_dict['model'])

            self.pre_model2.eval()
            enable_dropout(self.pre_model2)
        
        with torch.no_grad():
            pe_weights = []

            if mode == 'pre-train1': # use another part to infer
                start = self.split_point 
                end = len(self.train_loader)
            elif mode == 'pre-train2':
                start = 0
                end = self.split_point


            for batch_idx, (data, target) in enumerate(self.train_loader):
                if not (batch_idx >= start and batch_idx < end):
                    continue
                tmp_pe_preds = []
                for i in range(10):
                    if mode == 'pre-train1':
                        pe_embedd = self.pre_model1(data, self.graph)
                        pe_pred = self.pre_model1.pe_predict(pe_embedd)
                    elif mode == 'pre-train2':
                        pe_embedd = self.pre_model2(data, self.graph)
                        pe_pred = self.pre_model2.pe_predict(pe_embedd)

                    tmp_pe_preds.append(pe_pred)

                pe_preds = torch.cat(tmp_pe_preds, dim=1)

                pe_var = torch.var(pe_preds, dim=1, unbiased=False)

                pe_uncert = torch.mean(pe_var, dim=[1, 2], keepdim=False)
                
                if mode == 'pre-train1':
                    phy_consis = torch.mean(self.pre_model1.aleatoric_uncertainty(data, self.graph), dim=[1, 2])
                elif mode == 'pre-train2':    
                    phy_consis = torch.mean(self.pre_model2.aleatoric_uncertainty(data, self.graph), dim=[1, 2])
                
                # compute weights
                pe_weight = self.args.alpha*pe_uncert + self.args.beta*(1/phy_consis)
                pe_weight = F.softmax(pe_weight)

                pe_weights.append(pe_weight)


            return pe_weights

    @staticmethod
    def test(model, dataloader, scaler, graph, logger):
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                pe_embedd = model(data, graph)                
                pred_output = model.predict(pe_embedd)

                y_true.append(target)
                y_pred.append(pred_output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        test_results = []
        # inflow
        mae, mape = test_metrics(y_pred[..., 0], y_true[..., 0])
        logger.info("INFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape*100))
        test_results.append([mae, mape])
        # outflow 
        mae, mape = test_metrics(y_pred[..., 1], y_true[..., 1])
        logger.info("OUTFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape*100))
        test_results.append([mae, mape]) 

        return np.stack(test_results, axis=0)



        

