import os 
import json
import numpy as np
import tensorflow as tf


class ModelCheckPoint:

    def __init__(self, path, val='ep', k=None):
        self.model_save_dir = os.path.join(path, 'models')
        os.mkdir(self.model_save_dir)
        self.val = val  
        self.k = k
        self.min_loss = []
        self.ep_list = []

    def check_min_loss(self, loss):
        if loss < self.min_loss[-1]:
            return True
        else:
            return False

    def save_data(self):
        save_path = self.model_save_dir + '/' + 'chpk_' + '.json'
        data = {'loss': self.min_loss, 'epoches': self.ep_list}
        with open(save_path, "w", encoding="utf8") as write_file:
            json.dump(data, write_file, ensure_ascii=False)

    def check(self, model, loss, epoch):
        if epoch < 5:            
            self.min_loss.append(float(loss))
            self.ep_list.append(epoch)
        else:
            if self.check_min_loss(loss):
                if self.k:
                    model_save_path = self.model_save_dir + '/' + 'm32' + '_' + self.val + str(epoch) + '_k' + str(self.k) + '.h5'
                else:
                    model_save_path = self.model_save_dir + '/' + 'm64' + '_' + self.val + str(epoch) + '.h5'
                model.save(model_save_path)
                self.min_loss.append(float(loss))
                self.ep_list.append(epoch)
                print('=' * 50)
                print(f'Модель сохранена на {epoch} эпохе на {loss} значении функции ошибки')
                print('=' * 50)

    def check2(self, model, loss, epoch):
        if epoch < 5:            
            self.min_loss.append(float(loss))
            self.ep_list.append(epoch)
        else:
            if self.check_min_loss(loss):
                model_save_path = self.model_save_dir + '/' + str(epoch) + '/'
                tf.saved_model.save(model, model_save_path)
                self.min_loss.append(float(loss))
                self.ep_list.append(epoch)
                print(f'Модель сохранена на {epoch} эпохе')

class History:

    def __init__(self, main_train_dir_path, digit, val='ep', k=None):
        self.history_dir_path = os.path.join(main_train_dir_path, 'history')
        os.mkdir(self.history_dir_path)
        self.digit = digit
        self.val = val
        self.k = k
        self.train_loss = []
        self.train_rmse = []
        self.train_acc = []
        self.train_auc = []
        self.val_loss = []
        self.val_rmse = []
        self.val_acc = []
        self.val_auc = []

    def write_train_history(self, epoch_train_loss, train_metrics):
        self.train_loss.append(float(epoch_train_loss))
        self.train_rmse.append(float(train_metrics[0]))
        self.train_acc.append(float(train_metrics[1]))
        self.train_auc.append(float(train_metrics[2]))
    
    def write_val_history(self, epoch_val_loss, val_metrics):
        self.val_loss.append(float(epoch_val_loss))
        self.val_rmse.append(float(val_metrics[0]))
        self.val_acc.append(float(val_metrics[1]))
        self.val_auc.append(float(val_metrics[2]))

    def write(self, train_data, val_data=None):
        self.write_train_history(train_data[0], train_data[1])
        if val_data:
            self.write_val_history(val_data[0], val_data[1])

    def save_to_json(self):
        if self.k:
            save_path = self.history_dir_path + '/' + 'history_' + self.val + str(self.digit)+ '_k' + str(self.k) + '.json'
        else:
            save_path = self.history_dir_path + '/' + 'history_' + self.val + str(self.digit) + '.json'
        data = {'train': {'loss': self.train_loss, 'rmse': self.train_rmse, 'acc': self.train_acc, 'auc': self.train_auc},
                'val': {'loss': self.val_loss, 'rmse': self.val_rmse, 'acc': self.val_acc, 'auc': self.val_auc}}
        with open(save_path, "w", encoding="utf8") as write_file:
            json.dump(data, write_file, ensure_ascii=False)