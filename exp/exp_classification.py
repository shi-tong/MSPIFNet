from exp.exp_basic import Exp_Basic
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from exp.loss_acc_tool import create_file, write_acc_loss
from torch.utils.data import Dataset, DataLoader
from typing import Optional
warnings.filterwarnings('ignore')

class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.need_visual = (
            args.is_visual_TSNE or
            args.is_visual_ConfMatrix or
            args.is_visual_Atten_Map
        )
        if args.is_save:
            all_acc_loss_path = create_file(args)
            print(all_acc_loss_path)
            self.train_acc_path, self.valid_acc_path, self.test_acc_path,\
            self.train_loss_path, self.valid_loss_path, self.test_loss_path = all_acc_loss_path
    
    def initial_dataset(self, folder):
        if self.args.data_set == 'Ottawa':
            self.X_train, self.X_test, self.X_valid = self._get_data_for_experiment(root_folder=folder)
        elif self.args.data_set == 'CWRU':
            self.X_train, self.X_test, self.X_valid = self._get_data_for_experiment(root_folder=folder)
        else:
            raise ValueError("please check the data_set name")
    
    def _get_data_for_experiment(self, root_folder):
        if self.args.running_type == "clean_test":
            train, test, valid = self._selective_load_data(folder_name=root_folder)
        elif self.args.running_type == "noise_test":
            train, test, valid = self._selective_load_data(snr = self.args.noise_db, folder_name=root_folder)
        elif self.args.running_type == "small_test":
            train, test, valid = self._selective_load_data(small_sample = self.args.small_level, folder_name=root_folder)
        elif self.args.running_type == "joint_test":
            train, test, valid = self._selective_load_data(snr = self.args.joint_noise_db, small_sample = self.args.joint_sample_level, folder_name=root_folder)
        else:
            raise TypeError(f"No such running type: {self.args.running_type}")
        return train, test, valid
    
    def _load_data_from_folder(self, data_folder):
        # data
        train_data = np.load(os.path.join(data_folder, "train.npy")).astype(np.float32)
        test_data = np.load(os.path.join(data_folder, "test.npy")).astype(np.float32)
        valid_data = np.load(os.path.join(data_folder, "valid.npy")).astype(np.float32)
        # label
        train_label = np.load(os.path.join(data_folder, "train_label.npy")).astype(np.int64)
        test_label = np.load(os.path.join(data_folder, "test_label.npy")).astype(np.int64)
        valid_label = np.load(os.path.join(data_folder, "valid_label.npy")).astype(np.int64)

        return (train_data, train_label), (test_data, test_label), (valid_data, valid_label)
    
    def _selective_load_data(self, 
                             snr:Optional[int] = None,
                             snr_list:list = [-6, -4, -2, 0],
                             sample_list:list = ['L1', 'L2', 'L3', 'L4'],
                             small_sample:Optional[str] = None, 
                             folder_name = None):
        if not os.path.exists(folder_name):
            print(f"Folder: {folder_name} isn't a folder path")
        # Load noise data
        if snr is not None:
            if small_sample is not None:
                if snr not in snr_list and small_sample not in sample_list:
                    raise ValueError(f"error in snr = {snr} and small_sample = {small_sample}")
                # folder = small_sample_noise
                if snr < 0:
                    snr_name = f"snr_m{abs(snr)}"
                else:
                    snr_name = f"snr_{snr}"
                small_sample_name = f"{small_sample}_shot{small_sample[-1]}"
                small_sample_noise_data_folder = os.path.join(folder_name, "small_sample_noise", snr_name, small_sample_name)
                train, test, valid = self._load_data_from_folder(small_sample_noise_data_folder)
            else:
                if snr not in snr_list:
                    raise ValueError(f"snr should be in {snr_list}, but get snr = {snr}")
                if snr < 0:
                    snr_name = f"snr_m{abs(snr)}"
                else:
                    snr_name = f"snr_{snr}"
                snr_data_folder = os.path.join(folder_name, 'noise', snr_name)
                train, test, valid = self._load_data_from_folder(snr_data_folder)
        else: # snr = None
            # Load small sample data
            if small_sample is not None:
                if small_sample not in sample_list:
                    raise ValueError(f"smaple_sample should be in {sample_list}, but get {small_sample}")
                small_sample_name = f"{small_sample}_shot{small_sample[-1]}"
                small_sample_data_folder = os.path.join(folder_name, 'small_sample', small_sample_name)
                train, test, valid = self._load_data_from_folder(small_sample_data_folder)
            else: # snr = None, small_sample = None ---> clean data
                clean_data_folder = os.path.join(folder_name, "clean_full")
                train, test, valid = self._load_data_from_folder(clean_data_folder)
        
        return train, test, valid
    
    def _build_model(self):
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model
    
    def _get_data(self, flag):
        # create DataLoader
        if flag == 'train':
            data_set = Data(self.X_train[0], self.X_train[1])
            data_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        elif flag == 'test':
            data_set = Data(self.X_test[0], self.X_test[1])
            data_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        elif flag == 'valid':
            data_set = Data(self.X_valid[0], self.X_valid[1])
            data_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_scheduler(self, model_optim):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs)
        return scheduler
    
    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion
    
    def _get_loaders(self):
        _, train_loader = self._get_data(flag='train')
        _, valid_loader = self._get_data(flag='valid')
        _, test_loader = self._get_data(flag='test')
        return {
            "train": train_loader,
            "valid": valid_loader,
            "test": test_loader
        }

    def _forward_model(self, batch_x):
        if self.need_visual:
            outputs = self.model(batch_x)

            if isinstance(outputs, tuple):
                output = outputs[0]
                features_for_visualization = outputs[1]
                atten_outputs = outputs[-1]

            return output, features_for_visualization, atten_outputs
        else:
            outputs = self.model(batch_x)
            return outputs

    def _calculate_metrics(self, pred_list, true_list):
        pred_res = torch.cat(pred_list, dim=0)
        true_res = torch.cat(true_list, dim=0)
        predictions = torch.argmax(pred_res, dim=1).cpu().numpy()
        trues = true_res.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        return accuracy


    def _train_one_epoch(self, train_loader, criterion, optimizer, epoch):
        self.model.train()
        total_loss = 0.0
        pred_list = []
        true_list = []
        iter_count = 0
        time_now = time.time()
        train_steps = len(train_loader)
        for i, (batch_x, label) in enumerate(train_loader):
            iter_count += 1
            batch_x = batch_x.float().to(self.device)
            label = label.long().to(self.device)
            optimizer.zero_grad()
            outputs = self._forward_model(batch_x)
            loss = criterion(outputs, label)
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            optimizer.step()
            total_loss += loss.item()
            pred_list.append(outputs.detach())
            true_list.append(label.detach())
            if (i + 1) % 100 == 0:
                print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()
                    )
                )

                speed = (time.time() - time_now) / iter_count
                left_time = speed * (
                    (self.args.train_epochs - epoch - 1) * train_steps
                    + (train_steps - i - 1)
                )

                print(
                    "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                        speed, left_time
                    )
                )

                iter_count = 0
                time_now = time.time()

        avg_loss = total_loss / len(train_loader)
        accuracy = self._calculate_metrics(pred_list, true_list)

        return avg_loss, accuracy
    
    def _evaluate(self, data_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        pred_list = []
        true_list = []
        with torch.no_grad():
            for batch_x, label in data_loader:
                batch_x = batch_x.float().to(self.device)
                label = label.long().to(self.device)
                outputs = self._forward_model(batch_x)
                loss = criterion(outputs, label)
                total_loss += loss.item()
                pred_list.append(outputs.detach())
                true_list.append(label.detach())

        avg_loss = total_loss / len(data_loader)
        accuracy = self._calculate_metrics(pred_list, true_list)
        return avg_loss, accuracy
        
    def train(self, setting, show_meg = True):
        self.loaders = self._get_loaders()

        train_loader = self.loaders["train"]
        valid_loader = self.loaders["valid"]

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(model_optim)

        best_valid_loss = float("inf")
        best_model_path = os.path.join(path, "checkpoint.pth")

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()

            train_loss, train_acc = self._train_one_epoch(
                train_loader=train_loader,
                criterion=criterion,
                optimizer=model_optim,
                epoch=epoch
            )

            valid_loss, valid_acc = self._evaluate(
                data_loader=valid_loader,
                criterion=criterion
            )

            scheduler.step(valid_loss)

            if show_meg:
                print("Epoch: {} cost time: {:.4f}s".format(
                    epoch + 1, time.time() - epoch_time
                ))

                print(
                    "Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}".format(
                        epoch + 1,
                        self.args.train_epochs,
                        train_loss,
                        train_acc
                    )
                )

                print(
                    "Validation Loss: {:.4f}, Validation Accuracy: {:.4f}".format(
                        valid_loss,
                        valid_acc
                    )
                )

            if self.args.is_save:
                write_acc_loss(train_loss, self.train_loss_path)
                write_acc_loss(train_acc, self.train_acc_path)
                write_acc_loss(valid_loss, self.valid_loss_path)
                write_acc_loss(valid_acc, self.valid_acc_path)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), best_model_path)
                if show_meg:
                    print("Best model saved at epoch {}".format(epoch + 1))

        return self.model

    def test(self, setting, show_msg = True, visual_TSNE = True, visual_ConfMatrix = False, visual_atten_map = False):
        self.loaders = self._get_loaders()
        test_loader = self.loaders["test"]
        if show_msg:
            print('loading model')
            print(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        test_loss, test_acc = self._evaluate(
            data_loader=test_loader,
            criterion=self._select_criterion()
        )
        if show_msg:
            print(
                "Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(
                    test_loss,
                    test_acc
                )
            )

        if self.args.is_save:
            write_acc_loss(test_loss, self.test_loss_path)
            write_acc_loss(test_acc, self.test_acc_path)
        
        return test_loss, test_acc