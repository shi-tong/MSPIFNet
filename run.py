import argparse
import os
import torch
from exp.exp_classification import Exp_Classification
import random
import numpy as np                                                                                                       

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fault Diagnosis')
    parser.add_argument('--seed', type=int, default=2026)
    # Task settings
    parser.add_argument('--model', type=str, default='MSPIFNet')
    parser.add_argument('--data_set', type=str, default='Ottawa', help='options: CWRU, Ottawa')
    
    # ------------------>Experimental setting & Running
    parser.add_argument('--running_type', type=str, default='joint_test', choices=["clean_test", "noise_test", "small_test" ,"joint_test"],
                        help = "clean_test: Use all data\
                                noise_test: Add Gaussian white noise to all data\
                                small_test: Select a small sample from all data\
                                joint_test: Considering the combined small sample and noise conditions")
    
    # running status
    parser.add_argument('--is_running', type=bool, default = True)
    # Folder setting
    parser.add_argument('--root_folder', type=str, default="../")
    
    # ------------------>Experimental setting & Conditions
    # To noise options
    parser.add_argument('--noise_db', type=int, default=-6, choices = [-6, -4, -2, 0], help = "Select noise intensity")
    # To small sample options
    parser.add_argument('--small_level', type=str, default='L4', choices = ['L1', 'L2', 'L3', 'L4'], 
                        help = "L1 for 1 sample, L2 for 2 samples, ..., L4 for 4 samples")
    # To joint options
    parser.add_argument('--joint_noise_db', type=int, default=-2, choices = [-6, -4, -2, 0])
    parser.add_argument('--joint_sample_level', type=str, default='L1', choices = ['L1', 'L2', 'L3', 'L4'])
    
    # Loss-Accuracy save path
    parser.add_argument('--acc_loss', type=str, default='.../', 
                        help='path of saving accuracy and loss to help visualization')
    parser.add_argument('--is_save', type=bool, default=False,
                        help='whether or not to save accuracy and loss to txt')
    # data position
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='loscation of model checkpoints')
    
    # ------------------>Experimental setting & Dataloader
    parser.add_argument('--seq_len', type=int, default=512, help='input sequence length')
    parser.add_argument('--num_class', type=int, default=-1, help='10 for CWRU && 9 for Ottawa')
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=10, help='repeat experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate(0.000035, 0.0000095)')

    # ------------------>Experimental setting & GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    print('Args in experiment:')
    # print_args(args)
    print(f"<-------------------------------Runing Type: Training-------------------------------->")
    print(f"<------------------------Runing Style: {args.running_type}--------------------------->")
    print(f"<----------------------------Runing Model: {args.model}------------------------------>")

    from decimal import Decimal
    lr_str = f"{Decimal(str(args.learning_rate)):f}"

    if args.is_running:
        if args.itr > 1:
            print(f"<----------------------------Repeat Times: {args.itr}------------------------------>")
        else:
            print(f"<----------------------------------Repeat Once------------------------------------>")

        all_acc = []
        all_loss = []

        for ii in range(args.itr):
            print(f"Current: {ii + 1}/{args.itr}, Progress: {((ii + 1) / args.itr) * 100:.2f}%")
            root_folder = os.path.join(args.root_folder, args.data_set, 'Experiment', f'repeat_{ii:02d}')
            if not os.path.exists(root_folder):
                raise ValueError(f"Data Root folder: {root_folder} doesn't exist")

            print(f"Load data from Path: {root_folder}")
            set_random_seed(args.seed + ii)
            exp = Exp_Classification(args)
            exp.initial_dataset(folder=root_folder)

            setting = '{}_{}_{}_{}_{}_{}_{}_{}_repeat{:02d}'.format(
                args.data_set,
                args.num_class,
                args.model,
                args.seq_len,
                args.running_type,
                args.train_epochs,
                args.batch_size,
                lr_str,
                ii
            )

            # show_msg = args.itr == 1

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting, show_meg=False)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            test_loss, test_acc = exp.test(setting, show_msg=False)

            all_loss.append(float(test_loss))
            all_acc.append(float(test_acc))

            del exp
            torch.cuda.empty_cache()
            print('\n')

        print("-------------------------->Final Accuracy<--------------------------")
        print(",".join(f"{x:.4f}" for x in all_acc))

        mean_acc = np.array(all_acc).mean()
        std_acc = np.array(all_acc).std(ddof=1) if len(all_acc) > 1 else 0.0

        mean_loss = np.array(all_loss).mean()
        std_loss = np.array(all_loss).std(ddof=1) if len(all_loss) > 1 else 0.0

        print(f"Loss: {mean_loss:.4f}±{std_loss:.4f}")
        print(f"Accuracy: {mean_acc:.4f}±{std_acc:.4f}")
    else:
        root_folder = os.path.join(args.root_folder, args.data_set, 'Experiment', f'repeat_{0:02d}')
        if not os.path.exists(root_folder):
            raise ValueError(f"Data Root folder: {root_folder} doesn't exist")

        print(f"Load data from Path: {root_folder}")
        set_random_seed(args.seed)
        exp = Exp_Classification(args)
        exp.initial_dataset(folder=root_folder)
        setting = '{}_{}_{}_{}_{}_{}_{}_{}_repeat{:02d}'.format(
            args.data_set,
            args.num_class,
            args.model,
            args.seq_len,
            args.running_type,
            args.train_epochs,
            args.batch_size,
            lr_str,
            0
        )
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        test_loss, test_acc = exp.test(setting, show_msg=False)
        print("Test Result:", test_acc)

