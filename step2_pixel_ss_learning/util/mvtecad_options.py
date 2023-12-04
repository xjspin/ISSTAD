import argparse

#training options
parser = argparse.ArgumentParser(description='ISSTAD')

# training parameters
parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=1, type=int, help='batchsize for validation')
parser.add_argument('--num_workers', default=8, type=int, help='num of workers')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')
parser.add_argument('--img_size', default=224, type=int, help='imagesize')
parser.add_argument('--seed', default=111, type=int)
parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
parser.add_argument('--object_name', default='breakfast_box', type=str)
parser.add_argument('--arch', default='mae_vit_large_patch16', type=str)


# Optimizer parameters
parser.add_argument('--accum_iter', default=1, type=int,
                    help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--min_lr', type=float, default=1e-6, 
                    help='min_learning rate')
parser.add_argument('--T_max', type=int, default=30,
                    help='total number of steps in a training cycle')


# path for loading data from folder
parser.add_argument('--data_dir', default='./data/mvtecAD/', type=str)
parser.add_argument('--resize_data_dir', default='./resize_data/mvtecAD/', type=str)
parser.add_argument('--train_data_dir', default='./data/mvtecAD/breakfast_box/train/good', type=str, help='image in training set')
parser.add_argument('--test_data_dir', default='./data/mvtecAD/breakfast_box/test', type=str, help='image in testing set')
parser.add_argument('--test_label_dir', default='./data/mvtecAD/breakfast_box/ground_truth', type=str)

# pre_train_model loading
parser.add_argument('--step1_saved_models_dir', default='./saved_models/step1_saved_models/mvtecAD/', type=str, help='pre-trained modell with step1')

# network saving
parser.add_argument('--saved_models_dir', default='./saved_models/step2_saved_models/mvtecAD/', type=str, help='model save path')

# path for result save
parser.add_argument('--log_saved_dir', default='./step2_pixel_ss_learning/logs/mvtecAD/', type=str)
parser.add_argument('--auc_saved_dir', default='./result/auc/mvtecAD/', type=str)
parser.add_argument('--admap_saved_dir', default='./result/admap/mvtecAD/', type=str)
parser.add_argument('--spro_saved_dir', default='./result/spro/mvtecAD/', type=str)





