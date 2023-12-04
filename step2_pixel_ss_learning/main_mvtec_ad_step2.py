# coding=utf-8

from mimetypes import guess_all_extensions
import os
import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from scipy.ndimage import gaussian_filter

import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from sklearn.metrics import roc_auc_score
from PIL import Image
import subprocess

from util.losses import cross_entropy
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.simm import SSIM
from util.mtvecad import AD_TEST, AD_TRAIN
from util.mvtecad_options import parser
from model import models

def setdir(file_path):
    if not os.path.exists(file_path):  
        os.makedirs(file_path)

if __name__ == '__main__':
    
    args = parser.parse_args()
    object_name = args.object_name

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # set paths
    step1_saved_model = args.step1_saved_models_dir + object_name + '/checkpoint-0.75.pth'
    auc_saved_path = args.auc_saved_dir + object_name + '/'
    if auc_saved_path:
        setdir(auc_saved_path)
    spro_saved_path = args.spro_saved_dir + object_name + '/'
    if spro_saved_path:
        setdir(spro_saved_path)
    log_saved_path = args.log_saved_dir + object_name + '/'
    if log_saved_path:
        setdir(log_saved_path)
    model_saved_path = args.saved_models_dir + object_name + '/'
    if model_saved_path:
        setdir(model_saved_path)

    min_loss = 0

    # define model
    model_ad = getattr(models, args.arch)()

    checkpoint = torch.load(step1_saved_model, map_location='cpu')
    msg = model_ad.load_state_dict(checkpoint['model'], strict=False)

    model_ad = model_ad.to(device)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ad = torch.nn.DataParallel(model_ad, device_ids=range(torch.cuda.device_count()))

    param_groups = optim_factory.add_weight_decay(model_ad, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    weight = torch.tensor([3.0, 1.0]).to(device, dtype=torch.float)
    ce_loss = cross_entropy(weight=weight).to(device, dtype=torch.float)
    ssim_loss = SSIM()

    loss_scaler = NativeScaler()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr)
    best_auc = 0
    best_pixel_auc = 0


    test_set = AD_TEST(args, args.test_data_dir, args.test_label_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    # training
    for epoch in range(1, args.num_epochs+1):

        # load data
        train_set = AD_TRAIN(args.train_data_dir)
        train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batchsize, shuffle=True)
        
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'loss': 0 }

        model_ad.train(True)
        optimizer.zero_grad()
        for img, label in train_bar:
            running_results['batch_sizes'] += args.batchsize

            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)
            label = torch.argmax(label, 1).unsqueeze(1).float()

            rimg, mask_list = model_ad(img)
            loss_m, loss_fc = 0, 0  

            for mask in mask_list:
                loss_m = loss_m + ce_loss(mask, label)
            loss_m = loss_m/len(mask_list)
            loss_r = ((rimg - img)**2).mean()
            loss_s = 1 - ssim_loss(rimg,img)       

            loss = loss_m + 0.5*loss_r + 0.1*loss_s

            accum_iter = args.accum_iter
            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model_ad.parameters())

            torch.cuda.synchronize()

            running_results['loss'] += loss.item() * args.batchsize

            train_bar.set_description(
                desc='[%d/%d] loss: %.4f' % (
                    epoch, args.num_epochs,
                    running_results['loss'] / running_results['batch_sizes'],))
        scheduler.step()

        torch.save(model_ad.state_dict(),  model_saved_path + 'net_ad.pth')

        # testing
        model_ad.eval()
        with torch.no_grad():
            val_bar = tqdm(test_loader)
            inter, unin = 0,0
            last_inter, last_unin = 0,0
            valing_results = {'batch_sizes': 0}

            m = 0
            pn_mean_list = []
            mean_list = []
            pn_pixel_list = []
            pixel_list = []               

            for img, pn, label, tiff_path in val_bar:
                valing_results['batch_sizes'] += args.val_batchsize

                img = img.to(device, dtype=torch.float)

                rimg, mask_list = model_ad(img)

                pred_softmax_mean = 0
                for mask in mask_list:
                    pred_softmax_1 = torch.softmax(mask.permute(0,2,3,1),dim=-1)[:,:,:,1]
                    pred_softmax_mean = pred_softmax_mean + pred_softmax_1

                pred_softmax_mean = (pred_softmax_mean/len(mask_list))

                anomaly_map = gaussian_filter(pred_softmax_mean.squeeze(0).cpu(), sigma=3)
                
                m_admap = pred_softmax_mean
                r_admap = ((rimg - img)**2).mean(dim=1)
                pmap = m_admap
                rmap = r_admap
                m_admap = gaussian_filter(m_admap.squeeze(0).cpu(), sigma=4)
                r_admap = gaussian_filter(r_admap.squeeze(0).cpu(), sigma=4)
                admap = gaussian_filter(m_admap * r_admap, sigma=4)


                min_value = np.min(rmap.squeeze(0).cpu().numpy())
                max_value = np.max(rmap.squeeze(0).cpu().numpy())
                rmap = (rmap - min_value) / (max_value - min_value)
                mixmap =  (pmap * rmap)
                mixmap = mixmap*255.0
                mixmap = gaussian_filter(mixmap.squeeze(0).cpu(), sigma=4)
                
                
                image = Image.fromarray(mixmap)
                tif_save_path = folder_path = os.path.dirname(tiff_path[0])
                if tif_save_path:
                    setdir(tif_save_path)
                image.save(tiff_path[0], 'TIFF')

                mean_list.append(np.mean(admap))
                pn_mean_list.append(pn[0])

                pn_pixel_list.extend(label.squeeze(0).squeeze(0).cpu().numpy().astype(int).ravel())
                pixel_list.extend(mixmap.ravel())
                        
                                       



            now = datetime.datetime.now()
            now = now.strftime('%y%m%d%H')
            f = open( log_saved_path + now + 'log.txt','a')
            print(
                "image_auc", roc_auc_score(pn_mean_list, mean_list),  
                "pixel_auc", roc_auc_score(pn_pixel_list, pixel_list), file=f)
            f.close()


            if best_auc < roc_auc_score(pn_mean_list, mean_list):
                best_auc = roc_auc_score(pn_mean_list, mean_list)               
                f = open( auc_saved_path + 'image_level_result.txt','w')
                best_auc_ = best_auc
                best_auc_ = "{:.2f}".format(best_auc_*100)
                print("image_auc", best_auc_, file=f)
                f.close()   

            if best_pixel_auc < roc_auc_score(pn_pixel_list, pixel_list):
                best_pixel_auc = roc_auc_score(pn_pixel_list, pixel_list)               
                f = open( auc_saved_path + 'pixel_level_result.txt','w')
                best_pixel_auc_ = best_pixel_auc
                best_pixel_auc_ = "{:.2f}".format(best_pixel_auc_*100)
                print("pixel_auc", best_pixel_auc_, file=f)
                f.close() 

