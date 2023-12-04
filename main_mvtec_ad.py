# coding=utf-8
import subprocess
import os
import shutil
from PIL import Image


if __name__ == '__main__':
    
    objects_list = ['bottle', 'cable', 'capsule',  'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    data_dir = './data/mvtecAD/'

    for object_name in objects_list:

        print(object_name)

        train_data_dir = data_dir + object_name + '/train'  
        test_data_dir = data_dir + object_name + '/test'
        test_label_dir = data_dir + object_name + '/ground_truth'

            

        #step1 sub-thread

        step1_saved_models_dir = './saved_models/step1_saved_models/mvtecAD/'
        python_script = './step1_mae_pretrain/main_mvtec_ad_step1.py' 
        process = subprocess.Popen(['python', python_script, 
                                    '--object_name', object_name,
                                    '--data_path', train_data_dir,
                                    '--output_dir', step1_saved_models_dir])
        process.wait()   


        #step2 sub-thread
        python_script = './step2_pixel_ss_learning/main_mvtec_ad_step2.py' 
        process = subprocess.Popen(['python', python_script, 
                                    '--object_name', object_name, 
                                    '--train_data_dir', train_data_dir, 
                                    '--test_data_dir', test_data_dir,
                                    '--test_label_dir', test_label_dir])
        process.wait()   
