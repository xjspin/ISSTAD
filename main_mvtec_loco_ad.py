# coding=utf-8
import subprocess
import os
import shutil
from PIL import Image

def resize_images(folder_a, folder_b, target_size=(224, 224)):
    if not os.path.exists(folder_b):
        os.makedirs(folder_b)

    for root, dirs, files in os.walk(folder_a):

        relative_path = os.path.relpath(root, folder_a)
        folder_b_path = os.path.join(folder_b, relative_path)

        if not os.path.exists(folder_b_path):
            os.makedirs(folder_b_path)

        for file in files:
            file_path_a = os.path.join(root, file)
            file_path_b = os.path.join(folder_b_path, file)

            if os.path.exists(file_path_b):
                continue

            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img = Image.open(file_path_a)
                resized_image = img.resize(target_size, Image.NEAREST)
                resized_image.save(file_path_b)
            else:
                if not os.path.exists(file_path_b):
                    shutil.copy2(file_path_a, file_path_b)


if __name__ == '__main__':
    
    objects_list = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']

    data_dir = './data/mvteclocoAD/'
    resize_data_dir = './resize_data/mvteclocoAD/'
    resize_images(data_dir, resize_data_dir) 

    for object_name in objects_list:

        print(object_name)

        train_data_dir = data_dir + object_name + '/train'  
        test_data_dir = data_dir + object_name + '/test'
            

        #step1 sub-thread
        step1_saved_models_dir = './saved_models/step1_saved_models/mvteclocoAD/'
        python_script = './step1_mae_pretrain/main_mvtec_loco_ad_step1.py' 
        process = subprocess.Popen(['python', python_script, 
                                    '--object_name', object_name,
                                    '--data_path', train_data_dir,
                                    '--output_dir', step1_saved_models_dir ])
        process.wait()   


        #step2 sub-thread
        python_script = './step2_pixel_ss_learning/main_mvtec_loco_ad_step2.py' 
        process = subprocess.Popen(['python', python_script, 
                                    '--object_name', object_name, 
                                    '--train_data_dir', train_data_dir, 
                                    '--test_data_dir', test_data_dir,
                                    '--resize_data_dir', resize_data_dir,])
        process.wait()   
