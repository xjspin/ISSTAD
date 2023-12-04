#coding=utf-8
from os.path import join
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import random
from PIL import ImageFilter, ImageDraw, ImageOps
import cv2


def validate_and_adjust_rectangle(rectangle):
    x0, y0, x1, y1 = rectangle
    if x0 > x1:
        x0, x1 = x1, x0 
    if y0 > y1:
        y0, y1 = y1, y0 
    return (x0, y0, x1, y1)



def image_bonding(oral_img, trans_img, mask):

    assert (oral_img.size == trans_img.size)

    inverse_mask = ImageOps.invert(mask)
    result = Image.new('RGB', oral_img.size)
    result.paste(oral_img, mask=inverse_mask)

    inverse_result = Image.new('RGB', trans_img.size)
    inverse_result.paste(trans_img, mask=mask)

    result.paste(inverse_result, mask=mask)

    return result


def get_mask(label, largest_box, max_num, choice_list=['rectangle', 'ellipse', 'polygon']):

    x, y, x2, y2 = largest_box
    assert (x2 > x)
    assert (y2 > y)

    mask = Image.new('L', label.size, 0)
    draw = ImageDraw.Draw(mask)

    num = random.randint(1,max_num)

    for i in range(num):
        selected_shape = random.choice(choice_list)

        if selected_shape == 'rectangle':
            rect_x = random.randint(x, x2 - 50)  
            rect_y = random.randint(y, y2 - 50) 
            rect_width = random.randint(10, x2 - rect_x)
            rect_height = random.randint(10, y2 - rect_y)
            draw.rectangle((rect_x, rect_y, rect_x + rect_width, rect_y + rect_height), fill=255)

        elif selected_shape == 'ellipse':
            ellipse_x = random.randint(x, x2 - 50)
            ellipse_y = random.randint(y, y2 - 50)
            ellipse_width = random.randint(10, x2 - ellipse_x)
            ellipse_height = random.randint(10, y2 - ellipse_y)
            draw.ellipse((ellipse_x, ellipse_y, ellipse_x + ellipse_width, ellipse_y + ellipse_height), fill=255)

        elif selected_shape == 'polygon':
            num_points = random.randint(3, 8)
            points = [(random.randint(x, x2 - 50), random.randint(y, y2 - 50)) for _ in range(num_points)]
            draw.polygon(points, fill=255)

    return mask

def get_largest_box(img):

    cv_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        largest_box = cv2.boundingRect(max_contour)
        
    x, y, box_width, box_height = largest_box

    height, width, _ = cv_img.shape
    padding_x = box_width//10
    padding_y = box_height//5
    x = max(0, x - padding_x)  
    y = max(0, y - padding_y)
    box_width = min(width - x, box_width + 2 * padding_x)  
    box_height = min(height - y, box_height + 2 * padding_y)  

    x2, y2 = x + box_width, y + box_height

    return [x, y, x2, y2]


def remove_black(image):

    gray_image = image.convert('L')

    threshold = np.array(gray_image).mean()  
    threshold = gray_image.point(lambda x: 255 if x > threshold else 0, '1')

    mask = Image.new('L', image.size, 0)

    mask.paste(threshold)

    result_image = Image.new('RGB', image.size)
    result_image.paste(image, mask=mask)


    image = result_image.convert('RGBA')
    data = np.array(image)
    black_pixels = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0)
    data[black_pixels, 3] = 0 
    non_black_pixels = ~black_pixels
    average_color = tuple(np.mean(data[non_black_pixels][:, :3], axis=0).astype(int))

    new_image = Image.new('RGB', image.size, average_color)

    inverse_mask = ImageOps.invert(mask)
    inverse_image = Image.new('RGB', image.size)
    inverse_image .paste(new_image, mask=inverse_mask)
    result = result_image
    result.paste(inverse_image, mask=inverse_mask)

    return result


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class ImageTrans(object):

    def __init__(self):
        
        self.color = transforms.Compose([
            transforms.RandomChoice([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.4),
                transforms.RandomGrayscale(p=1),
                GaussianBlur([15.0, 20.0]),
                transforms.Compose([transforms.ColorJitter(0.8, 0.8, 0.8, 0.4),transforms.RandomGrayscale(p=1)]),
                transforms.Compose([transforms.ColorJitter(0.8, 0.8, 0.8, 0.4),GaussianBlur([15.0, 20.0])]),
                transforms.Compose([transforms.RandomGrayscale(p=1),GaussianBlur([15.0, 20.0])]),
                ])
        ])       

        self.flip = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)])])
        ])  


    def __call__(self, img, choice_list=['color','flip', 'noise']):
        choice = random.choice(choice_list)
        if 'color' == choice:
            img = self.color(img)
        elif 'flip' == choice:
            img = self.flip(img)
        elif 'noise' == choice:
            noise = np.random.randint(0, 256, (img.size[1], img.size[0], 3), dtype=np.uint8)
            noise_image = Image.fromarray(noise)
            img = Image.blend(img, noise_image, alpha=0.5)  
        return img
    


class ImageAugmentation:
    def __init__(self, object_name='breakfast_box'):


        if 'capsule' == object_name:

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False  
            self.mask_num = 3
            self.area_limitation = True
        
        elif 'screw' == object_name:

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3
            self.area_limitation = True

        elif 'pill' == object_name:

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3
            self.area_limitation = True

        elif 'cable' == object_name:

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3
            self.area_limitation = True

        elif 'carpet' == object_name: 

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3 
            self.area_limitation = True 

        elif 'metal_nut' == object_name: 

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3
            self.area_limitation = True

        elif 'bottle' == object_name:

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3
            self.area_limitation = True

        elif 'grid' == object_name:

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3
            self.area_limitation = True

        elif 'hazelnut' == object_name:

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3
            self.area_limitation = True

        elif 'leather' == object_name: 

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3
            self.area_limitation = True  

        elif 'tile' == object_name: 

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3
            self.area_limitation = True

        elif 'toothbrush' == object_name: 

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3
            self.area_limitation = True

        elif 'transistor' == object_name: 

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3
            self.area_limitation = True

        elif 'wood' == object_name: 

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = False
            self.mask_num = 3
            self.area_limitation = True

        elif 'zipper' == object_name: 

            self.trans_list =  ['color','flip', 'noise']
            self.mask_list = ['rectangle', 'ellipse', 'polygon']
            self.black_removed = True
            self.mask_num = 3
            self.area_limitation = False
    

    def __call__(self, image, label):

        trans = ImageTrans()
        
        if self.area_limitation:
            largest_box = get_largest_box(image)
        else:
            largest_box = [0, 0, image.size[0], image.size[1]]

        if self.black_removed:
            img = remove_black(image)
            img_trans = trans(img, choice_list=self.trans_list)
        else:
            img_trans = trans(image, choice_list=self.trans_list)

        mask = get_mask(label, largest_box, max_num=self.mask_num, choice_list=self.mask_list)

        img_result = image_bonding(image, img_trans, mask)

        return img_result, mask
    

