
"""
python3 cam.py --AR --h --name 0214_1641/AR_Anyang_high_lr001_b128 --batchsize 128 --data_dir "./dataset/Anyang_data/pytorch" --which_epoch "last" --cat "full_category"
"""

from model import AR
from gradcampp.gradcam import GradCAM, GradCAMpp
from gradcampp.utils import visualize_cam, Normalize, UnNormalize
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.utils import make_grid, save_image
from torchvision import datasets, transforms
import os
import argparse
import time
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
SAVE_ROUTE = "gradcam"

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--AR', action='store_true', help='use attribute recognition')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='training dir path')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--h', action='store_true', help='use high resolution dataset')
parser.add_argument('--cat', default='full', type=str, help='Select the categories')
opt = parser.parse_args()

def load_network(network):                                   

    network.load_state_dict(torch.load(os.path.join('./model',opt.name,'net_%s.pth'%opt.which_epoch)))                                 
                         
    return network      


def convert_ar_label_Anyang(att_keys):
    converted_labels = []
    for att in att_keys:
        tmp1 = att.split('_')
        ar_label = list(map(int,tmp1))
        converted_labels.append(ar_label)

    return converted_labels

def load_data():
    if opt.h:
        data_high = '_high'

    transform_test_list = [
    transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    image_datasets = {}
    image_datasets['test'] = datasets.ImageFolder(os.path.join(opt.data_dir, 'test' + data_high),
                                         transforms.Compose(transform_test_list))
    
    test_att_keys = list(image_datasets['test'].class_to_idx.keys())
    all_ar_labels = convert_ar_label_Anyang(test_att_keys)
    all_ar_labels = torch.LongTensor(all_ar_labels)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=8, pin_memory=True) # 8 workers may work faster
              for x in ['test']}

    return dataloaders, all_ar_labels


def load_category(cat_type):
    if cat_type == 'full_category':
        age_list = ['young', 'adult', 'old']
        gender_list = ['female', 'male']                    
        hair_type_list = ['hair type short', 'hair type long', 'hair type bald']
        top_type_list = ['short sleeve', 'long sleeve'] 
        top_color_list = ['top color black', 'top color blue', 'top color brown', 'top color green', 'top color gray', 'top color orange', 'top color pink', 'top color purple', 'top color red', 'top color white', 'top color yellow', 'top color others'] 
        bottom_type_list = ['short lower body clothing', 'long lower body clothing'] 
        bottom_color_list = ['bottom color black', 'bottom color blue', 'bottom color brown', 'bottom color green', 'bottom color gray', 'bottom color orange', 'bottom color pink', 'bottom color purple', 'bottom color red', 'bottom color white', 'bottom color yellow', 'bottom color others']
        item_list = ['no bag', 'bag']
        att_list = [age_list, gender_list, hair_type_list, top_type_list, top_color_list, bottom_type_list, bottom_color_list, item_list]
        att_name_list = {'age' : age_list, 
                        'gender' : gender_list, 
                        'hair_type' : hair_type_list, 
                        'top_type' : top_type_list, 
                        'top_color' : top_color_list, 
                        'bottom_type' : bottom_type_list, 
                        'bottom_color' : bottom_color_list, 
                        'item' : item_list}
        num_cat = [3, 2, 3, 2, 12, 2, 12, 2]

    elif cat_type == 'top3':
        top_type_list = ['short sleeve', 'long sleeve'] 
        top_color_list = ['top color black', 'top color blue', 'top color brown', 'top color green', 'top color gray', 'top color orange', 'top color pink', 'top color purple', 'top color red', 'top color white', 'top color yellow', 'top color others'] 
        bottom_type_list = ['short lower body clothing', 'long lower body clothing'] 
        bottom_color_list = ['bottom color black', 'bottom color blue', 'bottom color brown', 'bottom color green', 'bottom color gray', 'bottom color orange', 'bottom color pink', 'bottom color purple', 'bottom color red', 'bottom color white', 'bottom color yellow', 'bottom color others']
        item_list = ['no bag', 'bag']
        att_list = [top_type_list, top_color_list, bottom_type_list, bottom_color_list, item_list]
        att_name_list = {'top_type' : top_type_list, 
                        'top_color' : top_color_list, 
                        'bottom_type' : bottom_type_list, 
                        'bottom_color' : bottom_color_list, 
                        'item' : item_list
        }
        num_cat = [2, 12, 2, 12, 2]

    else:
        return
    
    if not os.path.isdir(os.path.join(SAVE_ROUTE, cat_type)):
        os.makedirs(os.path.join(SAVE_ROUTE, cat_type))

    return att_list, att_name_list, num_cat



if __name__ == '__main__':
    
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    att_list, att_name_list, num_cat = load_category(opt.cat)
    model = AR(num_cat) # we use
    
    model = load_network(model)
    model = model.eval()
    model = model.cuda()

    dataloaders, all_ar_labels = load_data()
    allFiles, _ = map(list, zip(*dataloaders['test'].dataset.samples))
    unnormalize = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    ar_model_dict = dict(type='ar', arch=model, layer_name='layer4', input_size=(224, 224))
    gradcam = GradCAM(ar_model_dict, True)
    gradcam_pp = GradCAMpp(ar_model_dict, True)


    for idx, data in enumerate(tqdm(dataloaders['test'])):
        inputs, labels = data
        ar_labels = all_ar_labels[labels]
        ar_labels = ar_labels.t()

        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda().detach())
            ar_labels = Variable(ar_labels.cuda().detach())
        else:
            inputs,  ar_labels = Variable(inputs), Variable(ar_labels)
        
        batch_size, channel, height, width = inputs.shape
        
        logit = model(inputs)

        for b in range(batch_size): # for an image
            img = inputs[b]
            img = unnormalize(img)
            for c in range(len(num_cat)): # for a category
                save_path = os.path.join(SAVE_ROUTE, opt.cat,
                                        list(att_name_list)[c])
                                # gradcam/full_category/[category]
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                
                pred = list(logit[c][b]).index(max(logit[c][b])) # get the index of the max value of 
                                                           # the current category in the current image
                class_idx = [c, pred]

                
                mask, _ = gradcam(inputs[b].unsqueeze(0), class_idx=class_idx)
                heatmap, result = visualize_cam(mask.cpu(), img.unsqueeze(0))
                # F.upsample(inputs[b], size=(224, 224), mode='bilinear', align_corners=False)

                mask_pp, _ = gradcam_pp(inputs[b].unsqueeze(0), class_idx=class_idx)
                heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), img.unsqueeze(0))

                
                images = []    
                images.append(torch.stack([img.cpu(), 
                                           heatmap, heatmap_pp, result, result_pp], 0))
                images = make_grid(torch.cat(images, 0), nrow=5)
                
                img_name = list(att_name_list)[c] + "-" + \
                            str(allFiles[ idx * opt.batchsize + b ]).split("/")[-1].split("_")[-1]
                output_path = os.path.join(save_path, img_name)

                save_image(images, output_path)
                
        # print(logit)
