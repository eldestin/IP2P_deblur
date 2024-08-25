from PIL import Image
import torch
from torch import nn
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPModel, CLIPImageProcessor

from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import torchvision.transforms as transforms
import random

class IP2P_dataset(Dataset):

    def __init__(self, path, train, size = 320, format = ".jpg", crop = True, tokenizer = None):
        super(IP2P_dataset, self).__init__()
        self.crop_size = size
        self.format = format
        self.train = train
        self.before_edit_dir = os.listdir(os.path.join(path, "before_edit"))
        # self.before_edit_imgs = [os.path.join()]
        self.before_edit_imgs = os.path.join(path, "before_edit")
        self.after_edit_dir = os.path.join(path, "after_edit")
        self.prompt_dir = os.path.join(path, "prompt")
        self.crop = crop
        # print(self.before_edit_dir)
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        before_edit = Image.open(os.path.join(self.before_edit_imgs, self.before_edit_dir[idx]))
        name = self.before_edit_dir[idx].split("_")[-1].split(".jpg")[0]
        after_name, prompt_name = "image_" + name + self.format, "text_" + name + ".txt"
        after_edit = Image.open(os.path.join(self.after_edit_dir, after_name))
        with open(os.path.join(self.prompt_dir, prompt_name)) as f:
            prompt = f.read()
        if self.crop:
            i,j,h,w=tfs.RandomCrop.get_params(before_edit,output_size=(self.crop_size,self.crop_size))
            before_edit=FF.crop(before_edit,i,j,h,w)
            after_edit=FF.crop(after_edit,i,j,h,w)
        before_edit, after_edit = self.aug_data(before_edit.convert("RGB"), after_edit.convert("RGB"))
        before_edit = tfs.ToTensor()(before_edit)
        after_edit = tfs.ToTensor()(after_edit)
        prompt = self.tokenizer(
            prompt, padding = "max_length", max_length = self.tokenizer.model_max_length, truncation = True,
            return_tensors = "pt"
        )
        un_cond = self.tokenizer([""] , padding = "max_length", max_length = self.tokenizer.model_max_length,
                             return_tensors = "pt")
        return before_edit, after_edit, prompt, un_cond

    def aug_data(self, data, target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        return data, target

    def __len__(self):
        return len(self.before_edit_dir)


class DPDD_prompt(Dataset):
    def __init__(self,path,train,size=240,format='.png',crop=True,mask_generator=None, tokenizer = None):
        super(DPDD_prompt,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
      #  self.AugDict = AugDict
        self.haze_imgs_dir=os.listdir(os.path.join(path,'inputC'))
        #self.haze_imgs_dir = [x for x in haze_imgs_dir if ('.png' in x)]
        self.haze_imgs = [os.path.join(path,'inputC',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'target')   
        self.crop=crop
        self.tokenizer = tokenizer
        self.prompt = "deblur the blurred image"
        if self.crop and not train:
            seed = 2
            random.seed(seed)
            img = Image.open(self.haze_imgs[0])
            i,j,h,w = tfs.RandomCrop.get_params(img,output_size=(self.size,self.size))
            self.info = [i,j,h,w]
        
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])

        img=self.haze_imgs[index]
        name_syn=img.split('/')[-1]#.split('_')[0]#
        id = name_syn
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        prompt = self.tokenizer(
            self.prompt, padding = "max_length", max_length = self.tokenizer.model_max_length, truncation = True,
            return_tensors = "pt"
        )
        if self.crop and not self.train:
            i,j,h,w=self.info
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        if self.crop and self.train:
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
          
            
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB"))
        haze = tfs.ToTensor()(haze)
        clear = tfs.ToTensor()(clear)
        un_cond = self.tokenizer([""] , padding = "max_length", max_length = self.tokenizer.model_max_length,
                             return_tensors = "pt")
        return haze,clear, prompt, un_cond
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
           
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
                

        return  data , target
    def __len__(self):
        return len(self.haze_imgs)