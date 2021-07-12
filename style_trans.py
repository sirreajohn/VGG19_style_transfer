# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 10:55:16 2021

@author: mahesh
"""

import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
#---------------------------QoL functions---------------------------------

def load_img(img_path, max_size = 400, shape = None):
    image = Image.open(img_path).convert("RGB")
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_trans = transforms.Compose([
                       transforms.Resize(size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))])
    image = in_trans(image).unsqueeze(0)
    return image

def tensor_to_image(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array([0.5,0.5,0.5]) + np.array([0.5,0.5,0.5])
    image = image.clip(0,1)
    return image
#--------------------------------------------------------------------------
class style_trans:
    
    def __init__(self,content, style, lr = 0.03,sample_rate = 500, 
                 epochs = 5000,content_ratio = 1,style_ratio = 1):
        self.content = content # content_image
        self.style = style     # style image
        self.lr = lr           #learning_rate
        self.sample_rate = sample_rate #sample rate 
        self.epochs = epochs  #epochs 
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") # CPU vs GPU
        self.style_weights = {"conv1_1":1,
                             "conv2_1":0.75,
                             "conv3_1":0.2,
                             "conv4_1":0.2,
                             "conv5_1":0.2}  # weights in weighted sum for style learning 
        self.content_ratio = content_ratio  # ratio of content to be learned by target
        self.style_ratio = style_ratio      # ratio of style to be learned by target
        
    def initialize_model(self):
        '''
        used to initialize VGG19 and freeze weights for features
        '''
        vgg19 = models.vgg19(pretrained = True).features
        
        for param in vgg19.parameters():
            param.requires_grad_(False)
            
        self.vgg19 = vgg19.to(self.dev)
    
    def get_features(self, image):
        '''
            generate feature maps for image provided, essential for learning purposes
            image is passed through layers of VGG19 and maps are extracted at pre defined layers(as defined in layers{})
        '''
        layers = {"0":"conv1_1",
                  "5":"conv2_1",
                  "10":"conv3_1",
                  "19":"conv4_1",
                  "21":"conv4_2", #content extraction
                  "28":"conv5_1"}
        features = {}
        
        for name, layer in self.vgg19._modules.items():
            image = layer(image)      # image being passed through layers of VGG19
            if name in layers:
                features[layers[name]] = image # extract map if current layer is in list 
        return features
    
    def gram_mat(self,tensor):
        
        '''
            a gram matrix is defined as matrix formed by matmul of its own transpose 
            ex: for given matrix A
            gram(A) = A.A^T
        '''
        _,d,h,w = tensor.size()
        tensor = tensor.view(d,h*w)
        gram = torch.mm(tensor, tensor.t())
        return gram
    
    def train(self):
        ''' 
            main train loop:
            does all the training and display
        '''
        #---feature extraction-----
        
        content_features = self.get_features(self.content)
        style_features = self.get_features(self.style)
        
        #-----------target creation/style gram matrix -------------
        
        self.target = self.content.clone().requires_grad_(True).to(self.dev)
        style_grams = {layer : self.gram_mat(style_features[layer]) for layer in style_features}
        #-------------- optimizer declaration---------------
        self.optimizer = optim.Adam([self.target], lr = self.lr)  
        
        #---------- main loop ---------------
        for epoch in tqdm(range(self.epochs)):
            target_features = self.get_features(self.target) # extract target features
            
            # loss oof content image (usually VERY less due to target being clone of content)
            content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"])**2)
        
            # loss is calcuated for every layer of VGG19 between style image and target image
            style_loss = 0
            
            for layer in self.style_weights:
                # get target features and calculate gram matrix of it 
                target_feature = target_features[layer]
                target_gram = self.gram_mat(target_feature)
                
                # get style gram matrix at specfic layer of VGG19
                style_gram = style_grams[layer]
                # style_loss calcualted at specific layer 
                layer_style_loss = torch.mean((target_gram - style_gram)**2)
                _, d,h,w = target_feature.shape
                
                # accumulating style loss for all layers
                style_loss += layer_style_loss/(d * h * w) # (d*h*w) is divided for normalization
                
            # total weighted sum of loss
            total_loss = (content_loss* self.content_ratio) + (style_loss * self.style_ratio)
            
            #--------back prop--------------
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            #---------------------------
            
            
            if epoch % self.sample_rate == 0:
                plt.imshow(tensor_to_image(self.target))
                plt.axis("off")
                plt.title(f"epochs : {epoch}, Total_loss : {total_loss.item()}")
                plt.show()

    def show_1x3(self,target):
        '''
         used to generate 1x3 image grid to display content, style and target image
        '''
        fig , (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (16,16))
        ax1.imshow(tensor_to_image(self.content))
        ax2.imshow(tensor_to_image(self.style))
        ax3.imshow(target)
        ax3.axis("off")
        ax2.axis("off")
        ax1.axis("off")
        fig.show()
        
if __name__ == "__main__":
    content_image = load_img("style Images/content/pilus.jpg")
    style_image = load_img("style Images/StarryNight.jpg")
    
    style_transfer = style_trans(content_image,style_image)
    style_transfer.initialize_model()
    
    #inspect = style_transfer.get_features(style_image)
    style_transfer.train()
    
    
    
    
    
    
    
    
    
    
    
    
    
    