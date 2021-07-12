# Image to Image style Transfer using VGG-19 Neural Network

 - A py script to demonstrate the use of VGG19 in style transfer

## Working 

We take two images, content/context and a style image. the "style" of style image is learned and appiled on a content image.

![out_1](https://github.com/sirreajohn/VGG19_style_transfer/blob/main/outs/download%20(1).png)

- Initially we import the SOTA VGG19 model without head and freeze the weights. we extract output from specific layers and use them to train our model.
  - For style extraction
    - conv1_1 (layer 0)
    - conv2_1 (layer 5)
    - conv3_1 (layer 10)
    - conv4_1 (layer 19)
    - conv5_1 (layer 28)
  - For Content Extraction
    - conv4_2 (layer 28)
- These specific layers are chosen according to this [paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
- We pass style and content images through this network and extract "features" at given layers, style images go through additional processing by making a gram matrix(G = A . A.t() ). 
- We initialize a "target image" and calculate loss between target and content,target and style and combine the losses in weighted manner.
```Python
content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"])**2)

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
      style_loss += layer_style_loss/(d * h * w) #(d*h*w) is divided for normalization

total_loss = (content_loss* self.content_ratio) + (style_loss * self.style_ratio)
```
- this combined weighted loss of content and style are fed into optimizer(Adam) to start gradient descent.
- for implementation purposes, the "target image" used is clone of "content image" to reduce the loss to be covered by GD and it just gives better results!
```
self.target = self.content.clone().requires_grad_(True).to(self.dev)
``` 

## Results
<img src="https://github.com/sirreajohn/VGG19_style_transfer/blob/main/outs/Mona.gif" width="425" height = "500"/> <img src="https://github.com/sirreajohn/VGG19_style_transfer/blob/main/outs/pilus_test_4.gif" width="425" height = "500"/>


img credits : [Piyush Raj](https://www.linkedin.com/in/piyush-raj-988961167/)

## Requirements
Make sure you have all these!
```
numpy==1.19.2
matplotlib==3.3.2
tqdm==4.50.2
torch==1.7.1
torchvision==0.8.2
Pillow==8.3.1
```

## Usage 
- make sure you are running >=python 3.8.5
- make sure you have all libraries mentioned or just run requirements.txt
- load images and initialize the class with both images, fetch VGG19 and train!

```
content_image = load_img("Path-to-content-image")
style_image = load_img("Path-to-style-image")

style_transfer = style_trans(content_image,style_image)
style_transfer.initialize_model()
#inspect = style_transfer.get_features(style_image)
style_transfer.train()
```

## Contact

Mahesh Patapalli - mahesh.patapali@gmail.com

[linkedin](https://www.linkedin.com/in/mahesh-patapalli-bba1aa191/) - [github](https://github.com/sirreajohn)

Project Link: [https://github.com/sirreajohn/VGG19_style_transfer](https://github.com/sirreajohn/VGG19_style_transfer)
     
