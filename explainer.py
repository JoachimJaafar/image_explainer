import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from operator import itemgetter

from lime import lime_image
from skimage.segmentation import mark_boundaries

import glob
import time
import magic

import requests
from io import BytesIO



NOIR = np.array([0,0,0])

class Explainer:
    def __init__(self, model_name, model_names=[]):
        """
        Add a classification model in the Explainer.
        """

        if model_name in model_names:
            self.model = models.__dict__[model_name](pretrained=True)
        else:
            raise KeyError("The chosen model does not exist, must be in ",model_names)

    def get_image(self, path, is_url):
        """
        Returns image in correct format.
        """
        if is_url:
            resp = requests.get(path)
            return Image.open(BytesIO(resp.content)).convert('RGB')
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def get_input_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.806],
                                        std=[0.229, 0.224, 0.225])       
        transf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])    

        return transf

    def get_input_tensors(self, img):
            transf = self.get_input_transform()
            # unsqeeze converts single image to batch of 1
            return transf(img).unsqueeze(0)

    def get_top_class(self, img):

        idx2label = []
        with open(os.path.abspath('./labels_map.json'), 'r') as read_file:
            class_idx = json.load(read_file)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

        img_t = self.get_input_tensors(img)
        self.model.eval()
        logits = self.model(img_t)

        probs = F.softmax(logits, dim=1)
        probs5 = probs.topk(5)

        tupl = tuple((p,c, idx2label[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy()))
        return tupl[0][2], tupl[0][0]

    def generate_image(self, img,long,larg,x,y,i):
        img_np = np.array(img)
        for lo in range(len(img_np)):
            for la in range(len(img_np[0])):
                if not(x * long < lo < ((x+1) * long) and y * larg < la < ((y+1) * larg)):
                    continue
                img_np[lo][la] = NOIR

        img_to_save = Image.fromarray(img_np)
        img_to_save.save('./tmp/tmp' + str(i) + '.jpg')

    def generate_all_images(self, img, size):

        long, larg = img.size
        i=0
        for la in range(larg//size):
            for lo in range(long//size):
                self.generate_image(img,size,size,la,lo,i)
                i+=1


    def get_stats(self, img_originale, top):
        self.top_predicted = self.get_top_class(img_originale)[0]

        l_mauvais = []
        l_bons = []
        for image in os.listdir('./tmp'): 
            img = Image.open('./tmp/' + image)
            label, p = self.get_top_class(img)
            if label != self.top_predicted:
                l_mauvais.append((image, label, p))
            else:
                l_bons.append((image, label, p))

        l_mauvais.sort(key=itemgetter(2), reverse=True)

        l_bons.sort(key=itemgetter(2))

        l = l_mauvais + l_bons

        return l[:top]

    def generate_result_image(self, image_path, size, is_url):

        img = self.get_image(image_path, is_url)

        img_np = np.array(img)
        
        long, larg = img.size
        long = long // size
        larg = larg // size
        top_class = self.get_stats(img, int(long*larg*0.3))

        i=0
        for la in range(larg):
            for lo in range(long):
                if i in [int(t[0].split('.')[0][3:]) for t in top_class]:
                    i+=1
                    continue
                i+=1
                for j in range(la*size,(la+1)*size):
                    for k in range(lo*size,(lo+1)*size):
                        img_np[j][k] = NOIR

        for la in range(len(img_np)):
            for lo in range(len(img_np[0])):
                if lo >= long*size or la >= larg*size:
                    img_np[la][lo] = NOIR
                    
        self.clean()

        nom_image = image_path.split('/')[-1]

        img_to_save = Image.fromarray(img_np)
        img_to_save.show(title=nom_image)


    def generate_all(self, image_path, size, is_url):
        image = self.get_image(image_path, is_url)
        self.generate_all_images(image, size)
        self.generate_result_image(image_path, size, is_url)

    def setup(self):
        if not os.path.exists("./tmp"):
            os.mkdir("./tmp")

    def clean(self):
        for tmp_file in os.listdir("./tmp"):
            os.remove("./tmp/" + tmp_file)
        os.rmdir("./tmp")

    def main(self, image_path, is_url):
        self.setup()

        run = True

        if not is_url:
            if not magic.Magic(mime=True).from_file(image_path) in ("image/png", "image/jpeg", "image/jpg"):
                return "This file is not an image."
        else:
            if not requests.head(image_path).headers["content-type"] in ("image/png", "image/jpeg", "image/jpg"):
                return "This URL is not an image."

        image = image_path.split('/')[-1]

        if os.path.exists("./"+image):
            answer = input(image+" is already explained. Do you want to overwrite it ? (Y/N)\n")
            while True:
                if answer.upper() == 'N':
                    run = False
                    break

                elif answer.upper() == 'Y':
                    break

                else:
                    answer = input("Incorrect input. Please try again. (Y/N)\n")

        if run:
            img = self.get_image(image_path, is_url)
            longueur,largeur = img.size
            
            size = max(longueur,largeur)
            size = int(size/10)
            
            self.generate_all(image_path, size, is_url)

            return "The classifier predicted : "+self.top_predicted.replace("_"," ")+". Choose the model to use."