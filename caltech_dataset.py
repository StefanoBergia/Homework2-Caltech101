from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):

    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root=root
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.splitted_data = {}
        self.dictionary = {}
        self.values = []

        id = 0;
        i=0;
        f = open('Caltech101'+'/'+split+'.txt', "r")
        line = f.readline()
        while(line):
            spl=line.split('/')
            if not spl[0] == 'BACKGROUND_Google':
                if not spl[0] in self.dictionary:
                    self.dictionary[spl[0]] = id
                    id += 1
                if not spl[0] in self.splitted_data:
                    self.splitted_data[spl[0]] = []
                self.splitted_data[spl[0]].append(i)
                i += 1
                self.values.append(line[:-1])
            line = f.readline()
        print(self.splitted_data)


    def __getitem__(self, index):
        if index >= len(self.values):
            return None, None

        image = pil_loader(self.root + '/' + self.values[index])
        label = self.dictionary[self.values[index].split('/')[0]]


        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        length = len(self.values)
        return length

    def split_dataset(self):
        trainIndexes=[]
        valIndexes=[]
        t=open('Caltech101/trueTraining.txt', 'w')
        v=open('Caltech101/trueValidation.txt','w')

        for key in self.splitted_data.keys():
            i=0
            for el in self.splitted_data[key]:
                if i<0.5*len(self.splitted_data[key]):
                    trainIndexes.append(el)
                    t.write(self.values[el]+'\n')
                else :
                    valIndexes.append(el)
                    v.write(self.values[el]+'\n')
                i +=1
        return (trainIndexes, valIndexes)

    def return_labels(self):
        labels=self.dictionary.keys()
        return labels