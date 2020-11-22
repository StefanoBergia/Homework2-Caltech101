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


class Caltech():
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root=root
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        self.dictionary = {}
        self.values = []

        id = 0;
        f = open(split+'.txt', "r")
        line = f.readline()
        while(line):
            spl=line.split('/')
            if not spl[0] == 'BACKGROUND_Google':
                if not spl[0] in self.dictionary:
                    self.dictionary[spl[0]] = id
                    id += 1

            self.values.append(line[:-1])
            line = f.readline()

    def __getitem__(self, index):
        image = pil_loader(self.root+'/'+self.values[index])
        label = self.dictionary[self.values[index].split('/')[0]]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):

        length = len(self.values)
        return length

c=Caltech('test');
print(c.__getitem__(53))
print(c.__len__())