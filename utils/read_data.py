#encoding: utf-8
from torchvision import transforms, datasets
import os
import torch
from PIL import Image
import scipy.io as scio

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

#读取MIT67数据集，训练函数只需要dataloders即可
def MIT67Data(args):
# data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {}
    #image_datasets['train'] = datasets.ImageFolder(os.path.join(args.data_dir, 'ILSVRC2012_img_train'), data_transforms['train'])

    image_datasets['train'] = MIT67_DataSet(os.path.join(args.data_dir, 'database'),
                                      os.path.join(args.data_dir, 'evaluation', 'TrainImages.txt'),
                                      os.path.join(args.data_dir, 'evaluation', 'categories.txt'),
                                      data_transforms['train'])
    image_datasets['val'] = MIT67_DataSet(os.path.join(args.data_dir, 'database'),
                                    os.path.join(args.data_dir, 'evaluation', 'TestImages.txt'),
                                    os.path.join(args.data_dir, 'evaluation', 'categories.txt'),
                                    data_transforms['val'])

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers) for x in ['train', 'val']}


    #dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders


class MIT67_DataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, listname,labelname,data_transforms):   #根目录，图片标签，
        self.path=root_dir
        self.listName = listname
        self.labelName=labelname
        self.images=[os.path.join(self.path,line.strip()) for line in open(self.listName)]#读取图片
        self.categories = [line.strip().split('/')[0] for line in open(self.listName)] #读取类别 
        self.dicts=dict() #英文类别和数字编码之间的关系
        self.labels=list() #存放图片的类别
        for line in open(self.labelName):
            self.dicts[str(line.strip().split()[0])]=int(line.strip().split()[1]) 
        for c in self.categories:
            self.labels.append(self.dicts[c])
        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        data=self.images[item]
        label=self.labels[item]
        img = Image.open(data).convert('RGB')
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return img, label

#读取Places365数据集
def Places365Data(args):
# data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {}
    #image_datasets['train'] = datasets.ImageFolder(os.path.join(args.data_dir, 'ILSVRC2012_img_train'), data_transforms['train'])

    # Places365 Challeng
    #image_datasets['train'] = Places365_DataSet(os.path.join(args.data_dir, 'database','data_large'),
    #                                  os.path.join(args.data_dir, 'evaluation', 'places365_train_standard.txt'),
    #                                  data_transforms['train'])
    #image_datasets['val'] = Places365_DataSet(os.path.join(args.data_dir, 'database','val_large'),
    #                                os.path.join(args.data_dir, 'evaluation', 'places365_val.txt'),
    #                                data_transforms['val'])

    #Places365-Standard
    image_datasets['train'] = Places365_DataSet(os.path.join(args.data_dir, 'data_large'),
                                      os.path.join(args.data_dir, 'filelist_places365-standard', 'places365_train_standard.txt'),
                                      data_transforms['train'])
    image_datasets['val'] = Places365_DataSet(os.path.join(args.data_dir, 'val_large'),
                                    os.path.join(args.data_dir, 'filelist_places365-standard', 'places365_val.txt'),
                                    data_transforms['val'])

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers) for x in ['train', 'val']}


    #dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders


class Places365_DataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, listname,data_transforms):   #根目录，图片标签，
        self.path=root_dir
        self.listName = listname
        self.images=[self.path+'/'+line.strip().split(' ')[0] for line in open(self.listName)]#读取图片
        self.labels=[]
        self.labels=[int(line.strip().split(' ')[1]) for line in open(self.listName)]#读取标签
        self.data_transforms = data_transforms
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        data=self.images[item]
        label=self.labels[item]
        img = Image.open(data).convert('RGB')
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return img, label
    
