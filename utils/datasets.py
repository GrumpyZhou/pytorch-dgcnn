import os
import torch 
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet, ShapeNet as ShapeNetSingle
from utils.shapenet import ShapeNetAll

__all__ = ['RandomSamplePoints', 'ModelNet10', 'ModelNet40', 'ShapeNet', 'ShapeNetCategory']

class RandomSamplePoints(object):
    r"""Uniformly samples :obj:`num` points from original set.
    Args:
        num (int): The number of points to sample.
    """

    def __init__(self, num):
        self.num = num
        
    def __call__(self, data):
        pos, y = data.pos, data.y
        indices = torch.randint(low=0, high=len(pos), size=(self.num,))
        data.pos = pos[indices]
        data.y = y[indices]
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)
    
class ModelNet10:
    categories = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 
                  'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    def __init__(self, root, **kwargs):
        self.root = os.path.join(root, 'ModelNet10')
        self.name = '10'
        self.pre_transform = T.NormalizeScale()
        self.transform = T.SamplePoints(1024)
        self.label_parser = lambda data : data.y 
        self.num_classes = 10
        
    def get_train_split(self):
        return ModelNet(self.root, self.name, train=True, 
                        transform=self.transform, pre_transform=self.pre_transform)
    
    def get_val_split(self): 
        return ModelNet(self.root, self.name, train=False, 
                        transform=self.transform, pre_transform=self.pre_transform)
    
    def get_test_split(self):
        return ModelNet(self.root, self.name, train=False, 
                        transform=self.transform, pre_transform=self.pre_transform)

class ModelNet40:
    categories = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 
                  'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 
                  'curtain', 'desk', 'door', 'dresser', 'flower_pot',
                  'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop',
                  'mantel', 'monitor', 'night_stand', 'person', 'piano', 
                  'plant', 'radio', 'range_hood', 'sink', 'sofa', 
                  'stairs', 'stool', 'table', 'tent', 'toilet', 
                  'tv_stand', 'vase', 'wardrobe', 'xbox']
    
    def __init__(self, root, **kwargs):
        self.root = os.path.join(root, 'ModelNet40')
        self.name = '40'
        self.pre_transform = T.NormalizeScale()
        self.transform = T.SamplePoints(1024)
        self.label_parser = lambda data : data.y
        self.num_classes = 40
        
    def get_train_split(self):
        return ModelNet(self.root, self.name, train=True, 
                        transform=self.transform, pre_transform=self.pre_transform)
    
    def get_val_split(self): 
        return ModelNet(self.root, self.name, train=False, 
                        transform=self.transform, pre_transform=self.pre_transform)
    
    def get_test_split(self):
        return ModelNet(self.root, self.name, train=False, 
                        transform=self.transform, pre_transform=self.pre_transform)
    
class ShapeNet:
    categories = list(ShapeNetAll.categories.keys())
    def __init__(self, root, classification=False, **kwargs):
        self.root = os.path.join(root, 'ShapeNet')
        self.pre_transform = T.NormalizeScale()
        self.train_transform = T.Compose([
            RandomSamplePoints(2048),
            T.RandomTranslate(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2),
        ])
        self.val_transform = RandomSamplePoints(2048)
        self.num_classes = len(self.categories)
        if not classification:
            self.label_parser = lambda data : data.y     
        else:
            self.label_parser = lambda data : data.cid 
        
    def get_train_split(self):
        return ShapeNetAll(self.root, split='train', 
                           transform=self.train_transform, 
                           pre_transform=self.pre_transform)
    
    def get_val_split(self): 
        return ShapeNetAll(self.root, split='val', 
                           transform=self.val_transform,
                           pre_transform=self.pre_transform)
    
    def get_test_split(self):
        return ShapeNetAll(self.root, split='test',
                           pre_transform=self.pre_transform)

class ShapeNetCategory:
    def __init__(self, root, category, **kwargs):
        self.root = os.path.join(root, 'ShapeNet')
        self.category = category
        self.pre_transform = T.NormalizeScale()
        self.train_transform = T.Compose([
            T.RandomTranslate(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2),
        ])
        self.label_parser = lambda data : data.y
        
    def get_train_split(self):
        return ShapeNetSingle(self.root, self.category, train=True,
                            transform=self.train_transform, 
                            pre_transform=self.pre_transform)
    
    def get_val_split(self): 
        return ShapeNetSingle(self.root, self.category, train=False,
                            pre_transform=self.pre_transform)
    
    def get_test_split(self):
        return ShapeNetSingle(self.root, self.category, train=False,
                            pre_transform=self.pre_transform)
        