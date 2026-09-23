import torch
from medAI.registry import register


@register("adapter", "dict_to_dino_format")
def dict_to_dino_format(): 
    def inner(sample): 
        images = sample['image'] if 'image' in sample else sample['images']
        labels = sample['label'] if 'label' in sample else torch.zeros(len(images[0])).long()
        return images, labels

    return inner 


@register("adapter", "dict_to_ibot_format")
def dict_to_ibot_format():
    def inner(sample):
        images = sample['image'] if 'image' in sample else sample['images']
        labels = sample['label'] if 'label' in sample else torch.zeros(len(images[0])).long()
        masks = sample['mask'] if 'mask' in sample else sample['masks'] if 'masks' in sample else None
        return images, labels, masks
    return inner


