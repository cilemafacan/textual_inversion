import json
import numpy as np
from PIL import Image
from utils import util
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, json_path, tokenizer, resolution=512, transform=None, img_padding=True):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.img_padding = img_padding
        self.transform = transform  
        with open(json_path, 'r') as f:
            self.image_caption_pairs = json.load(f)

    
    def __len__(self):
        return len(self.image_caption_pairs)
    
    def __getitem__(self, idx):
        image_caption_pair = self.image_caption_pairs[idx]
        image = image_caption_pair['image']
        caption = image_caption_pair['caption']
        text_ids = self.tokenizer.encode(caption,
                                         return_tensors='pt',
                                         padding='max_length',
                                         max_length=77,
                                         truncation=True).input_ids[0]
        
        if self.img_padding:
            np_image = np.array(image)
            padded_image = util.padding(np_image)
            image = Image.fromarray(padded_image)
        
        image.resize((self.resolution, self.resolution))

        if self.transform:
            pixel_values = self.transform(image)

        return pixel_values, text_ids