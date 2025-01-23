import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ClipDataset(Dataset):
    def __init__(self, data_folder, images_subfolder, descriptions_filename, preprocess_fn, tokenizer_fn):

        self.data_path = data_folder
        self.images_path = os.path.join(data_folder, images_subfolder)
        self.descriptions_file = os.path.join(data_folder, descriptions_filename)

        with open(self.descriptions_file, "r", encoding="utf-8") as f:
            descriptions_dict = json.load(f)

        self.entries = list(descriptions_dict.items())

        self.preprocess = preprocess_fn
        self.tokenizer = tokenizer_fn

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        img_filename, description = self.entries[idx]
        img_path = os.path.join(self.images_path, img_filename)

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)

        text_tokens = self.tokenizer(description)
        return img_tensor, text_tokens
    
def create_dataloaders(data_folder,
                       images_subfolder,
                       descriptions_filename,
                       preprocess_fn,
                       tokenizer_fn,
                       batch_size=16,
                       split_ratio=0.7):
    
    dataset = ClipDataset(data_folder,
                          images_subfolder,
                          descriptions_filename,
                          preprocess_fn,
                          tokenizer_fn)
    
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    ds_train, ds_val = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(ds_train,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4)
    
    val_loader = DataLoader(ds_val,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers=4)
    
    return train_loader, val_loader
    