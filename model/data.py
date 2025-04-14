from torchvision import transforms
from PIL import Image
import json
import os
import torch

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, caption_file, image_size=256, low_res_factor=4, random_flip=True):
        """
        image_dir: directory with MS-COCO images.
        caption_file: path to COCO captions annotation JSON (instances with image IDs and captions).
        image_size: the target size (height=width) for high-res images.
        low_res_factor: factor by which to downsample for low-res (e.g., 4 means downsample to 1/4 size then upsample).
        random_flip: if True, apply random horizontal flip augmentation.
        """
        super().__init__()
        # Load captions from the COCO annotation file
        with open(caption_file, 'r') as f:
            captions_data = json.load(f)
        # captions_data is a dict with 'images' and 'annotations'
        # We will create a mapping from image_id to caption (use first caption per image for simplicity)
        self.id_to_caption = {}
        for ann in captions_data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            if image_id not in self.id_to_caption:
                # Take the first caption encountered for each image (could also randomize or use all captions)
                self.id_to_caption[image_id] = caption
        # Gather image file paths and corresponding captions
        self.samples = []
        for img_info in captions_data['images']:
            img_id = img_info['id']
            file_name = img_info['file_name']
            if img_id in self.id_to_caption:  # ensure we have a caption
                path = os.path.join(image_dir, file_name)
                caption = self.id_to_caption[img_id]
                self.samples.append((path, caption))
        self.image_size = image_size
        self.low_res_factor = low_res_factor
        # Define transforms for high-res image
        self.high_res_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),            # resize (could also crop to maintain aspect)
            transforms.RandomHorizontalFlip(p=0.5 if random_flip else 0.0),
            transforms.ToTensor()  # convert to tensor [0,1]
        ])
        # Note: We'll create low-res version on the fly in __getitem__ for flexibility.
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, caption = self.samples[idx]
        # Load image
        img = Image.open(path).convert('RGB')
        # High-res image tensor
        high_res = self.high_res_transform(img)  # (3, H, W) with H=W=image_size
        # Create low-res version by downsampling and upsampling
        if self.low_res_factor is not None and self.low_res_factor > 1:
            low_size = self.image_size // self.low_res_factor
            low_res_img = transforms.Resize((low_size, low_size), interpolation=Image.BICUBIC)(img)
            # Upsample back to original size
            low_res_img = transforms.Resize((self.image_size, self.image_size), interpolation=Image.BICUBIC)(low_res_img)
            low_res = transforms.ToTensor()(low_res_img)
        else:
            # If no downsampling factor given, just use the high_res as low_res (or could apply slight blur)
            low_res = high_res.clone()
        return low_res, high_res, caption

# Usage
train_dataset = COCODataset(image_dir="../data/coco/images/train2017", caption_file="../data/coco/annotations/captions_train2017.json", image_size=256, low_res_factor=4)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
print("Success")