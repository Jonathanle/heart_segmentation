
import torch
from torch.utils.data import Dataset 
from datasets import Dataset, DatasetDict, Image
import albumentations as A

import logging
import os
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)



# What do we want to create as an output?

# Create a dataset interface from the hearts
# Interpret the files in such a way that the dataset is loaded with images and labels 
# TODO: files --> dataset

# Datastructure Defining filename prefix + the count


filename_prefixes = {"JPCLN": 154, "JPCNN": 93}

def get_filenames(filename_prefixes): 
    filenames = []
    
    for prefix, num_images in filename_prefixes.items(): 
        for i in range(1, num_images + 1):
        
            number = str(i) 

            while len(number) < 3: 
               number = "0" + number

        

            filenames.append(f"{prefix}{number}.png")
       
    return filenames


def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset
    # I hope that this dataset, casts the filenames of image_paths, and label_paths into actual images. 
    # loads the image objects from huggingface

def visualize_map(image, segmentation_map):
    segmentation_map = np.array(segmentation_map)
    color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

    id2color = { 0: 255, 1: 0}
    print(image)
    print(segmentation_map)


    for label, color in id2color.items():
        color_seg[segmentation_map == label, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")
    ax[1].imshow(img)
    ax[1].set_title("Segmentation Mask")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()




class SegmentationDataset(Dataset):
  def __init__(self, dataset, transform):
    self.dataset = dataset
    self.transform = transform

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    original_image = np.array(item["image"])
    original_segmentation_map = np.array(item["label"])


    # apply albumentations -- outpuit is ndarray
    transformed = self.transform(image=original_image, mask=original_segmentation_map)
    # convert ndarray to torch tensor
    image, target = torch.tensor(transformed['image']), torch.LongTensor(transformed['mask'])

    # convert to C, H, W - torch requires channel first
    image = image.permute(2,0,1)

    return image, target, original_image, original_segmentation_map



def main():
    logging.basicConfig(filename="debug.log", level=logging.INFO)


    logger.info("Retrieving Filenames")
    filenames = get_filenames(filename_prefixes)
    logger.info("Finished Retrieving Filenames")

    
    logger.info("Creating Dataset")

    heart_folder = "data/jsrt_images"
    mask_folder = "data/heart_masks"

    dataset = create_dataset([os.path.join(heart_folder, filename) for filename in filenames], [os.path.join(mask_folder, filename) for filename in filenames])
                   
    logging.info(f"Type of dataset: {type(dataset)}, info: {dataset.info}")



    # Create the Train-validation-test split
    print(dataset[0]["image"])
    print(dataset[15]["image"]) # seems to behave as a list of image - label objects. 



    visualize_map(dataset[0]["image"], dataset[0]["label"])

    

if __name__ == "__main__": 
   main()

