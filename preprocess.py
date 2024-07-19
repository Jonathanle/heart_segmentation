
import torch
from torch.utils.data import Dataset as TorchDataset
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

# Train_test split for data
SHUFFLE_SEED = 42
train_val_test_split = (0.01, 0.79, 0.2)
FLIP_PROBABILITY = 0

# Normalization Parameters for transforming image data for the model to accept.
ADE_MEAN = [0.485, 0.456, 0.406]
ADE_STD = [0.229, 0.224, 0.225]

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
                                "mask": sorted(label_paths)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("mask", Image())

    return dataset
    # I hope that this dataset, casts the filenames of image_paths, and label_paths into actual images. 
    # loads the image objects from huggingface


# Define Image Augmentation Transform
# Are these defined propertly to be applied everywhere?
train_transform = A.Compose([
# don't using crop
A.Resize(width=504, height=504),
A.HorizontalFlip(p=FLIP_PROBABILITY),
A.Normalize(mean=ADE_MEAN, std=ADE_STD),
], is_check_shapes= True)

val_transform = A.Compose([
    A.Resize(width=504, height=504),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ], is_check_shapes=True)

# takes 2 numpy images and overlays the images over one another
# np arrays must have the form of the (height, width, channel)
def visualize_map(image, segmentation_map):
    segmentation_map = np.array(segmentation_map)
    color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

    id2color = { 0: (0, 255, 255), 1: (255, 255, 0)}


    transform = A.Compose([
    A.Resize(width=504, height=504)
    ])

    image_np = np.array(image)




    # check if the image is permuted
    if image.shape[0] == 512 and image.shape[1] == 512 and image.shape[2] == 3:
        image = transform(image=image)['image']

    





    resized_image = transform(image=image_np)['image']

    #print(color_seg.shape) 
    #print(segmentation_map[297: 330, 245:275])  # binary masks with shape and 

    for label, color in id2color.items():

        color_seg[segmentation_map == label, :] = color



    color_seg = transform(image = color_seg)["image"]
    
    # Show image + mask
    img = np.array(image) * 0.2 + color_seg * 0.8
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

# How does huggingface Dataset relate to torch dataset? in dataprocessing? for batches? 
# I realized that I had 2 dataset options - Did not realize the vulnereability in inheriting a dataset from huggingface vs torch
class SegmentationDataset(TorchDataset):
  def __init__(self, dataset, transform):
    self.dataset = dataset
    self.transform = transform

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    original_image = np.array(item["image"])
    original_segmentation_map = np.array(item["mask"])


    # apply albumentations -- soutpuit is ndarray
    # what does albumentations do to the images and masks? could I just use torch vision?

    # the error occurs here

    # does this image also accept batch processing of multipel images?
    transformed = self.transform(image=original_image, mask=original_segmentation_map)
    # convert ndarray to torch tensor
    image, target = torch.tensor(transformed['image']), torch.LongTensor(transformed['mask'])

    # convert to C, H, W - torch requires channel first
    image = image.permute(2,0,1)

    return image, target, original_image, original_segmentation_map
    # Returned original Image and original segmentation map for debugging purposes.

def retrieve_heart_dataset(heart_folder = "data/jsrt_images", mask_folder = "data/heart_masks"):
    filenames = get_filenames(filename_prefixes)

    dataset = create_dataset([os.path.join(heart_folder, filename) for filename in filenames], [os.path.join(mask_folder, filename) for filename in filenames])
    
    trainval_test_datasets = dataset.train_test_split(test_size = train_val_test_split[2], seed = SHUFFLE_SEED)


    trainval_datasets = trainval_test_datasets["train"].train_test_split(test_size = train_val_test_split[1]/(train_val_test_split[0] + train_val_test_split[1]) , seed = SHUFFLE_SEED)


    train_dataset = trainval_datasets["train"]
    val_dataset = trainval_datasets["test"]
    test_dataset = trainval_test_datasets["test"]

    
    train_dataset = SegmentationDataset(train_dataset, transform = train_transform)
    val_dataset = SegmentationDataset(val_dataset, transform = val_transform)
    test_dataset = SegmentationDataset(test_dataset, transform = val_transform)


    return train_dataset, val_dataset, test_dataset


def main():
    logging.basicConfig(filename="debug.log", level=logging.CRITICAL)


    logger.info("Retrieving Filenames")
    filenames = get_filenames(filename_prefixes)
    logger.info("Finished Retrieving Filenames")

    
    logger.info("Creating Dataset")

    heart_folder = "data/jsrt_images"
    mask_folder = "data/heart_masks"


    # dataset object allowing for indexing of associated images and labels.
    dataset = create_dataset([os.path.join(heart_folder, filename) for filename in filenames], [os.path.join(mask_folder, filename) for filename in filenames])
                   
    logging.info(f"Type of dataset: {type(dataset)}, info: {dataset.info}")

    # Create the Train-validation-test split

    visualize_map(dataset[0]["image"], dataset[0]["mask"])


    # TODO: Create Multiple DAtasets for train, validation + test datasets.
    torch_dataset = SegmentationDataset(dataset, transform = train_transform) 


    print(torch_dataset[0][0].shape)
    print(torch_dataset[0][1].shape)
    print(len(torch_dataset))




    

if __name__ == "__main__": 
   main()

