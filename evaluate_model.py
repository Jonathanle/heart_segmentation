import preprocess
from preprocess import visualize_map
from model import Dinov2ForSemanticSegmentation
from transformers import Dinov2Config
from torch.utils.data import DataLoader
import torch
import albumentations as A
import numpy as np
import os
import pdb
import evaluate
from torch.optim import AdamW
from tqdm.auto import tqdm

from train_val import Evaluator, collate_fn

id2label={0: "background", 1: "heart"}
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./models/model2.pth"


train_dataset, val_dataset, test_dataset= preprocess.retrieve_heart_dataset()

val_dataloader = DataLoader(val_dataset, batch_size = 5, shuffle=False, collate_fn=collate_fn)


config = Dinov2Config(image_size = 504, patch_size = 14, num_labels=2)
model = Dinov2ForSemanticSegmentation(config).from_pretrained("facebook/dinov2-base", id2label=id2label, num_labels=len(id2label))
model.to(device)




if os.path.isfile(MODEL_PATH): 
    model.load_state_dict(torch.load(MODEL_PATH))
    
a = iter(val_dataloader)
data =next(a)
data = next(a)
 
image = data["image"].to(device) # give a variable to the model that is in cuda device
mask = data["mask"].to(device)

with torch.no_grad():
    predicted = model(pixel_values = image, labels = mask).logits.argmax(dim=1)


for i in range(5):
    visualize_map(data["original_image"][i], predicted.cpu()[i])
    visualize_map(data["original_image"][i], data["segmentation"][i])


evaluator = Evaluator(val_dataloader)
evaluator.evaluate_epoch(model)






# Plot Graphs using the attributes plotted in history datastructure using 6 metrics
# loss, val_loss, iou, val_iou, accuracy, val_accuracy

# %%
import matplotlib.pyplot as plt



plt.figure(figsize=(8, 8))


plt.subplot(3, 1, 1)
plt.plot(evaluator.history['loss'], label='Training Loss')
plt.plot(evaluator.history['val_loss'], label='Validation Loss')


plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.xlabel('epoch-batch')
plt.title('Training and Validation Loss')


plt.subplot(3, 1, 2)
plt.plot(evaluator.history['iou'], label='Training IoU')
plt.plot(evaluator.history['val_iou'], label='Validation IoU')
plt.legend(loc='upper right')
plt.ylabel('Intersection over Union (IoU)')
plt.title('Training and Validation IoU')
plt.xlabel('epoch-batch')

plt.subplot(3, 1, 3)
plt.plot(evaluator.history['acc'], label='Training Accuracy')
plt.plot(evaluator.history['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('epoch-batch')
plt.show()
