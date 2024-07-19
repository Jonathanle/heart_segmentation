import preprocess
from preprocess import visualize_map
from model import Dinov2ForSemanticSegmentation
from transformers import Dinov2Config
from torch.utils.data import DataLoader
import torch
import albumentations as A
import numpy as np
import os


# Define mapping
id2label={0: "background", 1: "heart"}


def collate_fn(inputs):
    batch = dict()
    batch["image"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["mask"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_image"] = np.array([i[2] for i in inputs])
    batch["segmentation"] = np.array([i[3] for i in inputs])

    return batch


""" Potential Factors Leading to Training Data Loss
Overfitting
Too little Training Data - dO i have enough?
Batch Size? - I need to take more deliberate steps but with more epochs? 
Too little epochs - It seems as though the data might be the biggest problem, maybe I should augment to get more training data. (too little epochs --> more data)
Lack of Normalization of Values
Miscalculation / misbuild of computational architecture
Mistructuring of the Datasets
misconfiguration of the dinov2
Too low learning rate
Misconfiguration of the Loss Metric - why is cross entropy loss important? 
""" 


""" Hidden Objectives
- Not collecting data on how the model does + Using various statistics + metrics of performance
- How Best to Organize Parameters for Easy Freezing of specific parts of the model
- How to Best Engineer the Datasets from raw data + creating automated data preprocessing pipelines.
- How to plot stuff efficiently on matplotlib
- interpreting various tensor / matrix transformations + various operations such as argmax, flatten what they do etc. (intpreting each as a bunch of knobs for querying for a value)
- This kind of code especially in machine might be much more reasonable for unit testing and machine learning. 
- how to unit test for these more "under the surface complex functions"
- how to test the model itself + other strategies for bootstrapping model
"""
# TODO: Determine why iou accuracy is yielding 100 even though the mask is 100% incorrect. 
# Why is huggingface-evaluate important? Why is the iou 100% even though, how is evaluation mechanism actually made. 
# how do i connect huggingface's evaluate iou to the model's logits outputs. 
#TODO: Document what the dataset returns
train_dataset, val_dataset, test_dataset= preprocess.retrieve_heart_dataset()


print(len(train_dataset))
input()

preprocess.visualize_map(train_dataset[0][2], train_dataset[0][3])


# create a thing that Load our data into batches for the model to train
# Cannot have a batch size of 1 due to internal errors of code compresing dimensions.
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size = 2, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(train_dataset, batch_size =2, shuffle=False, collate_fn=collate_fn)




config = Dinov2Config(image_size = 504, patch_size = 14, num_labels=2)
model = Dinov2ForSemanticSegmentation(config).from_pretrained("facebook/dinov2-base", id2label=id2label, num_labels=len(id2label))


# %% [markdown]
# # Freeze the pretrained DINOv2 model
# 
# We only use the DINOv2 backbone as feature extractor, we only train the the linear classification head on the top of the DINOv2 model.

# %%
# TODO: Figure out ways in the future of more cleanly organizing parameters by name in the nn layers + freezing specific types --> more mechanisms? 

for name, param in model.named_parameters():
    if name.startswith("dinov2"): 
        param.requires_grad = False

# %%
# do a forward pass to verify
# put model on GPU (set runtime to GPU in Google Colab)
# TODO: Figure out why this is important for single batch training
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
batch = next(iter(train_dataloader))

print(batch["image"].shape)
print(batch["mask"].shape)
print("------")
outputs = model(pixel_values=batch["image"].to(device), labels=batch["mask"].to(device))
print(outputs.logits.shape)
print(outputs.loss)


import evaluate
metric = evaluate.load("mean_iou")

# %% [markdown]
# Implement the training loop

# %%
from torch.optim import AdamW
from tqdm.auto import tqdm



# Figure out why it is important
learning_rate = 5e-5
# train for at least 20 epochs for a good result
NUM_EPOCH = 200


optimizer = AdamW(model.parameters(), lr=learning_rate)

# put model on GPU (set runtime to GPU in Google Colab)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

train_model = True

if os.path.isfile("./model.pth"): 
    model.load_state_dict(torch.load("./model.pth"))
    train_model = False

if train_model: 
    for epoch in range(NUM_EPOCH):
        model.train()
        epoch_loss = []
        epoch_iou = []
        epoch_acc = []
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(train_dataloader)):
            pixel_values = batch["image"].to(device)
            labels = batch["mask"].to(device)

            # forward pass
            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            # evaluate
            with torch.no_grad():
                predicted = outputs.logits.argmax(dim=1) # Why is this important? ? is it doing? 

                # note that the metric expects predictions + labels as numpy arrays
                metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

            # let's print loss and metrics every 100 batches
            if idx % 100 == 0:
                metrics = metric.compute(num_labels=len(id2label),
                                        ignore_index=0,
                                        reduce_labels=False,
                )
                epoch_loss.append(loss.item())
                epoch_iou.append(metrics["mean_iou"])
                epoch_acc.append(metrics["mean_accuracy"])





    torch.save(model.state_dict(), "./model.pth")

# Eval (Learn Importances and tools that I cna use for later building + fine tuning) 
# Gather tools and libraries + devise interleaving exercises on coding skills for creating evaluation

    
# validation
model.eval()
history = {"loss":[], "val_loss":[], "iou":[], "val_iou":[], "acc":[], "val_acc":[]}
val_loss = []
val_iou = []
val_acc = []
for idx, batch in enumerate(tqdm(test_dataloader)):
    pixel_values = batch["image"].to(device)
    labels = batch["mask"].to(device)
    outputs = model(pixel_values, labels=labels)


    loss = outputs.loss
    with torch.no_grad():
        predicted = outputs.logits.argmax(dim=1) # figure out systematic way of interpret * keeping all things constant find index of max; eliminate dimension then index by everything else to end with argmax. (2, 2, 3) --> (2, 3) (eliminate by dim 1)
    metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
    metrics = metric.compute(num_labels=len(id2label), ignore_index=0, reduce_labels=False)




    visualize_map(batch["original_image"][0], predicted.detach().cpu()[0])
    visualize_map(batch["original_image"][0], batch["mask"][0])
    
    
    # append all metrics
    val_loss.append(loss.item())
    val_iou.append(metrics["mean_iou"])
    val_acc.append(metrics["mean_accuracy"])

    mean_loss = np.mean(np.array(epoch_loss))
    mean_iou = np.mean(np.array(epoch_iou))
    mean_acc = np.mean(np.array(epoch_acc))
    mean_val_loss = np.mean(np.array(val_loss))
    mean_val_iou = np.mean(np.array(val_iou))
    mean_val_acc = np.mean(np.array(val_acc))

    history['acc'].append(mean_acc)
    history['iou'].append(mean_iou)
    history['loss'].append(mean_loss)
    history['val_acc'].append(mean_val_acc)
    history['val_iou'].append(mean_val_iou)
    history['val_loss'].append(mean_loss)
    message = f"==== Epoch: {epoch}, loss: {mean_loss:.3f}, IoU: {mean_iou:.3f}, accuracy: {mean_acc:.3f}, val loss: {mean_val_loss:.3f}, val accuracy: {mean_val_acc:.3f}===="
    print(message)


# %%
import matplotlib.pyplot as plt

loss = history['loss']
iou = history['iou']
accuracy = history['acc']
val_loss = history['val_loss']
val_iou = history['val_iou']
val_accuracy = history['val_acc']

plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.title('Training and Validation Loss')

plt.subplot(3, 1, 2)
plt.plot(iou, label='Training IoU')
plt.plot(val_iou, label='Validation IoU')
plt.legend(loc='upper right')
plt.ylabel('Intersection over Union (IoU)')
plt.title('Training and Validation IoU')
plt.xlabel('epoch')

plt.subplot(3, 1, 3)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.show()