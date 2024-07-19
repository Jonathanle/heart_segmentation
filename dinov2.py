

# %%
from datasets import load_dataset
import preprocess

#dataset
dataset = load_dataset("EduardoPacheco/FoodSeg103")


# %%


# %% [markdown]
# Encode the idx to label dict

# %%
id2label = {
    0: "background",
    1: "candy",
    2: "egg tart",
    3: "french fries",
    4: "chocolate",
    5: "biscuit",
    6: "popcorn",
    7: "pudding",
    8: "ice cream",
    9: "cheese butter",
    10: "cake",
    11: "wine",
    12: "milkshake",
    13: "coffee",
    14: "juice",
    15: "milk",
    16: "tea",
    17: "almond",
    18: "red beans",
    19: "cashew",
    20: "dried cranberries",
    21: "soy",
    22: "walnut",
    23: "peanut",
    24: "egg",
    25: "apple",
    26: "date",
    27: "apricot",
    28: "avocado",
    29: "banana",
    30: "strawberry",
    31: "cherry",
    32: "blueberry",
    33: "raspberry",
    34: "mango",
    35: "olives",
    36: "peach",
    37: "lemon",
    38: "pear",
    39: "fig",
    40: "pineapple",
    41: "grape",
    42: "kiwi",
    43: "melon",
    44: "orange",
    45: "watermelon",
    46: "steak",
    47: "pork",
    48: "chicken duck",
    49: "sausage",
    50: "fried meat",
    51: "lamb",
    52: "sauce",
    53: "crab",
    54: "fish",
    55: "shellfish",
    56: "shrimp",
    57: "soup",
    58: "bread",
    59: "corn",
    60: "hamburg",
    61: "pizza",
    62: "hanamaki baozi",
    63: "wonton dumplings",
    64: "pasta",
    65: "noodles",
    66: "rice",
    67: "pie",
    68: "tofu",
    69: "eggplant",
    70: "potato",
    71: "garlic",
    72: "cauliflower",
    73: "tomato",
    74: "kelp",
    75: "seaweed",
    76: "spring onion",
    77: "rape",
    78: "ginger",
    79: "okra",
    80: "lettuce",
    81: "pumpkin",
    82: "cucumber",
    83: "white radish",
    84: "carrot",
    85: "asparagus",
    86: "bamboo shoots",
    87: "broccoli",
    88: "celery stick",
    89: "cilantro mint",
    90: "snow peas",
    91: "cabbage",
    92: "bean sprouts",
    93: "onion",
    94: "pepper",
    95: "green beans",
    96: "French beans",
    97: "king oyster mushroom",
    98: "shiitake",
    99: "enoki mushroom",
    100: "oyster mushroom",
    101: "white button mushroom",
    102: "salad",
    103: "other ingredients"
}

# %% [markdown]
# Visualize the images and masks

# %%
import numpy as np
import matplotlib.pyplot as plt

# map every class to a random color
id2color = {k: list(np.random.choice(range(256), size=3)) for k,v in id2label.items()}

# %% [markdown]
# The image and the mask are both PIL image objects with the identical height and width, the image has 3 channels, the mask has only 1 channel.

# %%
"""
def visualize_map(image, segmentation_map):
    segmentation_map = np.array(segmentation_map)
    color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
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



"""
# %%
idx = 1
image = dataset['train'][idx]['image']
segmentation_map = dataset['train'][idx]['label']
class_idx = dataset['train'][idx]['classes_on_image']

#preprocess.visualize_map(image, segmentation_map)

# %% [markdown]
# Prepare a PyTorch dataset with albumentations as the augmentation tool for training
# 
# albumentations is pre-installed on colab, need to use pip to install albumentations on other platform
# 
# ! pip install albumentations

# %%
import torch
import albumentations as A
from torch.utils.data import Dataset

# %%
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


# %%
import albumentations as A
# add the color normalization parameters - based on the ImageNet dataset parameters

ADE_MEAN = [0.485, 0.456, 0.406]
ADE_STD = [0.229, 0.224, 0.225]

# %%
train_transform = A.Compose([
    # don't using crop
    A.Resize(width=448, height=448),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ], is_check_shapes=False)


val_transform = A.Compose([
    A.Resize(width=448, height=448),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ], is_check_shapes=False)


# %% [markdown]
# Create the segmentation datasets

# %%
train_dataset = SegmentationDataset(dataset["train"], transform=train_transform)
val_dataset = SegmentationDataset(dataset["validation"], transform=val_transform)

# %% [markdown]
# verify the shape

# %%
pixel_values, target, original_image, original_segmentation_map = train_dataset[3]
print(pixel_values.shape) # make sure it is channel first
print(target.shape)

# %% [markdown]
# Create PyTorch dataloaders to get data batches

# %%
from torch.utils.data import DataLoader

BATCH_SIZE = 10 # good for T4 GPU

def collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_images"] = [i[2] for i in inputs]
    batch["original_segmentation_maps"] = [i[3] for i in inputs]

    return batch

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# %% [markdown]
# Verify the output shape

# %%


batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v,torch.Tensor):
    print(k,v.shape)

# %% [markdown]
# # Define the segmentation model
# 
# *   Use the DINOv2 as the backbone - DINOv2 outputs patch embeddings, with a patch resolution of 14, eventually we will get $(448/14)^{2} = 1024$ patches. The final output is (batch_size, number_of_patches, hidden_size), where batch_size=4, number_of_patches=1024, hidden_size=768 (embedding dimension)
# *   To restore the feature map for a 2D image, we take $\sqrt{1024}=32$, to reshape the tensor to (4,32,32,768)
# *   Use $1\times1$ Conv to reshape the output - (batch_size, num_labels, height, width)
# 
# 

# %%
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput

# %%
class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1)) # 1-by-1 Conv

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2) # channel first

        return self.classifier(embeddings)

# %%
# define a DINOv2 based semantic segmentation model inherited from the Dinov2PretrainedModel

class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)

    self.dinov2 = Dinov2Model(config)
    self.classifier = LinearClassifier(config.hidden_size, 32, 32, config.num_labels)

  def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
    # use frozen features from pretrained DINOv2
    outputs = self.dinov2(pixel_values,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions)
    # if we want to compute the attention head map, set output_attentions = True

    # get the patch embeddings - so we remove the CLS token at the 1st channel of the hidden state
    # hidden output - (batch,CLS_token+features,height, width)
    patch_embeddings = outputs.last_hidden_state[:,1:,:]

    # convert to logits and upsample to the size of the pixel values
    logits = self.classifier(patch_embeddings)
    logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

    loss = None
    if labels is not None:
      # important: we're going to use 0 here as ignore index instead of the default -100
      # as we don't want the model to learn to predict background
      # use cross entropy as the loss objective

      loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)
      loss = loss_function(logits.squeeze(), labels.squeeze())

    return SemanticSegmenterOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

# %%
# instantiate the DINOv2 model
model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", id2label=id2label, num_labels=len(id2label))

# %% [markdown]
# # Freeze the pretrained DINOv2 model
# 
# We only use the DINOv2 backbone as feature extractor, we only train the the linear classification head on the top of the DINOv2 model.

# %%
# freeze all the layers with prefix 'dinov2'

for name, param in model.named_parameters():
  if name.startswith("dinov2"):
    param.requires_grad = False

# %%
# do a forward pass to verify
# put model on GPU (set runtime to GPU in Google Colab)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
batch = next(iter(train_dataloader))
outputs = model(pixel_values=batch["pixel_values"].to(device), labels=batch["labels"].to(device))
print(outputs.logits.shape)
print(outputs.loss)

# %% [markdown]
# We use the IoU as the metric. Intersection over Union (IoU) is the area of overlap between the predicted segmentation and the ground truth divided by the area of union between the predicted segmentation and the ground truth.

# %%
import evaluate
metric = evaluate.load("mean_iou")

# %% [markdown]
# Implement the training loop

# %%
from torch.optim import AdamW
from tqdm.auto import tqdm

learning_rate = 5e-5
# train for at least 20 epochs for a good result ()
epochs = 1

optimizer = AdamW(model.parameters(), lr=learning_rate)

# set the model to training mode
model.train()

# %%
from torch.optim import AdamW
from tqdm.auto import tqdm

history = {"loss":[], "val_loss":[], "iou":[], "val_iou":[], "acc":[], "val_acc":[]}
learning_rate = 5e-5
epochs = 1

optimizer = AdamW(model.parameters(), lr=learning_rate)

# put model on GPU (set runtime to GPU in Google Colab)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(epochs):
  model.train()
  epoch_loss = []
  epoch_iou = []
  epoch_acc = []
  print("Epoch:", epoch)
  for idx, batch in enumerate(tqdm(train_dataloader)):
      pixel_values = batch["pixel_values"].to(device)
      labels = batch["labels"].to(device)

      # forward pass
      outputs = model(pixel_values, labels=labels)
      loss = outputs.loss

      loss.backward()
      optimizer.step()

      # zero the parameter gradients
      optimizer.zero_grad()

      # evaluate
      with torch.no_grad():
        predicted = outputs.logits.argmax(dim=1)

        # note that the metric expects predictions + labels as numpy arrays
        metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

      # let's print loss and metrics every 100 batches
      if idx % 5 == 0:
        metrics = metric.compute(num_labels=len(id2label),
                                ignore_index=0,
                                reduce_labels=False,
        )
        epoch_loss.append(loss.item())
        epoch_iou.append(metrics["mean_iou"])
        epoch_acc.append(metrics["mean_accuracy"])
        print(epoch_iou[-1])



  # validation
  model.eval()
  val_loss = []
  val_iou = []
  val_acc = []
  for idx, batch in enumerate(tqdm(val_dataloader)):
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(pixel_values, labels=labels)
    loss = outputs.loss
    with torch.no_grad():
      predicted = outputs.logits.argmax(dim=1)
    metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
    metrics = metric.compute(num_labels=len(id2label), ignore_index=0, reduce_labels=False)
    val_loss.append(loss.item())
    val_iou.append(metrics["mean_iou"])
    val_acc.append(metrics["mean_accuracy"])

    mean_loss = np.mean(np.array(epoch_loss))
    mean_iou = np.mean(np.array(epoch_iou))
    mean_acc = np.mean(np.array(epoch_acc))
    mean_val_loss = np.mean(np.array(val_loss))
    mean_val_iou = np.mean(np.array(val_iou))
    mean_val_acc = np.mean(np.array(val_acc))


    # shouldnt I append this during the traainig? to monitor convergence? 
    history['acc'].append(mean_acc)
    history['iou'].append(mean_iou)
    history['loss'].append(mean_loss)
    history['val_acc'].append(mean_val_acc)
    history['val_iou'].append(mean_val_iou)
    history['val_loss'].append(mean_loss)
    message = f"==== Epoch: {epoch}, loss: {mean_loss:.3f}, IoU: {mean_iou:.3f}, accuracy: {mean_acc:.3f}, val loss: {mean_val_loss:.3f}, val accuracy: {mean_val_acc:.3f}===="
    print(message)


from PIL import Image
test_image = dataset["validation"][6]["image"]
test_image


pixel_values = val_transform(image=np.array(test_image))["image"]
pixel_values = torch.tensor(pixel_values)
pixel_values = pixel_values.permute(2,0,1).unsqueeze(0) # convert to (batch_size, num_channels, height, width)
print(pixel_values.shape)

upsampled_logits = torch.nn.functional.interpolate(outputs.logits,
                                                   size=test_image.size[::-1],
                                                   mode="bilinear", align_corners=False)
predicted_map = upsampled_logits.argmax(dim=1)

torch.save(model, "model.pth")
preprocess.visualize_map(np.array(test_image), predicted_map.squeeze().cpu())


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




