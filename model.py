import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel



# Why is Semantic Segmenter Output Important? 
from transformers.modeling_outputs import SemanticSegmenterOutput
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
# Dinov2 encodes the image through 14x14 patch embeddings
class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)

    self.dinov2 = Dinov2Model(config)
    self.classifier = LinearClassifier(config.hidden_size, 36, 36, config.num_labels)

    

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

      loss_function = torch.nn.CrossEntropyLoss(ignore_index=255) #the mask outputs classes of either 0 or 1. 

      # This fucntion could be removing more than necessary?
      loss = loss_function(logits.squeeze(), labels.squeeze())

    return SemanticSegmenterOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )