import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor

class VisualEncoder(nn.Module):
  def __init__(self):
    super(VisualEncoder, self).__init__()
    self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")

  def forward(self, inputs):
    outputs = self.model(**inputs)

    return outputs.pooler_output