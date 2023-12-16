import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel

class TextEncoder(nn.Module):
  def __init__(self):
    super(TextEncoder, self).__init__()
    self.model = RobertaModel.from_pretrained("roberta-base")

  def forward(self, inputs):
    outputs = self.model(**inputs)

    return outputs.pooler_output