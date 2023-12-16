import torch
import torch.nn as nn

from transformers import ViTModel, ViTImageProcessor
from transformers import AutoTokenizer, RobertaModel

class Classifier(nn.Module):
  def __init__(self, input_size=768*2, hidden_size=512, n_layers=1, dropout_prob=0.2, n_classes=2):
    super(Classifier, self).__init__()
    self.lstm = nn.LSTM(
        input_size, hidden_size,
        num_layers=n_layers,
        batch_first=True,
        bidirectional=True
    )

    self.dropout = nn.Dropout(dropout_prob)
    self.fc1 = nn.Linear(hidden_size*2, n_classes)

  def forward(self, x):
    x, (hidden, cell) = self.lstm(x)
    x = self.dropout(x)
    x = self.fc1(x)

    return x

class TextEncoder(nn.Module):
  def __init__(self):
    super(TextEncoder, self).__init__()
    self.model = RobertaModel.from_pretrained("roberta-base")

  def forward(self, inputs):
    outputs = self.model(**inputs)

    return outputs.pooler_output

class VisualEncoder(nn.Module):
  def __init__(self):
    super(VisualEncoder, self).__init__()
    self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")

  def forward(self, inputs):
    outputs = self.model(**inputs)

    return outputs.pooler_output

class VQAModel(nn.Module):
  def __init__(self, visual_encoder, text_encoder, classifier):
    super(VQAModel, self).__init__()
    self.visual_encoder = visual_encoder
    self.text_encoder = text_encoder
    self.classifier = classifier

  def forward(self, image, question):
    text_out = self.text_encoder(question)
    image_out = self.visual_encoder(image)
    x = torch.cat((text_out, image_out), dim=1)
    x = self.classifier(x)

    return x

  def freeze(self, visual=True, textual=True, cla=False):
    if visual:
      for n,p in self.visual_encoder.named_parameters():
        p.requires_grad = False
    if textual:
      for n,p in self.text_encoder.named_parameters():
        p.requires_grad = False
    if cla:
      for n,p in self.classifier.named_parameters():
        p.requires_grad = False