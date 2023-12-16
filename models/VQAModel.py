import torch
import torch.nn as nn

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