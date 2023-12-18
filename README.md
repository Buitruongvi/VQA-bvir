# Visual Question Answering
## 1. Quick insight about my method
This is a project aimed at applying the Transformer model to address the Visual Question Answering problem. With a simple approach, it involves using [ViT](https://huggingface.co/google/vit-base-patch16-224) to extract features from the input image and the [RoBERTa](https://huggingface.co/roberta-base) model to process the input question. Subsequently, these features are connected through an LSTM model, and finally, a classifier is employed to predict either "yes" or "no."
<img width="1062" alt="image" src="https://github.com/Buitruongvi/VQA-bvir/assets/49474873/e11fe7e5-5d61-48a3-9acd-cff59585eb62">
## 2. Requirements
- pytorch
- timm
- transformers

## 3. Dataset
I'm using the COCO-VQA dataset, and you can download it [here](https://drive.google.com/file/d/1kc6XNqHZJg27KeBuoAoYj70_1rT92191/view)
![image](https://github.com/Buitruongvi/VQA-bvir/assets/49474873/39efffd2-2a3c-4b61-b810-4cc6292247d4)

## 4. Pretrained VQA models
- Vision Transformer (base-sized model): Visual extractor
```model = ViTModel.from_pretrained("google/vit-base-patch16-224")```
- RoBERTa base model: Text extractor
```model = RobertaModel.from_pretrained("roberta-base")```

## 5. Train models
## VQAModel:

The `VQAModel` is a PyTorch neural network model designed for Visual Question Answering (VQA) tasks. It consists of three main components:

- **Visual Encoder:** Responsible for encoding visual information from images.
- **Text Encoder:** Handles the encoding of textual information from questions.
- **Classifier:** A classifier that combines the outputs of the visual and text encoders for making predictions.

#### Initialization:

```python
class VQAModel(nn.Module):
    def __init__(self, visual_encoder, text_encoder, classifier):
        super(VQAModel, self).__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.classifier = classifier
```
#### Forward Pass:
```python
    def forward(self, image, question):
        text_out = self.text_encoder(question)
        image_out = self.visual_encoder(image)
        x = torch.cat((text_out, image_out), dim=1)
        x = self.classifier(x)
        return x
```
#### Parameter Freezing:
```python
    def freeze(self, visual=True, textual=True, cla=False):
        if visual:
            for n, p in self.visual_encoder.named_parameters():
                p.requires_grad = False
        if textual:
            for n, p in self.text_encoder.named_parameters():
                p.requires_grad = False
        if cla:
            for n, p in self.classifier.named_parameters():
                p.requires_grad = False
```

## 6. Evaluate models



### Sample predictions



## 7. References
AIO2023
