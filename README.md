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



## 5. Train models



## 6. Evaluate models



### Sample predictions



## 7. References
AIO2023
