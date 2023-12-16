import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor
from transformers import AutoTokenizer

from modules.trainer import fit
from modules.load_data import LoadData
from modules.dataloader import VQADataset
from modules.classifier import Classifier
from modules.visual_extractor import VisualEncoder
from modules.text_extractor import TextEncoder
from models.VQAModel import VQAModel

def main():
    train_data = LoadData('data/vqa_coco_dataset/vaq2.0.TrainImages.txt')
    val_data = LoadData('data/vqa_coco_dataset/vaq2.0.DevImages.txt')
    test_data = LoadData('data/vqa_coco_dataset/vaq2.0.TestImages.txt')

    classes = set([sample['answer'] for sample in train_data])
    classes_to_idx = {
        cls_name: idx for idx, cls_name in enumerate(classes)
    }
    idx_to_classes = {
        idx: cls_name for idx, cls_name in enumerate(classes)
    }

    img_feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    text_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = VQADataset(
        train_data,
        classes_to_idx,
        img_feature_extractor,
        text_tokenizer,
        device
    )

    val_dataset = VQADataset(
        val_data,
        classes_to_idx,
        img_feature_extractor,
        text_tokenizer,
        device
    )

    test_dataset = VQADataset(
        test_data,
        classes_to_idx,
        img_feature_extractor,
        text_tokenizer,
        device
    )

    train_batch_size = 128
    test_batch_size = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_batch_size,
        shuffle=False
    )

    n_classes = len(classes)
    hidden_size = 1024
    n_layers = 1
    dropout_prod = 0.2

    text_encoder = TextEncoder().to(device)
    visual_encoder = VisualEncoder().to(device)
    classifier = Classifier().to(device)

    model = VQAModel(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        classifier=classifier
    ).to(device)

    model.freeze()

    lr = 1e-2
    epochs = 50
    scheduler_step_size = epochs * 0.6
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=0.1
    )

    train_losses, val_losses = fit(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs
    )

if __name__ == '__main__':
    main()

