import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import open_clip

class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_epochs = 10
    learning_rate = 4e-8
    weight_decay = 0.001
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    loss_func = nn.functional.cross_entropy
    img_size = 224
    logit_scale = 100

# Instantite model and preprocessors
model, _, preprocess = open_clip.create_model_and_transforms(model_name=CFG.model_name, pretrained=CFG.pretrained)
tokenizer = open_clip.get_tokenizer(CFG.model_name)

# Load and preprocess dataset
train_dataset = datasets.CocoCaptions(
    root='./coco/images/val2017/',
    annFile='./coco/annotations/captions_val2017.json',
    transform=preprocess,
    target_transform=lambda texts: tokenizer(texts[0])
    )
train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, drop_last=True)

optimizer = torch.optim.Adam(model.parameters(), lr = CFG.learning_rate,
                             weight_decay=CFG.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=len(train_loader)*CFG.num_epochs)

# Check device
print(f'Using device {CFG.device}.')

epoch_losses = []

# Train loop
model.to(CFG.device).train()
for epoch in range(1, CFG.num_epochs+1):
    
    print(f"Epoch {epoch}/{CFG.num_epochs}:")
    iter_losses = []

    pbar = tqdm(train_loader, total = len(train_loader))
    for idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        images, texts = batch
        images = images.to(CFG.device)
        texts = texts.to(CFG.device)

        texts = torch.flatten(texts, start_dim=0, end_dim=1)

        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        
        logits_per_image = CFG.logit_scale * image_features @ text_features.T
        logits_per_text = CFG.logit_scale * text_features @ image_features.T
        target = torch.arange(CFG.batch_size).to(CFG.device)

        loss_i = CFG.loss_func(logits_per_image, target)
        loss_t = CFG.loss_func(logits_per_text, target)
        loss = (loss_i + loss_t) / 2

        iter_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pbar.set_description(f"Loss: {loss.item():.4f}")

    # Epoch loss
    epoch_losses.append(np.mean(iter_losses))

    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, f'./model_checkpoints/val2017_epoch_{epoch}.pt')
    
# Log epoch losses
with open('./losses/epoch_losses.txt', 'w') as f:
    f.write("Epoch Losses:\n")
    for i, loss in enumerate(epoch_losses):
        f.write(f'Epoch {i+1}/{CFG.num_epochs}: {loss}\n')
