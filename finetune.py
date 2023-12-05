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
    num_epochs = 30
    learning_rate = 5e-5
    weight_decay = 0.0
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    loss_func = nn.functional.cross_entropy
    img_size = 224

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

        # image_features = model.encode_image(images)
        # text_features = model.encode_text(texts)
        
        image_embedding, text_embedding, _ = model(images, texts)
        
        logits = image_embedding @ text_embedding.T
        
        target = torch.arange(CFG.batch_size).to(CFG.device)

        loss_i = CFG.loss_func(logits, target)
        loss_t = CFG.loss_func(logits.T, target)
        loss = (loss_i + loss_t) / 2

        iter_losses.append(loss)

        loss.backward()
        optimizer.step()
        
        break

    # Epoch loss
    epoch_losses.append(torch.mean(torch.tensor(iter_losses)).item())

    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, f'./model_checkpoints/val2017_epoch_{epoch}.pt')
    
    break

# Plot losses
dim = np.arange(1, CFG.num_epochs, 1)
plt.figure()
plt.plot(epoch_losses, label="Train")
plt.xticks(dim)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("clip/val2017_losses.png")
plt.clf()
