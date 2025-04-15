import torch.optim as optim
import torch
from data import COCODataset
from dit import ConditionalDiT
import torchvision

# Hyperparameters
num_epochs = 100
learning_rate = 1e-4
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize dataset and dataloader
train_dataset = COCODataset(image_dir="../data/coco/images/train2017", caption_file="../data/coco/annotations/captions_train2017.json", image_size=256, low_res_factor=4)
val_dataset   = COCODataset(image_dir="../data/coco/images/val2017", caption_file="../data/coco/annotations/captions_val2017.json", image_size=256, low_res_factor=4, random_flip=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize model, optimizer, and loss
model = ConditionalDiT(image_size=256, patch_size=4, in_channels=3, hidden_dim=768, depth=12, num_heads=12)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Diffusion hyperparameters
T = 1000  # total diffusion steps
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32).to(device)  # (T,)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)  # cumulative product \bar{alpha}_t

def diffusion_loss(model, low_res, high_res, captions):
    """
    Compute diffusion loss (MSE) for a batch.
    """
    N = high_res.size(0)
    # Sample random diffusion timesteps for each sample
    t = torch.randint(0, T, (N,), device=device, dtype=torch.long)
    # Sample random noise
    noise = torch.randn_like(high_res, device=device)
    # Calculate alpha_bar_t for each sample and reshape for broadcasting
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)  # shape (N,1,1,1)
    # Produce the noisy image for timestep t: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*noise
    noisy_high_res = torch.sqrt(alpha_bar_t) * high_res + torch.sqrt(1 - alpha_bar_t) * noise
    # Randomly drop text condition (CFG technique)
    # Create a modified captions list where some are replaced with "" (empty) for unconditional training
    captions_cond = []
    for cap in captions:
        if torch.rand(1).item() < 0.1:  # 10% chance to drop
            captions_cond.append("")    # empty prompt (unconditional)
        else:
            captions_cond.append(cap)
    # Run the model to predict noise
    noise_pred = model(noisy_high_res, low_res, t, captions_cond)
    # If model outputs both image and sigma, we'd take the image part. Here model outputs predicted noise image.
    # Compute MSE loss between predicted noise and true noise
    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    return loss

# Training loop

print("Starting training...")
print (f"Using device: {device}")
print (f"Number of epochs: {num_epochs}")
print (f"Batch size: {batch_size}")
print (f"Learning rate: {learning_rate}")
print (f"Training on {len(train_dataset)} samples")
print (f"Validation on {len(val_dataset)} samples")
print (f"Training on {len(train_loader)} batches")
print (f"Validation on {len(val_loader)} batches")


for epoch in range(num_epochs):
    print (f"Training Epoch {epoch+1}/{num_epochs}")
    model.train()
    total_loss = 0.0
    for batch_idx, (low_res, high_res, captions) in enumerate(train_loader):
        low_res = low_res.to(device)
        high_res = high_res.to(device)
        # Compute loss
        loss = diffusion_loss(model, low_res, high_res, captions)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Print training status periodically
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} Iter {batch_idx}: Loss = {loss.item():.4f}", flush=True)
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} completed. Average training loss: {avg_loss:.4f}")
    
    # save model to ../checkpoints
    torch.save(model.state_dict(), f"../checkpoints/epoch_{epoch}.pth")
    
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for low_res, high_res, captions in val_loader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            # Use captions as-is for validation (no dropout)
            loss = diffusion_loss(model, low_res, high_res, captions)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Validation loss: {val_loss:.4f}")
    # (Optional) Generate a few sample outputs for qualitative validation
    if epoch % 10 == 0:
        sample_low_res, sample_high_res, sample_captions = next(iter(val_loader))
        sample_low_res = sample_low_res.to(device)
        sample_high_res = sample_high_res.to(device)
        # Generate a denoised output (taking t=0 for simplicity here, or performing a few diffusion steps)
        # Here we'll just do one step denoising (which is not the full sampling).
        t = torch.zeros(sample_low_res.size(0), dtype=torch.long, device=device)  # t=0 (predict no noise)
        pred_noise = model(sample_high_res, sample_low_res, t, sample_captions)
        # Predicted clean image x0 approximation: x0 ~ (x_t - sqrt(1-alpha_bar_t)*pred_noise) / sqrt(alpha_bar_t), at t where alpha_bar ~ 1 for t=0.
        # For t=0, alpha_bar=1, so x0 ≈ sample_high_res - 0*pred_noise = sample_high_res (not useful).
        # In practice, we would run a full sampling loop to get output images. This is a placeholder.
        recon = torch.clamp(sample_high_res - pred_noise, 0.0, 1.0)  # rough estimate of denoised image
        # Save or visualize recon vs low_res vs high_res as needed.
        # For example, save the first sample:
        torchvision.utils.save_image(torch.cat([sample_low_res[:1], recon[:1], sample_high_res[:1]], dim=0), f"validation_sample_epoch{epoch}.png", nrow=3)
