import torchvision.utils as vutils
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T = 1000  # total diffusion steps
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32).to(device)  # (T,)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)  # cumulative product \bar{alpha}_t

def generate_high_res(model, low_res_img, text_prompt, guidance_scale=7.5, steps=50):
    """
    Generate a high-res image from low_res_img conditioned on text_prompt using the diffusion model.
    model: Trained ConditionalDiT model (in eval mode).
    low_res_img: Tensor of shape (1,3,H,W) low-resolution input image.
    text_prompt: String, the text prompt describing the desired output.
    guidance_scale: float, how strongly to apply text guidance (>=1.0).
    steps: number of diffusion timesteps to use for sampling (less than or equal to training T).
    Returns: Tensor of shape (1,3,H,W) of the generated high-res image (pixel values 0-1).
    """
    model.eval()
    device = next(model.parameters()).device
    low_res_img = low_res_img.to(device)
    # Prepare text conditions: actual prompt and empty prompt for guidance
    text_cond = [text_prompt]
    text_uncond = [""]
    # We will use a reduced number of inference steps for speed (e.g., steps=50 vs training T=1000).
    # Define a schedule of timesteps to iterate (e.g., linear spacing or use a predefined scheduler).
    # Here we use linear step spacing for simplicity:
    step_indices = torch.linspace(model.time_embedder.embed_dim - 1, 0, steps, dtype=torch.long, device=device)  # from T-1 to 0
    step_indices = step_indices.tolist()  # list of timesteps to go through
    
    # Start from pure noise at timestep T-1 (highest noise level)
    C, H, W = low_res_img.shape[1], low_res_img.shape[2], low_res_img.shape[3]
    x_t = torch.randn((1, C, H, W), device=device)  # random noise image
    # If we want to start closer to low_res, we could initialize x_t as a mix of low_res and noise (not implemented here).
    
    for t in step_indices:
        t = int(t)
        # Predict noise for both conditional and unconditional inputs
        with torch.no_grad():
            # Conditional prediction (with text)
            pred_noise_cond = model(x_t, low_res_img, torch.tensor([t], device=device), text_cond)
            # Unconditional prediction (empty text)
            pred_noise_uncond = model(x_t, low_res_img, torch.tensor([t], device=device), text_uncond)
        # Merge predictions with guidance
        noise_pred = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
        # Compute the posterior mean x_{t-1} from predicted noise (DDIM update for simplicity: no random noise term)
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        if t > 0:
            alpha_bar_prev = alpha_bars[t-1]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)
        # Predicted x0 from current prediction: x0_pred = (x_t - sqrt(1-alpha_bar_t)*noise_pred) / sqrt(alpha_bar_t)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        # DDIM interpolation: x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred + sqrt(1 - alpha_bar_{t-1}) * noise_pred
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * noise_pred
        x_t = x_prev  # update for next step
    # After loop, x_t at t=0 is the final denoised image
    x_final = x_t.clamp(0, 1)  # ensure in [0,1] range
    return x_final

# Example usage:
# model = ConditionalDiT(...); model.load_state_dict(torch.load("model_checkpoint.pth")); model = model.to(device)
# low_res_img = Image.open("path/to/low_res_input.png").convert("RGB")
# low_res_tensor = transforms.ToTensor()(low_res_img).unsqueeze(0)  # shape (1,3,256,256)
# output_img_tensor = generate_high_res(model, low_res_tensor, "A photo of a cat sitting on a chair", guidance_scale=7.5, steps=50)
# vutils.save_image(output_img_tensor, "output_high_res.png")
