import math
import torch 
import torch.nn as nn
from encoder import LowResImageEncoder, TextEncoder

def timestep_embedding(timesteps, dim):
    """
    Create sinusoidal timestep embeddings as in Vaswani et al. and DDPM&#8203;:contentReference[oaicite:4]{index=4}.
    timesteps: Tensor of shape (N,) with diffusion steps.
    dim: Dimension of the output embedding vector.
    """
    # Compute the sinusoidal embeddings
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
    freqs = freqs.to(timesteps.device)
    # Outer product timesteps (N,1) * freqs (1,half) -> (N, half)
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (N, dim) where dim=2*half (if dim is odd, the last term is dropped)
    if dim % 2 == 1:
        # pad one zero if dim is odd
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding  # (N, dim)

class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into a vector of given size via an MLP."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(embed_dim, embed_dim)  # will project sinusoidal embedding to embed_dim
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.act = nn.SiLU()  # activation

    def forward(self, t):
        # t: (N,) int tensor of timesteps
        # Get sinusoidal positional embedding of size embed_dim
        t_emb = timestep_embedding(t, self.embed_dim)  # (N, embed_dim)
        # Two-layer MLP with SiLU non-linearity
        x = self.linear1(t_emb)
        x = self.act(x)
        x = self.linear2(x)
        return x  # (N, embed_dim)

class AdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm (AdaLN) which modulates normalized activations 
    with a conditioning vector via shift and scale (learned per block).
    """
    def __init__(self, normalized_shape, cond_dim):
        super().__init__()
        # LayerNorm without learnable affine parameters (affine False) because we'll modulate adaptively
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        # Linear to map conditioning vector to shift and scale parameters (for two sub-layers: self-attn & cross-attn or cross-attn & MLP)
        # We output 6 parameters: shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp.
        # (If cross-attention is used, we allocate parameters for it in the same sequence as attn.)
        # Actually, we will produce 9 parameters: including cross-attn as well (shift_cross, scale_cross, gate_cross).
        self.linear = nn.Linear(cond_dim, 9 * normalized_shape)  # 9 modulation params for hidden dim
        self.act = nn.SiLU()

    def forward(self, x, cond):
        """
        x: (N, L, D) input sequence to normalize.
        cond: (N, cond_dim) conditioning vector.
        """
        # Normalize x (apply layer norm per token)
        x_normed = self.norm(x)
        # Produce modulation parameters from cond
        cond_out = self.linear(self.act(cond))  # (N, 9*D)
        # Split into 9 chunks for modulation (for attn, cross-attn, mlp respectively)
        # Each chunk is of size D.
        shift_attn, scale_attn, gate_attn, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = torch.split(cond_out, x.size(-1), dim=1)
        # Reshape for broadcasting: (N, 1, D) so it can apply to all tokens in the sequence
        shift_attn = shift_attn.unsqueeze(1); scale_attn = scale_attn.unsqueeze(1); gate_attn = gate_attn.unsqueeze(1)
        shift_cross = shift_cross.unsqueeze(1); scale_cross = scale_cross.unsqueeze(1); gate_cross = gate_cross.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1); scale_mlp = scale_mlp.unsqueeze(1); gate_mlp = gate_mlp.unsqueeze(1)
        # Modulate x for each sub-layer:
        # (We don't immediately apply gating here; gating will be applied to the sub-layer outputs in the block.)
        # We return the parameters needed for each sub-layer.
        return (shift_attn, scale_attn, gate_attn,
                shift_cross, scale_cross, gate_cross,
                shift_mlp, scale_mlp, gate_mlp)

class DiTBlock(nn.Module):
    """A single Transformer block in the diffusion model with dual conditioning (image & text)."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        # Self-Attention for image patches
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Cross-Attention for text conditioning
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # MLP (feed-forward network)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),  # Activation
            nn.Linear(4 * embed_dim, embed_dim)
        )
        # Adaptive layer normalization for this block (affine parameters modulated by cond)
        # We will use AdaLN to modulate three sub-layers: self-attn, cross-attn, and mlp.
        self.adaLN = AdaLayerNorm(normalized_shape=embed_dim, cond_dim=embed_dim)

    def forward(self, x, cond, text_tokens, text_mask=None):
        """
        x: (N, T_img, D) image patch tokens (current latent image representation).
        cond: (N, D) conditioning vector (combined from low-res image embedding and timestep).
        text_tokens: (N, T_txt, D) text conditioning tokens.
        text_mask: (N, T_txt) boolean mask for text padding (True where padded).
        """
        # Obtain modulation parameters from cond
        (shift_attn, scale_attn, gate_attn,
         shift_cross, scale_cross, gate_cross,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN(x, cond)
        # --- Self-Attention sub-layer ---
        # Normalize and modulate x for self-attention
        x_mod = x * (1 + scale_attn) + shift_attn  # apply AdaLN modulation (affine) to x
        # Self-attention on image tokens
        attn_out, _ = self.self_attn(query=x_mod, key=x_mod, value=x_mod)
        # Apply gating and residual connection
        x = x + gate_attn * attn_out
        # --- Cross-Attention sub-layer (image tokens attending to text tokens) ---
        # Normalize and modulate x for cross-attention
        x_mod2 = x * (1 + scale_cross) + shift_cross
        # Perform cross-attention: query = x_mod2 (image tokens), key = value = text_tokens
        # Provide key_padding_mask to ignore padded text positions
        cross_out, _ = self.cross_attn(query=x_mod2, key=text_tokens, value=text_tokens, key_padding_mask=text_mask)
        # Residual connection with gating
        x = x + gate_cross * cross_out
        # --- MLP sub-layer ---
        # Normalize and modulate x for MLP
        x_mod3 = x * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(x_mod3)
        # Residual with gating
        x = x + gate_mlp * mlp_out
        return x

class ConditionalDiT(nn.Module):
    """
    Diffusion Transformer model that generates a high-res image from a low-res image and text prompt.
    Combines a ViT image encoder, a T5 text encoder, and a DiT backbone.
    """
    def __init__(self, image_size=256, patch_size=4, in_channels=3, hidden_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Patch embedding for the (noisy) high-res image input
        # This conv will divide the HxW image into patches of size patch_size x patch_size.
        # out_channels=hidden_dim gives us patch tokens of dimension hidden_dim.
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2  # total number of patch tokens

        # Positional embedding for image patches (learned or fixed). We'll use a fixed sinusoidal embedding for simplicity.
        # Here, we create a parameter for positional embeddings (optionally trainable).
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim), requires_grad=True)
        nn.init.normal_(self.pos_embed, std=0.02)  # small random init

        # Diffusion timestep embedder
        self.time_embedder = TimestepEmbedder(hidden_dim)
        # Low-res image encoder (ViT) to get conditioning vector. We use a linear to project it to hidden_dim if needed.
        self.low_res_encoder = LowResImageEncoder(model_name='vit_base_patch16_224', embed_dim=hidden_dim)
        self.image_cond_proj = nn.Linear(self.low_res_encoder.embed_dim, hidden_dim) if self.low_res_encoder.embed_dim != hidden_dim else nn.Identity()
        # Text encoder (T5) to get text token embeddings for cross-attention
        self.text_encoder = TextEncoder(model_name='t5-base', embed_dim=hidden_dim)

        # Transformer backbone: a sequence of DiTBlocks
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(depth)])
        # Final projection layer to convert transformer output back to image patches
        # If we predict both mean and variance (learn_sigma), out_channels would be 2*in_channels. Here we predict only image (eps).
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)  # project each token to patch pixel values

    def forward(self, high_res_noisy, low_res_img, t, text_prompts):
        """
        high_res_noisy: (N, 3, H, W) input high-res image at noise level t (the noised image).
        low_res_img: (N, 3, H, W) low-res image condition.
        t: (N,) tensor of diffusion timesteps.
        text_prompts: list of N text strings (prompts) or already tokenized format.
        """
        N, C, H, W = high_res_noisy.shape
        assert H == self.image_size and W == self.image_size, "High-res image must be of size {}x{}".format(self.image_size, self.image_size)
        # 1. Encode low-res image to a conditioning embedding
        img_emb = self.low_res_encoder(low_res_img)              # (N, embed_dim)
        img_emb = self.image_cond_proj(img_emb)                  # (N, hidden_dim) ensure it matches hidden_dim
        # 2. Encode text prompts to sequence of embeddings
        text_tokens, text_mask = self.text_encoder(text_prompts) # text_tokens: (N, L, hidden_dim), text_mask: (N, L) boolean
        # 3. Compute diffusion timestep embedding
        t_emb = self.time_embedder(t)                            # (N, hidden_dim)
        # Combine conditioning: sum of time embedding and image embedding (both are (N, hidden_dim))
        cond = t_emb + img_emb                                   # (N, hidden_dim)
        # 4. Patchify the noisy high-res image
        x = self.patch_embed(high_res_noisy)                     # shape (N, hidden_dim, H/patch, W/patch)
        # Rearrange to (N, T_img, hidden_dim) where T_img = number of patches
        x = x.flatten(2).transpose(1, 2)                         # (N, T_img, hidden_dim)
        # Add positional embeddings to patch tokens
        x = x + self.pos_embed                                   # (N, T_img, hidden_dim)
        # 5. Transformer diffusion blocks
        for block in self.blocks:
            x = block(x, cond=cond, text_tokens=text_tokens, text_mask=text_mask)
        # 6. Final layer: normalize and project tokens back to image space
        x = self.final_ln(x)                                     # (N, T_img, hidden_dim)
        x = self.final_proj(x)                                   # (N, T_img, patch_size*patch_size*in_channels)
        # Unpatchify: reshape to image
        out = x.transpose(1, 2).reshape(N, C, H, W)              # (N, 3, H, W)
        return out  # Predicted noise (or image) for the high-res image
