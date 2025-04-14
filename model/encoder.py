import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer
import timm

class LowResImageEncoder(nn.Module):
    """Encodes a low-resolution image into a global embedding using a ViT backbone."""
    def __init__(self, model_name='vit_base_patch16_224', embed_dim=768):
        super().__init__()
        # Use a pretrained ViT model (from timm) and take the CLS token as image embedding
        self.vit = timm.create_model(model_name, pretrained=True)
        # Freeze ViT if desired (can be fine-tuned as well)
        # for p in self.vit.parameters():
        #     p.requires_grad = False
        # The embed_dim should match the ViT's output embed dim (e.g., 768 for vit_base_patch16_224)
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        x: low-res image tensor of shape (N, 3, H, W), expected H=W=256.
        Returns: image embedding tensor of shape (N, embed_dim).
        """
        # The timm ViT model returns the class token embedding by default when trained for classification.
        # We ensure the model is in evaluation mode for inference of features.
        self.vit.eval()
        with torch.no_grad():
            # timm ViT forward will give class token embedding if we access vit(x) directly (if it's set up for classification).
            img_emb = self.vit(x)  # shape (N, embed_dim)
        return img_emb  # global image embedding

class TextEncoder(nn.Module):
    """Encodes a text prompt into a sequence of embeddings using a T5 encoder."""
    def __init__(self, model_name='t5-base', embed_dim=768):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        # Optionally freeze the text encoder initially to retain pre-trained language features
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        # Project text embeddings to the diffusion model's hidden size if needed
        self.proj = None
        if embed_dim != self.encoder.config.d_model:
            # Project to match the diffusion model dimension (e.g., DiT hidden size)
            self.proj = nn.Linear(self.encoder.config.d_model, embed_dim)

    def forward(self, texts):
        """
        texts: List of strings (length N) or a list of already tokenized inputs.
        Returns: text_emb (N, L, embed_dim) and attn_mask (N, L) for cross-attention.
        """
        # Tokenize the batch of text prompts
        enc = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = enc.input_ids.to(self.encoder.device)
        attn_mask = enc.attention_mask.to(self.encoder.device)
        # Encode text to get hidden states
        outputs = self.encoder(input_ids=input_ids, attention_mask=attn_mask)
        text_emb = outputs.last_hidden_state  # (N, L, d_model)
        if self.proj is not None:
            text_emb = self.proj(text_emb)    # (N, L, embed_dim)
        # Convert attention mask to boolean mask for later use (True for valid tokens, False for pad)
        # We'll use this mask in cross-attention to ignore padding positions.
        key_padding_mask = attn_mask == 0  # bool mask shape (N, L), True where padding
        return text_emb, key_padding_mask
