import torch
import torch.nn as nn
from typing import Optional


class PositionalEncodingLearnable(nn.Module):
    """ Learnable positional encoding for transformer inputs.
    - Works with variable sequence lengths up to `max_len`.
    - Expects input of shape (B, S, E): batch, sequence length, embedding dim.
    - Adds a learned embedding for each position index [0..S-1].
    """
    def __init__(self, embed_dim: int, max_len: int = 30, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Embedding table for positions: (max_len, embed_dim)
        self.position_embeddings = nn.Embedding(max_len, embed_dim)

        # Initialize embeddings uniformly in [-0.1, 0.1]
        nn.init.uniform_(self.position_embeddings.weight, -0.1, 0.1)

        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (B, S, E)
        Returns:
            x + positional embeddings of shape (B, S, E)
        """
        B, S, E = x.shape

        # Guard against sequences longer than supported
        if S > self.max_len:
            raise ValueError(
                f"Sequence length {S} exceeds max_len {self.max_len}."
            )

        # Position indices [0..S-1], on the same device as x
        positions = torch.arange(S, device=x.device)

        # Look up embeddings for these positions: (S, E)
        pos_emb = self.position_embeddings(positions)

        # Expand across the batch dimension: (B, S, E)
        pos_emb = pos_emb.unsqueeze(0).expand(B, S, E)

        # Add to input embeddings and apply dropout
        return self.dropout(x + pos_emb)




class CNN_Transformer(nn.Module):
    """
    CNN + Transformer classifier (batch-first throughout).

    Backwards-compatible with older configs:
    - Accepts `cnn_embed_dim` (projection size right after CNN).
    - Ignores legacy `seq_len` / `seq_steps` if present.
    - Transformer d_model is `transformer_embed_dim` (can differ from cnn_embed_dim).
    """

    def __init__(
        self,
        nr_channels: int = 1,
        transformer_embed_dim: int = 16,   # d_model for the Transformer
        num_heads: int = 2,
        num_layers: int = 1,
        nr_classes: int = 6,
        max_seq_len: int = 64,
        dropout: float = 0.3,
        cnn_out_channels: int = 64,         # last CNN channel count before pooling
        cnn_width_mult: int = 1,

        # ---- legacy / optional knobs ----
        cnn_embed_dim: Optional[int] = None,  # projection size after CNN; default = transformer_embed_dim
        seq_len: Optional[int] = None,       
        seq_steps: Optional[int] = None,      
        **_ignored,                            # swallow any other unexpected args
    ) -> None:
        super().__init__()

        # Resolve legacy args
        if cnn_embed_dim is None:
            cnn_embed_dim = transformer_embed_dim  # default: match transformer size

        # -------------------
        # 1) CNN backbone
        # -------------------
        c1 = 16 * cnn_width_mult
        c2 = 32 * cnn_width_mult
        c3 = cnn_out_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(nr_channels, c1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B*, c3, 1, 1)
        )

        # 2) Projections: CNN vector (c3) -> cnn_embed_dim (your desired CNN embedding size)
        self.proj_cnn = nn.Linear(c3, cnn_embed_dim)

        # Then (optionally) cnn_embed_dim -> transformer_embed_dim (if they differ)
        self.proj_to_tx = (nn.Identity() if cnn_embed_dim == transformer_embed_dim else nn.Linear(cnn_embed_dim, transformer_embed_dim))

        # 3) Positional encoding for (B, S, transformer_embed_dim)
        self.positional_encoding = PositionalEncodingLearnable(
            embed_dim=transformer_embed_dim,
            max_len=max_seq_len,
            dropout=dropout,
        )

        # 4) Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * transformer_embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 5) Classification head
        self.fc = nn.Linear(transformer_embed_dim, nr_classes)

    def _cnn_vec(self, x: torch.Tensor, debug_shapes: bool = False) -> torch.Tensor:
        """
        Vectorized CNN pass over frames.

        Args:
            x: 4D (B, C, H, W) or 5D (B, S, C, H, W)
        Returns:
            z_cnn: (B, S, cnn_embed_dim)  -- after self.proj_cnn
        """

        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, C, H, W)
        assert x.dim() == 5, f"Expected 5D input after adjustment, got {x.shape}"
        B, S, C, H, W = x.shape

        x_flat = x.reshape(B * S, C, H, W)
        if debug_shapes:
            print(f"[CNN] input: {x_flat.shape} (B*S, C, H, W)")

        f = self.cnn(x_flat)        # (B*S, c3, 1, 1)
        f = f.view(B * S, -1)       # (B*S, c3)
        z = self.proj_cnn(f)        # (B*S, cnn_embed_dim)
        z = z.view(B, S, -1)        # (B, S, cnn_embed_dim)

        if debug_shapes:
            print(f"[CNN] features (post-proj_cnn): {z.shape} (B, S, E_cnn)")
        return z

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_latent: bool = False,
        return_latent_per_seq: bool = False,
        return_cnn_latent: bool = False,  # returns (B, S, E_cnn)
        use_transformer: Optional[bool] = None,
        debug_shapes: bool = False,
    ):
        # 1) CNN -> (B, S, E_cnn)
        z_cnn = self._cnn_vec(x, debug_shapes=debug_shapes)
        B, S, E_cnn = z_cnn.shape

        # 2) Project to transformer size if needed -> (B, S, E_tx)
        z = self.proj_to_tx(z_cnn)
        if debug_shapes and isinstance(self.proj_to_tx, nn.Linear):
            print(f"[PROJ] to transformer dim: {z.shape} (B, S, E_tx)")

        # Optional early return of CNN features (pre-Transformer)
        if return_cnn_latent:
            return z_cnn  # (B, S, E_cnn)

        # Decide whether to run the transformer
        if use_transformer is None:
            do_tx = S > 1
        else:
            do_tx = bool(use_transformer)

        # 3) PE + Transformer (batch-first)
        if do_tx:
            z_pe = self.positional_encoding(z)  # (B, S, E_tx)
            if debug_shapes:
                print(f"[PE] after PE: {z_pe.shape} (B, S, E_tx)")
            h = self.transformer_encoder(z_pe)  # (B, S, E_tx)
            if debug_shapes:
                print(f"[TX] output: {h.shape} (B, S, E_tx)")
        else:
            if debug_shapes:
                msg = "Transformer skipped (S==1)" if S == 1 else "Transformer forced off"
                print(f"[TX] {msg}")
            h = z

        if return_latent_per_seq:
            return h  # (B, S, E_tx)

        # 4) Pool over sequence
        pooled = h.mean(dim=1)  # (B, E_tx)
        if debug_shapes:
            print(f"[POOL] pooled: {pooled.shape} (B, E_tx)")

        if return_latent:
            return pooled

        # 5) Classify
        logits = self.fc(pooled)  # (B, nr_classes)
        if debug_shapes:
            print(f"[CLS] logits: {logits.shape} (B, nr_classes)")
        return logits




# class PositionalEncodingLearnable(nn.Module):
#     def __init__(self, embed_dim, seq_len=10, dropout=0.1):
#         super(PositionalEncodingLearnable, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
        
#         # Learnable positional embeddings
#         self.position_embeddings = nn.Embedding(seq_len, embed_dim)
        
#         # Initialize embeddings
#         nn.init.uniform_(self.position_embeddings.weight, -0.1, 0.1)
#         self.register_buffer("position_ids", torch.arange(seq_len).unsqueeze(0), persistent=False) 
        
#     def forward(self, x):
#         pos = self.position_ids.expand(x.size(0), -1)  # no new tensor
#         return self.dropout(x + self.position_embeddings(pos))


# class CNN_Transformer(nn.Module):
#     def __init__(self, nr_channels=1, cnn_embed_dim=16, transformer_embed_dim=16, num_heads=4, 
#                  num_layers=2, nr_classes=6, seq_len:int=10, seq_steps:int=1, dropout=0.1):
#         super(CNN_Transformer, self).__init__()
        
#         # CNN backbone to extract features from images
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels=nr_channels, out_channels=16, kernel_size=3, stride=2, padding=1),  # (16, 64, 64)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 32, 32)

#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # (32, 16, 16)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # (32, H/16, W/16)

#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # (64, H/32, W/32)
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)),  # Output size: (batch_size, 64, 1, 1)
#             #nn.AdaptiveMaxPool2d((1, 1)),
#         )
#         self.seq_len = seq_len
#         self.seq_steps = seq_steps
#         # Flatten CNN output and project to transformer_embed_dim
#         self.embedding = nn.Linear(64, transformer_embed_dim)
        
#         # Positional encoding
#         #self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len=seq_len)
#         self.positional_encoding = PositionalEncodingLearnable(
#             transformer_embed_dim, seq_len=self.seq_len, dropout=dropout)

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=transformer_embed_dim, nhead=num_heads, 
#             # dim_feedforward= 4*transformer_embed_dim, dropout=dropout)
#             dim_feedforward= 4*transformer_embed_dim, dropout=dropout, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Classification head
#         self.fc = nn.Linear(transformer_embed_dim, nr_classes)
        

#     def forward(self, x, return_latent=False, 
#                 return_latent_per_seq=False,
#                 return_cnn_latent=False):

#         if self.seq_len == 1:
#             if x.dim() != 4:
#                 raise ValueError(f"Expected input with 4 dimensions for seq_len=1, got {x.dim()} dimensions.")
            
#             batch_size = x.size(0)
#             # Process the image through the CNN
#             features = self.cnn(x)  # Shape: (batch_size, 64, 1, 1)
#             features = features.view(batch_size, -1)  # Shape: (batch_size, 64)
#             pooled_output = self.embedding(features)  # Shape: (batch_size, embed_dim)
#             if return_latent:
#                 return pooled_output # "pooled_output, self.fc(pooled_output)" -->  would latent, labels 
#             # Classification
#             out = self.fc(pooled_output)
#             return out
#         else:  
#             batch_size, seq_len, channels, height, width = x.size()
#             #batch_size, seq_len, height, width = x.size()

#             # Extract features for each image in the sequence
#             cnn_features = []
#             for t in range(seq_len):
#                 img = x[:, t, :, :, :]  # Shape: (batch_size, seq_length, channels, height, width)
#                 features = self.cnn(img)  # Shape: (batch_size, 64, 1, 1)
#                 features = features.view(batch_size, -1)  # Shape: (batch_size, 64)
#                 features = self.embedding(features)  # Shape: (batch_size, embed_dim)
#                 cnn_features.append(features)
            
#             # Stack features to form a sequence
#             cnn_features = torch.stack(cnn_features, dim=1)  # Shape: (batch_size, seq_len, embed_dim)
            
#             # Add positional encoding
#             cnn_features = self.positional_encoding(cnn_features)
            
#             # Permute for transformer input: (seq_len, batch_size, transformer_embed_dim)
#             cnn_features = cnn_features.permute(1, 0, 2)
            
#             # Pass through transformer encoder
#             transformer_output = self.transformer_encoder(cnn_features)  # Shape: (seq_len, batch_size, embed_dim)
            
#             # Use mean pooling over the sequence dimension
#             transformer_output = transformer_output.permute(1, 0, 2)  # Shape: (batch_size, seq_len, embed_dim)
#             pooled_output = transformer_output.mean(dim=1)  # Shape: (batch_size, transformer_embed_dim)
#             if return_cnn_latent:
#                 return features  # Return CNN features before transformer
#             if return_latent:
#                 return pooled_output  # Return transformer latent features if specified
#             if return_latent_per_seq:
#                 return transformer_output   

#             # Classification
#             out = self.fc(pooled_output)  # Shape: (batch_size, nr_classes)
            
#             return out