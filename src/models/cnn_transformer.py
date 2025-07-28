import torch
import torch.nn as nn


class PositionalEncodingLearnable(nn.Module):
    def __init__(self, embed_dim, seq_len=10, dropout=0.1):
        super(PositionalEncodingLearnable, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable positional embeddings
        self.position_embeddings = nn.Embedding(seq_len, embed_dim)
        
        # Initialize embeddings
        nn.init.uniform_(self.position_embeddings.weight, -0.1, 0.1)
        self.register_buffer("position_ids",
                        torch.arange(seq_len).unsqueeze(0), persistent=False) # VR: register once instead of registering position id in every forward call
        
    # def forward(self, x):
    #     # x shape: (batch_size, seq_len, embed_dim)
    #     seq_len = x.size(1)
    #     position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
    #     position_ids = position_ids.unsqueeze(0).expand(x.size(0), seq_len)
    #     position_embeddings = self.position_embeddings(position_ids)
    #     x = x + position_embeddings
    #     return self.dropout(x)
    def forward(self, x):
        pos = self.position_ids.expand(x.size(0), -1)  # no new tensor
        return self.dropout(x + self.position_embeddings(pos))
    

class CNN_Transformer(nn.Module):
    def __init__(self, nr_channels=1, cnn_embed_dim=16, transformer_embed_dim=16, num_heads=4, 
                 num_layers=2, nr_classes=6, seq_len:int=10, seq_steps:int=1, dropout=0.1):
        super(CNN_Transformer, self).__init__()
        
        # CNN backbone to extract features from images
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=nr_channels, out_channels=16, kernel_size=3, stride=2, padding=1),  # (16, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 32, 32)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # (32, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, H/16, W/16)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # (64, H/32, W/32)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Output size: (batch_size, 64, 1, 1)
            #nn.AdaptiveMaxPool2d((1, 1)),
        )
        self.seq_len = seq_len
        self.seq_steps = seq_steps
        # Flatten CNN output and project to transformer_embed_dim
        self.embedding = nn.Linear(64, transformer_embed_dim)
        
        # Positional encoding
        #self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len=seq_len)
        self.positional_encoding = PositionalEncodingLearnable(
            transformer_embed_dim, seq_len=self.seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim, nhead=num_heads, 
            dim_feedforward= 4*transformer_embed_dim, dropout=dropout,
            batch_first=True)  # VR: I added batch_first to avoid permuting multiple times. we can now remove permuting in the forward pass (check comment below)
        
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(transformer_embed_dim, nr_classes)
        

    def forward(self, x, return_latent=False, 
                return_latent_per_seq=False,
                return_cnn_latent=False):

        if self.seq_len == 1:
            if x.dim() != 4:
                raise ValueError(f"Expected input with 4 dimensions for seq_len=1, got {x.dim()} dimensions.")
            
            batch_size = x.size(0)
            # Process the image through the CNN
            features = self.cnn(x)  # Shape: (batch_size, 64, 1, 1)
            features = features.view(batch_size, -1)  # Shape: (batch_size, 64)
            pooled_output = self.embedding(features)  # Shape: (batch_size, embed_dim)
            if return_latent:
                return pooled_output # "pooled_output, self.fc(pooled_output)" -->  would latent, labels 
            # Classification
            out = self.fc(pooled_output)
            return out
        else:  
            batch_size, seq_len, channels, height, width = x.size()
            x = x.view(batch_size * seq_len, channels, height, width) 
            
            feats = self.cnn(x) # (B·S, 64, 1, 1)
            feats = feats.flatten(1)           # (B·S, 64)  
            feats = self.embedding(feats)      # (B·S, d_model)
            cnn_feats = feats.view(batch_size, seq_len, -1)       # restore sequence dim
            pe_feats = self.positional_encoding(cnn_feats)   # (B, S, d_model)
            transformer_out = self.transformer_encoder(pe_feats)  # (B, S, d_model)
            
            pooled = transformer_out.mean(dim=1)        # (B, d_model)
            if return_latent_per_seq:   return cnn_feats
            if return_latent:           return pooled
            if return_cnn_latent:       return feats.view(batch_size*seq_len, -1) # VR: Here we concatenate all the cnn embedding of all seqs. we could also average them.

            # Classification
            cls_out = self.fc(pooled_output)  # Shape: (batch_size, nr_classes)

            return cls_out
            """
            # Extract features for each image in the sequence
            cnn_features = []
            for t in range(seq_len):
                img = x[:, t, :, :, :]  # Shape: (batch_size, seq_length, channels, height, width)
                features = self.cnn(img)  # Shape: (batch_size, 64, 1, 1)
                features = features.view(batch_size, -1)  # Shape: (batch_size, 64)
                features = self.embedding(features)  # Shape: (batch_size, embed_dim)
                cnn_features.append(features)
            
            # Stack features to form a sequence
            cnn_features = torch.stack(cnn_features, dim=1)  # Shape: (batch_size, seq_len, embed_dim)
            
            # Add positional encoding
            cnn_features = self.positional_encoding(cnn_features)
            
            # Permute for transformer input: (seq_len, batch_size, transformer_embed_dim)
            cnn_features = cnn_features.permute(1, 0, 2)
            
            # Pass through transformer encoder
            transformer_output = self.transformer_encoder(cnn_features)  # Shape: (seq_len, batch_size, embed_dim)
            
            # Use mean pooling over the sequence dimension
            transformer_output = transformer_output.permute(1, 0, 2)  # Shape: (batch_size, seq_len, embed_dim)
            pooled_output = transformer_output.mean(dim=1)  # Shape: (batch_size, transformer_embed_dim)
            if return_cnn_latent:
                return features  # Return CNN features before transformer
            if return_latent:
                return pooled_output  # Return transformer latent features if specified
            if return_latent_per_seq:
                return transformer_output   

            # Classification
            out = self.fc(pooled_output)  # Shape: (batch_size, nr_classes)
            
            return out
            """