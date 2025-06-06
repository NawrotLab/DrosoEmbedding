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
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), seq_len)
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        return self.dropout(x)
    

class CNN_Transformer(nn.Module):
    def __init__(self, nr_channels=1, embed_dim=128, num_heads=4, 
                 num_layers=2, nr_classes=6, seq_len=10, seq_steps =1, dropout=0.1):
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
        # Flatten CNN output and project to embed_dim
        self.embedding = nn.Linear(64, embed_dim)
        
        # Positional encoding
        #self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len=seq_len)
        self.positional_encoding = PositionalEncodingLearnable(
            embed_dim, seq_len=self.seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward= 4*embed_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(embed_dim, nr_classes)
        

    def forward(self, x, return_latent=False, 
                return_latent_per_seq=False,
                return_embeddings=False):

        if self.seq_len == 1:
            #x = x.unsqueeze(1)
            if x.dim() != 4:
                raise ValueError(f"Expected input with 4 dimensions for seq_len=1, got {x.dim()} dimensions.")
            
            batch_size = x.size(0)
            # Process the image through the CNN
            features = self.cnn(x)  # Shape: (batch_size, 64, 1, 1)
            features = features.view(batch_size, -1)  # Shape: (batch_size, 64)
            pooled_output = self.embedding(features)  # Shape: (batch_size, embed_dim)
            if return_latent:
                return pooled_output
            if return_embeddings:
                return self.fc(pooled_output), pooled_output
            # Classification
            out = self.fc(pooled_output)
            return out
        else:  
            batch_size, seq_len, channels, height, width = x.size()
            #batch_size, seq_len, height, width = x.size()

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
            
            # Permute for transformer input: (seq_len, batch_size, embed_dim)
            cnn_features = cnn_features.permute(1, 0, 2)
            
            # Pass through transformer encoder
            transformer_output = self.transformer_encoder(cnn_features)  # Shape: (seq_len, batch_size, embed_dim)
            
            # Use mean pooling over the sequence dimension
            transformer_output = transformer_output.permute(1, 0, 2)  # Shape: (batch_size, seq_len, embed_dim)
            pooled_output = transformer_output.mean(dim=1)  # Shape: (batch_size, embed_dim)
            if return_latent:
                return pooled_output  # Return latent features if specified
            if return_latent_per_seq:
                return transformer_output
            
            if return_embeddings:
                return self.fc(pooled_output), pooled_output

            # Classification
            out = self.fc(pooled_output)  # Shape: (batch_size, nr_classes)
            
            return out