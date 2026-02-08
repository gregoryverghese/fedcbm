"""Attention mechanisms for MINOTAUR models."""
import torch
import torch.nn.functional as F


# Multi-head attention mechanism with query, key, and value projections
class MultiKQVAtten(torch.nn.Module):
    """
    Implements a multi-head attention mechanism with query, key, and value projections.
    Input: (batch_size, n_tiles, embedding_dim)
    Output: (batch_size, n_tiles, num_heads)
    """
    def __init__(self, args):
        super(MultiKQVAtten, self).__init__()

        self.num_heads = args.num_heads
        self.attn_dim = args.atn_dim
        self.dropout = args.dropout
        self.head_dim = args.atn_dim // self.num_heads
        self.embedding_dim = args.feature_depth

        self.query_proj = torch.nn.Linear(self.embedding_dim, self.attn_dim)
        self.key_proj = torch.nn.Linear(self.embedding_dim, self.attn_dim)
        self.value_proj = torch.nn.Linear(self.embedding_dim, self.attn_dim)
        self.out_proj = torch.nn.Linear(self.attn_dim, self.attn_dim)
        self.dropout = torch.nn.Dropout(self.dropout)

    def forward(self, x):
        """
        Forward pass for multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_tiles, embedding_dim).
        
        Returns:
            torch.Tensor: Attention scores of shape (batch_size, n_tiles, num_heads).
        """
        b_size, n_tiles, _ = x.shape

        Q = self.query_proj(x)  #(batch_size, n_tiles, attn_dim)
        K = self.key_proj(x)    #(batch_size, n_tiles, attn_dim)
        V = self.value_proj(x)  #(batch_size, n_tiles, attn_dim)

        Q = Q.view(b_size, n_tiles, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(b_size, n_tiles, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(b_size, n_tiles, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  #(batch_size, num_heads, n_tiles, n_tiles)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute weighted values, concatenate heads and project
        attn_output = torch.matmul(attn_weights, V)  #(batch_size, num_heads, n_tiles, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b_size, n_tiles, self.attn_dim)
        output = self.out_proj(attn_output)  #(batch_size, n_tiles, attn_dim)

        return output


# Attention block for multi-concept attention
class MultiConceptAttention(torch.nn.Module):
    """
    Implements the multi-head attention mechanism with linear transformations.

    Input:
        x (torch.Tensor): Tensor of shape (batch, nb_tiles, features)
    Output:
        Attention scores of shape (batch, nb_tiles, num_heads)
    """ 

    def __init__(self, in_dim, attn_dim, n_att_heads, dropout) -> None:
        """
        Initializes the MultiHeadAttention module.

        Args:
            in_dim: Input feature dimension
            attn_dim: Total attention dimension
            n_att_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiConceptAttention, self).__init__()
        self.in_dim = in_dim
        self.n_att_heads = n_att_heads
        self.dropout = dropout
        self.head_dim = attn_dim // self.n_att_heads

        # Define linear layers for attention transformations
        self.atn_1_linear = torch.nn.Linear(self.in_dim, attn_dim)
        self.atn_2_linear = torch.nn.Linear(self.head_dim, 1)  # Projects each head's subspace to a single score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, nb_tiles, features).

        Returns:
            torch.Tensor: Output attention scores of shape (batch, nb_tiles, num_heads).
        """
        bs, nbt, _ = x.shape

        # Apply the first linear layer to project into the attention dimension
        x = self.atn_1_linear(x)  # Shape: (batch, nb_tiles, atn_dim)
        x = torch.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Reshape to separate heads for multi-head processing &
        # apply the second linear layer independently to each head
        x = x.view(bs, nbt, self.n_att_heads, self.head_dim)  # (batch, nb_tiles, num_heads, head_dim)
        x = self.atn_2_linear(x)  # (batch, nb_tiles, num_heads, 1)
        x = x.squeeze(-1)  # (batch, nb_tiles, num_heads)
        return x


# Concept-level attention mechanism
class ConceptAttention(torch.nn.Module):
    """
    Implements attention mechanism to learn importance weights for each concept.
    
    Input:
        concept_embeddings (torch.Tensor): Tensor of shape (batch, n_concepts, emb_size)
    Output:
        attention_weights (torch.Tensor): Attention weights of shape (batch, n_concepts)
        weighted_embeddings (torch.Tensor): Weighted concept embeddings of shape (batch, n_concepts, emb_size)
    """
    
    def __init__(self, emb_size, attn_dim=64, dropout=0.1):
        """
        Initializes the ConceptAttention module.
        
        Args:
            emb_size: Size of concept embeddings
            attn_dim: Attention dimension for computing attention weights
            dropout: Dropout rate
        """
        super(ConceptAttention, self).__init__()
        self.emb_size = emb_size
        self.attn_dim = attn_dim
        self.dropout = dropout
        
        # Linear layers for attention computation
        self.attention_linear = torch.nn.Sequential(
            torch.nn.Linear(emb_size, attn_dim),
            torch.nn.Tanh(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(attn_dim, 1)
        )
    
    def forward(self, concept_embeddings):
        """
        Forward pass for concept attention.
        
        Args:
            concept_embeddings (torch.Tensor): Input tensor of shape (batch, n_concepts, emb_size)
            
        Returns:
            attention_weights (torch.Tensor): Attention weights of shape (batch, n_concepts)
            weighted_embeddings (torch.Tensor): Weighted concept embeddings of shape (batch, n_concepts, emb_size)
        """
        # Compute attention scores for each concept
        attention_scores = self.attention_linear(concept_embeddings)  # (batch, n_concepts, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, n_concepts)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, n_concepts)
        
        # Apply attention weights to concept embeddings
        weighted_embeddings = concept_embeddings * attention_weights.unsqueeze(-1)  # (batch, n_concepts, emb_size)
        
        return attention_weights, weighted_embeddings
    
    def get_concept_attention_weights(self, concept_embeddings):
        """
        Get concept attention weights for analysis purposes.
        
        Args:
            concept_embeddings (torch.Tensor): Concept embeddings of shape (batch, n_concepts, emb_size)
            
        Returns:
            attention_weights (torch.Tensor): Attention weights of shape (batch, n_concepts)
        """
        if self.concept_attention is not None:
            attention_weights, _ = self.concept_attention(concept_embeddings)
            return attention_weights
        else:
            return None

