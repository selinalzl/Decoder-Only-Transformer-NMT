#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    Code from https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding slightly modified.
    """
    def __init__(self, d_model, max_length, dropout=None):
        """
        Positional Encoding with a maximum length of max_length.

        :param d_model: Dimensionality of the input embeddings
        :param max_length: Maximum length of source/target sentence
        :param dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds the positional encodings to the input embeddings.

        :param x: Input
        """
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        if self.dropout is None:
            return x
        else:
            return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """
    Implements Scaled Dot-Product Attention.
    3.2.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, d_k, dropout=None):
        """
        :param d_k: Head dimension 
        :param dropout: Dropout probability
        """
        super(ScaledDotProductAttention, self).__init__()

        self.d_k = d_k

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        """
        Computes Scaled Dot-Product Attention.

        :param query: Query = [batch_size, n_heads, seq_len, d_k]
        :param key: Key = [batch_size, n_heads, seq_len, d_k]
        :param value: Value = [batch_size, n_heads, seq_len, d_k]
        :param mask: Mask = [batch_size, n_heads, seq_len, seq_len]
        :return: Output tensor and attention weights
        """
        # compute scores: [batch_size, n_heads, seq_len, seq_len]
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.d_k ** 0.5)

        # apply mask
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # apply softmax function to obtain weights and apply dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # output = [batch_size, n_heads, seq_len, d_k]
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MaskedMultiHeadAttention(nn.Module):
    """
    Implements Masked Multi-Head Attention module from "Attention is All You Need".
    3.2.2 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, d_model, n_heads, dropout=None):
        """
        Creates Masked Multi-Head Attention layer.

        :param d_model: Size of model (must be divisible by n_heads)
        :param n_heads: Number of heads
        :param dropout: Dropout probability
        """
        super(MaskedMultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.sdp_attention = ScaledDotProductAttention(self.head_dim, dropout)

        self.fc_linear = nn.Linear(d_model, d_model)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        """
        Computes Masked Multi-Head Attention.

        :param query: Query = [batch_size, seq_len, d_model]
        :param key: Key = [batch_size, seq_len, d_model]
        :param value: Value = [batch_size, seq_len, d_model]
        :param mask: Mask = [batch_size, seq_len, key_len]
        :return: Output tensor and attention weights
        """
        batch_size = query.size(0)

        # reshape query, key, value for computation to [batch_size, n_heads, seq_len, head_dim]
        q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # apply the mask: [batch_size, n_heads, seq_len, seq_len]
        mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # compute Scaled Dot-Product Attention
        # attention = [batch_size, n_heads, seq_len, head_dim], attention_weights = [batch_size, n_heads, seq_len, seq_len]
        attention, attention_weights = self.sdp_attention(q, k, v, mask)
        
        # reshape attention to [batch_size, seq_len, d_model]
        attention = attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        # output = [batch_size, seq_len, d_model]
        output = self.fc_linear(attention)

        return output, attention_weights


class PositionwiseFeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Projects to ff_dim and then back to d_model.
    """
    def __init__(self, d_model, ff_dim, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.

        :param d_model: Dimensionality of input and output
        :param ff_dim: Dimensionality of the inner layer 
        :param dropout: Dropout probability
        """
        super(PositionwiseFeedForwardNetwork, self).__init__()

        self.linear_1 = nn.Linear(d_model, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for position-wise FFN

        :param x: Input = [batch_size, seq_len, d_model]
        :return: Output = [batch_size, seq_len, d_model]
        """
        # project input to [batch_size, seq_len, ff_dim]
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)

        x = self.linear_2(x)

        return x


class DecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    """
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        """
        A single Transformer decoder layer made of masked self-attention and feed-forward.

        :param d_model: Size of the model
        :param n_heads: Number of heads
        :param ff_dim: Dimensionality of feed-forward inner layer
        :param dropout: Dropout probability
        """
        super(DecoderLayer, self).__init__()

        self.masked_self_attention = MaskedMultiHeadAttention(d_model, n_heads, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.attention_layer_norm = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForwardNetwork(d_model, ff_dim)
        self.dropout_2 = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(d_model)

    def forward(self, input, mask):
        """
        Forward pass of Transformer decoder layer.

        :param input: Input = [batch_size, seq_len, d_model]
        :param mask: Mask = [batch_size, seq_len, seq_len]
        :return: Output = [batch_size, seq_len, d_model] and 
                 attention weights = [batch_size, n_heads, seq_len, seq_len]
        """
        # decoder masked self-attention
        output, attention_weights = self.masked_self_attention(input, input, input, mask)
        output = self.dropout_1(output)

        # residual connection and layer norm
        output = self.attention_layer_norm(input + output)

        ffn_output = self.ffn(output)
        ffn_output = self.dropout_2(ffn_output)

        # residual connection and layer norm
        ffn_output = self.ffn_layer_norm(output + ffn_output)

        return ffn_output, attention_weights


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-Only Transformer.
    """
    def __init__(self, pad_index, vocab_size, d_model, max_seq_len, n_heads, n_layers, ff_dim, dropout):
        """
        Initialize a Transformer decoder with N layers. Decoder self-attention layers are masked 
        so that an attention head cannot attend to future words during training.

        :param pad_index: Index of the padding token '[PAD]'
        :param vocab_size: The size of the vocabulary
        :param d_model: The size of hidden dimensions
        :param max_seq_len: Maximum total input sequence length after tokenization
        :param n_heads: Number of heads in the multi-headed attention for each layer
        :param n_layers: Number of decoder layers
        :param ff_dim: Position-wise feed-forward size
        :param dropout: Dropout probability
        """
        super(DecoderOnlyTransformer, self).__init__()

        self.pad_index = pad_index
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model,
                                                  n_heads,
                                                  ff_dim,
                                                  dropout) for _ in range(n_layers)])
        self.fc_linear = nn.Linear(d_model, vocab_size)

    def forward(self, input):
        """
        Decoder-Only Transformer forward pass.

        :param input: Input = [batch_size, seq_len]
        :return: Output = [batch_size, seq_len, vocab_size] and 
                 attention = [batch_size, n_heads, seq_len, seq_len]
        """
        # multiplication to reduce variance in the embeddings
        output = self.embedding(input) * math.sqrt(self.d_model)

        # output = [batch_size, seq_len, d_model]
        output = self.pos_enc(output)

        # pad_mask = [batch_size, 1, seq_len]
        pad_mask = self.padding_mask(input, self.pad_index)

        # subsequent_mask = [1, seq_len, seq_len]
        subsequent_mask = self.subsequent_pos_mask(input).to(device=pad_mask.device)

        # attention_mask = [batch_size, seq_len, seq_len]
        attention_mask = pad_mask & subsequent_mask

        for layer in self.layers:
            output, attention = layer(output, attention_mask)

        output = self.fc_linear(output)

        return output, attention

    def padding_mask(self, input, pad_index):
        """
        Masks out padding so model doesn't attend to padding tokens.

        :param input: Model input
        :param pad_index: Index of the padding token '[PAD]'
        :return: Padding mask
        """
        pad_mask = (input != pad_index).unsqueeze(1)
        return pad_mask

    def subsequent_pos_mask(self, input):
        """
        Masks out subsequent positions.

        :param input: Model input
        :return: Mask for subsequent positions
        """
        size = input.size(-1)
        subsequent_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0


def initialize_weights(model):
    """
    Weight initialization using Xavier uniform.
    """
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)


def count_parameters(model):
    """
    Counts number of model parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

