import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """
    Construct the embeddings from word/sub-word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(Embeddings, self).__init__()

        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids=None, segment_ids=None, position_ids=None):
        assert token_ids is not None, ValueError("ERROR: The input cannot be None!!!")

        input_shape = token_ids.shape
        device = token_ids.device
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)

        token_embeddings = self.token_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)

        embeddings = token_embeddings + position_embeddings + segment_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings



