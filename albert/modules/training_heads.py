import torch
import torch.nn as nn
from albert.utils import ACT2FN


class MLMHead(nn.Module):
    def __init__(self, config):
        super(MLMHead, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class NSPHead(nn.Module):
    def __init__(self, config):
        super(NSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class PreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(PreTrainingHeads, self).__init__()
        self.mask_predictions = MLMHead(config)
        self.seq_relationship = NSPHead(config)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.mask_predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score
