import torch.nn as nn

from albert.utils import ACT2FN


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()

        # Sub-Modules
        self.w_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_2 = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.activation_function = ACT2FN[config.hidden_act]

    def forward(self, attention_output):
        ffn_output = self.w_1(attention_output)
        ffn_output = self.activation_function(ffn_output)
        ffn_output = self.dropout_1(ffn_output)

        ffn_output = self.w_2(ffn_output)
        ffn_output = self.dropout_2(ffn_output)

        hidden_states = self.layer_norm(ffn_output + attention_output)

        return hidden_states
