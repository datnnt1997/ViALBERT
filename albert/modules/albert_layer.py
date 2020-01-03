import torch.nn as nn

from albert.modules.position_ffn import PositionwiseFeedForward
from albert.modules.multi_headed_attn import MultiHeadedAttention


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super(AlbertLayer, self).__init__()

        # Sub-Modules
        self.attention = MultiHeadedAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_out = self.attention(hidden_states, attention_mask, head_mask)
        attention_out = self.layer_norm(attention_out[0] + hidden_states)
        hidden_states = self.feed_forward(attention_out[0])
        outputs = (hidden_states, ) + attention_out[1:]
        return outputs
