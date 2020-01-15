import torch.nn as nn

from albert.modules.albert_layer import AlbertLayer


class AlbertGroup(nn.Module):
    def __init__(self, config):
        super(AlbertGroup, self).__init__()
        # Parameters
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        # Sub-Modules
        self.albert_groups = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_groups):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index])
            hidden_states = layer_output[0]

            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs

