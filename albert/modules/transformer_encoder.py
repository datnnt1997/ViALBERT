import torch
import torch.nn as nn

from albert.modules.albert_group import AlbertGroup


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        # Parameters
        self.hiddend_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups

        # Sub-Modules
        self.embedding_hidden_mapping_in = nn.Linear(self.embedding_size, self.hiddend_size)
        self.albert_groups = nn.ModuleList([AlbertGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        if self.hiddend_size != self.embedding_size:
            hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        if self.output_attentions:
            all_attentions = ()
        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        for layer_idx in range(self.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(layer_idx / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            group_module = self.albert_groups[group_idx]
            group_output = group_module(
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group: (group_idx + 1) * layers_per_group],
            )

            hidden_states = group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs





