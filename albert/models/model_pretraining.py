import torch
import torch.nn as nn

from albert.models.model import AlbertModel
from albert.modules import PreTrainingHeads


class AlbertForPreTraining(nn.Module):
    def __init__(self, config):
        super(AlbertForPreTraining, self).__init__()
        self.config = config
        self.albert = AlbertModel(config)
        self.classifier = PreTrainingHeads(config)
        self.init_weights()

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """

        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight


        if hasattr(first_module, 'bias') and first_module.bias is not None:
            first_module.bias.data = torch.nn.functional.pad(
                first_module.bias.data,
                (0, first_module.weight.shape[0] - first_module.bias.shape[0]),
                'constant',
                0
            )
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        first_module = self.classifier.mask_predictions.decoder
        second_module = self.albert.embeddings.token_embeddings
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

        if hasattr(first_module, 'bias') and first_module.bias is not None:
            first_module.bias.data = torch.nn.functional.pad(
                first_module.bias.data,
                (0, first_module.weight.shape[0] - first_module.bias.shape[0]),
                'constant',
                0
            )

    def init_weights(self):
        """ Initialize and prunes weights if needed."""
        # Initialize weights
        self.apply(self._init_weights)

        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        # Tie weights if needed
        self.tie_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                masked_lm_labels=None,
                next_sentence_label=None):

        outputs = self.albert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                              position_ids=position_ids, head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.classifier(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            nsp_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = mlm_loss + nsp_loss
            outputs = (total_loss,) + outputs
        return outputs
