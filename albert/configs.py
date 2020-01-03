def model_opts(parser):
    """
    These configuration parameters are passed to the construction of the model.
    """
    # Embedding Parameters
    group = parser.add_argument_group('Model-Embeddings')
    group.add('--vocab_size', '-vocab_size',
              type=int, default=30000,
              help='Vocabulary size of `inputs_ids` in `AlbertModel`.')
    group.add('--embedding_size', '-embedding_size',
              type=int, default=128,
              help='Size of voc embeddings.')
    group.add('--max_position_embeddings', '-max_position_embeddings',
              type=int, default=512,
              help='The maximum sequence length that this model might ever be used with.'
                   'Typically set this to something large just in case (e.g., 512 or 1024 or 2048).')
    group.add('--type_vocab_size', '-type_vocab_size',
              type=int, default=2,
              help='The vocabulary size of the `token_type_ids` passed into `AlbertModel`.')

    # Modeling Parameters
    group = parser.add_argument_group('Model-Layers')
    group.add('--hidden_size', '-hidden_size',
              type=int, default=768,
              help='Size of the encoder layers and the pooler layer.')
    group.add('--num_hidden_layers', '-num_hidden_layers',
              type=int, default=12,
              help='Number of hidden layers in the Transformer encoder.')
    group.add('--num_hidden_groups', '-num_hidden_groups',
              type=int, default=1,
              help='Number of group for the hidden layers, parameters in the same group are shared.')
    group.add('--num_attention_heads', '-num_attention_heads',
              type=int, default=12,
              help='Number of attention heads for each attention layer in the Transformer encoder.')
    group.add('--intermediate_size', '-intermediate_size',
              type=int, default=3072,
              help='The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.')
    group.add('--inner_group_num', '-inner_group_num',
              type=int, default=1,
              help='Number of inner repetition of attention and ffn.')
    group.add('--down_scale_factor', '-down_scale_factor',
              type=float, default=1.0,
              help='The scale to apply.')
    group.add('--hidden_act', '-hidden_act',
              type=str, default="gelu",
              help='The non-linear activation function (function or string) in the encoder and pooler.')
    group.add('--hidden_dropout_prob', '-hidden_dropout_prob',
              type=float, default=0.0,
              help='The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.')
    group.add('--attention_probs_dropout_prob', '-attention_probs_dropout_prob',
              type=float, default=0.0,
              help='The dropout ratio for the attention probabilities.')
    group.add('--initializer_range', '-initializer_range',
              type=float, default=0.02,
              help='The stdev of the truncated_normal_initializer for initializing all weight matrices.')
    group.add('--layer_norm_eps', '-layer_norm_eps',
              type=float, default=1e-12,
              help='The epsilon used by LayerNorm.')

    # Training Parameters
    group = parser.add_argument_group('Model-Train')

    # Task Parameters
    group = parser.add_argument_group('Model-Task')
    group.add('--finetuning_task', '-finetuning_task',
              type=str, default=None,
              help='Name of the task used to fine-tune the model. '
                   'This can be used when converting from an original (TensorFlow or PyTorch) checkpoint.')
    group.add('--num_labels', '-num_labels',
              type=int, default=0,
              help='Number of classes to use when the model is a classification model (sequences/tokens)')
    group.add('--output_attentions', '-output_attentions',
              type=bool, default=False,
              help='Should the model returns attentions weights.')
    group.add('--output_hidden_states', '-output_hidden_states',
              type=bool, default=False,
              help='Should the model returns all hidden-states.')
    group.add('--torchscript', '-torchscript',
              type=str, default=False,
              help='Is the model used with Torchscript.')
    group.add('--pruned_heads', '-pruned_heads',
              type=dict, default={},
              help='Is the model used with Torchscript.')
