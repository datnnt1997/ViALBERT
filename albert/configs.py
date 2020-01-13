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


def pretrain_opts(parser):
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--config_path", default=None, type=str, required=True)
    parser.add_argument("--vocab_path", default=None, type=str, required=True)
    parser.add_argument("--spm_model_path", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--do_split", default=False, type=bool, help="Whether to split big file")
    parser.add_argument("--line_per_file", type=int, default=1000000000, help="Number of line in each file")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--model_path", default='', type=str)
    parser.add_argument('--data_name', default='albert', type=str)
    parser.add_argument("--file_num", type=int, default=10,
                        help="Number of dynamic masking to pregenerate (with different masks)")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of epochs to train for")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_cased", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--max_ngram', default=3, type=int)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,  # 128 * 0.15
                        help="Maximum number of tokens to mask in each sequence")

    parser.add_argument('--num_eval_steps', default=100)
    parser.add_argument('--num_save_steps', default=200)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training.")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument("--learning_rate", default=0.00176, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")


