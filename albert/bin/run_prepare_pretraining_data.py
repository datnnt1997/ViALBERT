import os
import six
import json
import random
import numpy as np
import collections

from albert.inputters.tokenization import FullTokenizer
from albert.utils import init_logger, logger
from pathlib import Path
from tqdm import tqdm

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def _is_start_piece_sp(piece):
  """Check if the current word piece is the starting piece (sentence piece)."""
  special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
  special_pieces.add(u"€".encode("utf-8"))
  special_pieces.add(u"£".encode("utf-8"))
  # Note(mingdachen):
  # For foreign characters, we always treat them as a whole piece.
  english_chars = set(list("abcdefghijklmnopqrstuvwhyz"))
  if (six.ensure_str(piece).startswith("▁") or
      six.ensure_str(piece).startswith("<") or piece in special_pieces or
      not all([i.lower() in english_chars.union(special_pieces)
               for i in piece])):
    return True
  else:
    return False


def _is_start_piece_bert(piece):
  """Check if the current word piece is the starting piece (BERT)."""
  # When a word has been split into
  # WordPieces, the first token does not have any marker and any subsequence
  # tokens are prefixed with ##. So whenever we see the ## token, we
  # append it to the previous set of word indexes.
  return not six.ensure_str(piece).startswith("##")


def is_start_piece(piece, spm_model_file):
  if spm_model_file:
    return _is_start_piece_sp(piece)
  else:
    return _is_start_piece_bert(piece)


def create_masked_lm_predictions(tokens, vocab_words, opt):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if opt.do_whole_word_mask and len(cand_indexes) >= 1 and not is_start_piece(token, opt.spm_model_path):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(token, opt.spm_model_path):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if opt.masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels, token_boundary)

    num_to_predict = min(opt.max_predictions_per_seq, max(1, int(round(len(tokens) * opt.masked_lm_prob))))

    # Note(mingdachen):
    # By default, we set the probilities to favor longer ngram sequences.
    ngrams = np.arange(1, opt.max_ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, opt.max_ngram + 1)
    pvals /= pvals.sum(keepdims=True)

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

    random.shuffle(ngram_indexes)

    masked_lms = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np.random.choice(ngrams[:len(cand_index_set)],
                             p=pvals[:len(cand_index_set)] /
                               pvals[:len(cand_index_set)].sum(keepdims=True))
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict

    random.shuffle(ngram_indexes)

    select_indexes = set()

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)


def create_instances_from_document(all_documents,  document_index, vocab_words, opt):
    """ Creates `TrainingInstance`s for a single document . """
    document = all_documents[document_index]
    max_num_tokens = opt.max_seq_len - 3

    target_seq_length = max_num_tokens
    if random.random() < opt.short_seq_prob:
        target_seq_length = random.randint(2, max_num_tokens)
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])
                if len(tokens_a) == 0 or len(tokens_b) == 0: continue
                if random.random() < 0.5:
                    is_random_next = True
                    temp = tokens_a
                    tokens_a = tokens_b
                    tokens_b = temp
                else:
                    is_random_next = False
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1
                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                (tokens, masked_lm_positions, masked_lm_labels, token_boundary) = \
                    create_masked_lm_predictions(tokens, vocab_words, opt)
                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels}
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1
    return instances


def create_training_instances(input_file, tokenizer, opt):
    """
    Tạo các "TrainingInstance" từ dữ liệu thô
    Định dạng input file:
        _Mỗi dòng là một sentence .
        _Giữa hai document là một dòng trống .    
    """
    all_documents = [[]]
    lines = open(input_file, 'r', encoding="utf-8").readlines()
    tqdmbar = tqdm(enumerate(lines), desc='read data')
    for line_cnt, line in tqdmbar:
        line = line.strip()
        # Nếu là dòng trống thì tạo một document mới
        if not line:
            all_documents.append([])
        tokens = tokenizer._tokenize(line)
        if tokens:
            all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    random.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    tqdmbar = tqdm(range(len(all_documents)), desc='create instances')
    for document_index in tqdmbar:
        instances.extend(
            create_instances_from_document(all_documents, document_index, vocab_words, opt))

    ex_idx = 0
    while ex_idx < min(5, len(instances)):
        instance = instances[ex_idx]
        logger.info("-------------------------Example-----------------------")
        logger.info(f"id: {ex_idx}")
        logger.info(f"tokens: {' '.join([str(x) for x in instance['tokens']])}")
        logger.info(f"masked_lm_labels: {' '.join([str(x) for x in instance['masked_lm_labels']])}")
        logger.info(f"segment_ids: {' '.join([str(x) for x in instance['segment_ids']])}")
        logger.info(f"masked_lm_positions: {' '.join([str(x) for x in instance['masked_lm_positions']])}")
        logger.info(f"is_random_next : {instance['is_random_next']}")
        ex_idx += 1
    random.shuffle(instances)
    return instances


def prepare(opt):
    opt.data_dir = Path(opt.data_dir)
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
    init_logger(log_file=opt.output_dir + "pregenerate_training_data_ngram.log")
    logger.info("pregenerate training data parameters:\n %s", opt)
    tokenizer = FullTokenizer(vocab_file=opt.vocab_path, do_lower_case=opt.do_lower_case,
                              do_cased=opt.do_cased, spm_model_file=opt.spm_model_path)

    # Tách nhỏ dữ liệu kích thước lớn
    if opt.do_split:
        corpus_path = opt.data_dir / "corpus/corpus.txt"
        split_save_path = opt.data_dir / "corpus/train"
        if not split_save_path.exists():
            split_save_path.mkdir(exist_ok=True)
        line_per_file = opt.line_per_file
        command = f'split -a 4 -l {line_per_file} -d {corpus_path} {split_save_path}/shard_'
        os.system(f"{command}")

    data_path = opt.data_dir / "corpus/train"
    files = sorted([f for f in data_path.parent.iterdir() if f.exists() and '.txt' in str(f)])
    for idx in range(opt.file_num):
        logger.info(f"pregenetate {opt.data_name}_file_{idx}.json")
        save_filename = data_path / f"{opt.data_name}_file_{idx}.json"
        num_instances = 0
        with save_filename.open('w', encoding='utf-8') as fw:
            for file_idx in range(len(files)):
                file_path = files[file_idx]
                file_examples = create_training_instances(input_file=file_path, tokenizer=tokenizer, opt=opt)
                file_examples = [json.dumps(instance) for instance in file_examples]
                for instance in file_examples:
                     fw.write(str(instance) + '\n')
                     num_instances += 1
        metrics_file = data_path / f"{opt.data_name}_file_{idx}_metrics.json"
        print(f"num_instances: {num_instances}")
        with metrics_file.open('w') as metrics_file:
            metrics = {
                "num_training_examples": num_instances,
                "max_seq_len": opt.max_seq_len
            }
            metrics_file.write(json.dumps(metrics))
