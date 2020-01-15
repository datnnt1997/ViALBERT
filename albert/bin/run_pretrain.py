import json
import torch
import torch.nn as nn
import time
import numpy as np

from pathlib import Path
from collections import namedtuple
from albert.inputters import FullTokenizer
from torch.utils.data.distributed import DistributedSampler
from albert.utils import logger, init_logger, set_seed, AverageMeter, LMAccuracy
from albert.models import AlbertForPreTraining
from tempfile import TemporaryDirectory
from albert.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, RandomSampler

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids
    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1
    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids
    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, file_id, tokenizer, data_name, reduce_memory=False):
        self.tokenizer = tokenizer
        self.file_id = file_id
        data_file = training_path / f"{data_name}_file_{self.file_id}.json"
        metrics_file = training_path / f"{data_name}_file_{self.file_id}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir / 'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir / 'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir / 'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir / 'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir / 'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
        logger.info(f"Loading training examples for {str(data_file)}")
        with open(data_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logger.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)))


def train(opt):
    opt.data_dir = Path(opt.data_dir)
    opt.output_dir = Path(opt.output_dir)
    pregenerated_data = opt.data_dir / "corpus/train"
    init_logger(log_file=str(opt.output_dir / "train_albert_model.log"))
    assert pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by prepare_lm_data_mask.py!"

    samples_per_epoch = 0
    for i in range(opt.file_num):
        data_file = pregenerated_data / f"{opt.data_name}_file_{i}.json"
        metrics_file = pregenerated_data / f"{opt.data_name}_file_{i}_metrics.json"
        if data_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch += metrics['num_training_examples']
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({opt.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            break

    logger.info(f"samples_per_epoch: {samples_per_epoch}")
    if opt.local_rank == -1 or opt.no_cuda:
        device = torch.device(f"cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu")
        opt.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        opt.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info(
        f"device: {device} , distributed training: {bool(opt.local_rank != -1)}, 16-bits training: {opt.fp16}, ")

    if opt.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: {opt.gradient_accumulation_steps}, should be >= 1")
    opt.train_batch_size = opt.train_batch_size // opt.gradient_accumulation_steps

    set_seed(opt.seed)
    tokenizer = FullTokenizer(vocab_file=opt.vocab_path, do_lower_case=opt.do_lower_case,
                              do_cased=opt.do_cased, spm_model_file=opt.spm_model_path)
    total_train_examples = samples_per_epoch * opt.epochs

    num_train_optimization_steps = int(total_train_examples / opt.train_batch_size / opt.gradient_accumulation_steps)
    if opt.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    opt.warmup_steps = int(num_train_optimization_steps * opt.warmup_proportion)

    model = AlbertForPreTraining(config=opt)

    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': opt.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=opt.learning_rate, eps=opt.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opt.warmup_steps,
                                                num_training_steps=num_train_optimization_steps)
    # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if opt.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt.fp16_opt_level)

    if opt.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if opt.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank],
                                                          output_device=opt.local_rank)
    global_step = 0
    mask_metric = LMAccuracy()
    sop_metric = LMAccuracy()
    tr_mask_acc = AverageMeter()
    tr_sop_acc = AverageMeter()
    tr_loss = AverageMeter()
    tr_mask_loss = AverageMeter()
    tr_sop_loss = AverageMeter()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    train_logs = {}
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_examples}")
    logger.info(f"  Batch size = {opt.train_batch_size}")
    logger.info(f"  Num steps = {num_train_optimization_steps}")
    logger.info(f"  warmup_steps = {opt.warmup_steps}")
    start_time = time.time()

    set_seed(opt.seed)  # Added here for reproducibility
    for epoch in range(opt.epochs):
        for idx in range(opt.file_num):
            epoch_dataset = PregeneratedDataset(file_id=idx, training_path=pregenerated_data, tokenizer=tokenizer,
                                                reduce_memory=opt.reduce_memory, data_name=opt.data_name)
            if opt.local_rank == -1:
                train_sampler = RandomSampler(epoch_dataset)
            else:
                train_sampler = DistributedSampler(epoch_dataset)
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=opt.train_batch_size)
            model.train()
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
                outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                prediction_scores = outputs[0]
                seq_relationship_score = outputs[1]

                masked_lm_loss = loss_fct(prediction_scores.view(-1, opt.vocab_size), lm_label_ids.view(-1))
                next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), is_next.view(-1))
                loss = masked_lm_loss + next_sentence_loss

                mask_metric(logits=prediction_scores.view(-1, opt.vocab_size), target=lm_label_ids.view(-1))
                sop_metric(logits=seq_relationship_score.view(-1, 2), target=is_next.view(-1))

                if opt.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if opt.gradient_accumulation_steps > 1:
                    loss = loss / opt.gradient_accumulation_steps
                if opt.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                nb_tr_steps += 1
                tr_mask_acc.update(mask_metric.value(), n=input_ids.size(0))
                tr_sop_acc.update(sop_metric.value(), n=input_ids.size(0))
                tr_loss.update(loss.item(), n=1)
                tr_mask_loss.update(masked_lm_loss.item(), n=1)
                tr_sop_loss.update(next_sentence_loss.item(), n=1)

                if (step + 1) % opt.gradient_accumulation_steps == 0:
                    if opt.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), opt.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step % opt.num_eval_steps == 0:
                    now = time.time()
                    eta = now - start_time
                    if eta > 3600:
                        eta_format = ('%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60))
                    elif eta > 60:
                        eta_format = '%d:%02d' % (eta // 60, eta % 60)
                    else:
                        eta_format = '%ds' % eta
                    train_logs['loss'] = tr_loss.avg
                    train_logs['mask_acc'] = tr_mask_acc.avg
                    train_logs['sop_acc'] = tr_sop_acc.avg
                    train_logs['mask_loss'] = tr_mask_loss.avg
                    train_logs['sop_loss'] = tr_sop_loss.avg
                    show_info = f'[Training]:[{epoch}/{opt.epochs}]{global_step}/{num_train_optimization_steps} ' \
                                f'- ETA: {eta_format}' + "-".join(
                        [f' {key}: {value:.4f} ' for key, value in train_logs.items()])
                    logger.info(show_info)
                    tr_mask_acc.reset()
                    tr_sop_acc.reset()
                    tr_loss.reset()
                    tr_mask_loss.reset()
                    tr_sop_loss.reset()
                    start_time = now

                if global_step % opt.num_save_steps == 0:
                    if opt.local_rank in [-1, 0] and opt.num_save_steps > 0:
                        # Save model checkpoint
                        output_dir = opt.output_dir / f'lm-checkpoint-{global_step}'
                        if not output_dir.exists():
                            output_dir.mkdir()
                        # save model
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(str(output_dir))
                        torch.save(opt, str(output_dir / 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        # save config
                        output_config_file = output_dir / "config.json"
                        with open(str(output_config_file), 'w') as f:
                            f.write(model_to_save.config.to_json_string())
