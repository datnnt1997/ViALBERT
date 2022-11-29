# albert_vi_pytorch-master
ALBERT for Vietnamese

## Corpus
Albert model was training on Vietnamese wikipedia corpus which was preprocessed and extracted from [Wikipedia](https://github.com/NTT123/viwik18/tree/viwik19). 

## Clean and normalize corpus
```bash
python clean_and_normalize_corpus.py \
--corpus viwik19.txt \
--output norm_viwik19.txt
```
## Tokenizer
```bash
spm_train --input norm_viwik19.txt \
--model_prefix=30k-vi-clean --vocab_size=30000 \
--pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 \
--control_symbols=[CLS],[SEP],[MASK] \
--user_defined_symbols="(,),\",-,.,–,£,€" \
--shuffle_input_sentence=true --character_coverage=0.99995 \
--model_type=unigram
```

## Prepare pretraining data
```bash
python create_pretraining_data.py \
  --input_file=norm_viwik19.txt \
  --output_file=viwik19.tfrecord \
  --vocab_file resources/30k-vi-clean.vocab \
  --spm_model_file resources/30k-vi-clean.model \
  --do_lower_case=False --dupe_factor=10
```