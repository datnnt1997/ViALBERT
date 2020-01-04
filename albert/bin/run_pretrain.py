from albert.inputters.tokenization import FullTokenizer

def train(configs):
    tokenizer = FullTokenizer(vocab_file=configs.vocab_path, do_lower_case=configs.do_lower_case)
