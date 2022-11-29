from pyvi import ViTokenizer
from underthesea import text_normalize, sent_tokenize

import argparse


def run(inpath: str, outpath: str):
    count = 0
    with open(outpath, "w", encoding="utf-8") as fw:
        for line in open(inpath, "r", encoding="utf-8").readlines():

            line = line.strip()
            if line == '':
                continue
            norm_line = text_normalize(line)
            norm_sents = sent_tokenize(norm_line)
            tokenized_sents = [ViTokenizer.tokenize(norm_sent) for norm_sent in norm_sents]
            tokenized_sents = "\n".join(tokenized_sents)
            fw.write(tokenized_sents + "\n\n")
            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} lines")
                print(f"Result text: ")
                print(f"\t{tokenized_sents}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--corpus', type=str, required=True,
                        help='The path of corpus file.')
    parser.add_argument('--output', type=str, required=True,
                        help='The path is used to store normalized corpus.')

    args = parser.parse_args()
    run(args.corpus, args.output)

