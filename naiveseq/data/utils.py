import re
import torchtext

SPACE_NORMALIZER = re.compile(r"\s+")


def naive_tokenizer(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

def spacy_tokenizer(line, language: str='en'):
    return torchtext.data.utils.get_tokenizer("spacy", language=language)

