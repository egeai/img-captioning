from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from typing import List


class Vocabulary(object):
    """Simple vocabulary wrapper"""
    def __init__(self, vocabs: List):
        self.tokenizer = get_tokenizer('basic_english')
        self.vocabs = vocabs

    def yield_tokens(self):
        for vocab in self.vocabs:
            tokens = self.tokenizer(vocab)
            yield tokens

    def build_vocabulary(self) -> Vocab:
        token_generator = self.yield_tokens()
        return build_vocab_from_iterator(token_generator)



# vocab = build_vocab_from_iterator(yield_tokens(file_path), specials=["<unk>"]



