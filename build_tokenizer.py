#!/usr/bin/env python3

"""
Script to build and train a Byte-Pair Encoding (BPE) tokenizer using the Huggingface Tokenizers library.

https://github.com/huggingface/tokenizers

"""

from pathlib import Path
import transformers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


tokenizer = Tokenizer(BPE())

tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()


# get IWSLT files
files = [str(x) for x in Path("iwslt/de-en/").glob("*[!s].de-en.[ed][ne]")]
# remove test files
files.remove('iwslt/de-en/IWSLT15.TED.tst2012.de-en.en')
files.remove('iwslt/de-en/IWSLT15.TED.tst2012.de-en.de')


trainer = BpeTrainer(vocab_size=52000, min_frequency=2,
                     special_tokens=["[UNK]", "[SOS]", "[SEP]", "[EOS]", "[PAD]"])

tokenizer.train(trainer, files)


files = tokenizer.model.save("tokenizer/", "tokenizer_iwslt")
tokenizer.model = BPE.from_file(*files, unk_token="[UNK]")


# specify template for processing single sentences and pairs of sentences with special tokens
tokenizer.post_processor = TemplateProcessing(
    single="[SOS] $A [EOS]",
    pair="[SOS] $A [EOS] [SEP] [SOS]:1 $B:1 [EOS]:1",
    special_tokens=[
        ("[SOS]", tokenizer.token_to_id("[SOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)


# padding length is maximum length of source/target sentence (= max_sent_len)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id(
    "[PAD]"), pad_token="[PAD]", length=128)


tokenizer.save("tokenizer/tokenizer_en_de.json")

