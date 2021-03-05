# Decoder-Only Transformer for NMT
This project for the course "Introduction to Neural Networks and Sequence-to-Sequence Learning" at Heidelberg University implements a decoder-only [Transformer (Vaswani et al.,2017)](https://arxiv.org/pdf/1706.03762.pdf) model with self-attention for Neural Machine Translation.

## Requirements
The project was created using:
* Python 3.8.5
* PyTorch 1.7.1
* torchtext 0.8.1
* [Transformers](https://github.com/huggingface/transformers) 4.1.1 by Huggingface
* [Tokenizers](https://github.com/huggingface/tokenizers) 0.9.4 by Huggingface

## Data
The IWSLT 2015 German-English dataset was downloaded [here](https://wit3.fbk.eu/2015-01) and can be found in `iwslt/de-en/`.

`dataset.py` contains the class for the dataset.

## Usage

### Data Preprocessing
First run:
```
python3 data_preprocessing.py
```
When executed, it extracts source/target sentences from the xml files into new files.

### Tokenizer
Then build and train a BPE tokenizer:
```
python3 build_tokenizer.py
```
It will create three files in `tokenizer/`:
* `tokenizer_en_de.json`
* `tokenizer_iwslt-merges.txt`
* `tokenizer_iwslt-vocab.json`

Note that padding length in the code must be adjusted if the default value of max_sent_len in `main.py` is changed.

### Training
`decoder_model.py` contains the decoder-only Transformer model.

`training.py` consists of two methods for training and evaluating the model.

To train a new model run and change the following default parameters:
```
python3 main.py --max_sent_len 128 --seq_len 257 --d_model 256 --ff_dim 1024 --n_heads 8 --n_layers 9 --dropout 0.1 --learning_rate 0.00005 --batch_size 32 --n_epochs 50
```
The model will be saved in `model.pt` and train and validation loss will be plotted in `losses_model.jpg`.

### Testing
For testing the model use (with the same parameters chosen in training):
```
python3 inference.py --max_sent_len 128 --seq_len 257 --d_model 256 --ff_dim 1024 --n_heads 8 --n_layers 9 --dropout 0.1 --batch_size 32
```
The script computes the test loss and BLEU score. It can also display generated translations and plot the attention for a translated sentence by uncommenting these lines in main().

### Demo
```
python3 demo.py
```
Asks the user for an input sentence (German). The model saved in `model.pt` will then predict its translation (English).
