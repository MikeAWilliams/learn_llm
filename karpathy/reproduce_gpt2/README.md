# following allong with Andrej Karpathy's Let's Reproduce GPT-

https://youtu.be/l8pRSuU81PU?si=XvJehsYm13tIjahx

## Requirements

``` bash
pip install -r requirements.txt
sudo apt-get install python3-dev
```

## usage

notebook_zed.py uses zed REPL comments.

### Main script

Trains the model

multiple gpu mode
```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```
single gpu mode
```bash
python train_gpt2.py
```

### fineweb.py

Downloads and tokenizes the data and saves data shards to disk.
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Run simply as:
``` bash
$ python fineweb.py
```
