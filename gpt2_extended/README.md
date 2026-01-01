# GPT-2 Extended

A GPT-2 implementation and extension project based on Andrej Karpathy's educational video series.

## Origins and Attribution

This project is derived from the code demonstrated in Andrej Karpathy's excellent YouTube video:
**"Let's reproduce GPT-2 (124M)"**
- Video: https://youtu.be/l8pRSuU81PU?si=t3iVNPqauqdfMTD3
- Original implementation by Andrej Karpathy

The original code has been copied and will be modified, extended, and played with for experimentation and learning.

## Project Structure

- `train_gpt2.py` - Main GPT-2 training script with model architecture implementation
- `fineweb.py` - Dataset preparation script for FineWeb-Edu dataset tokenization and processing
- `hellaswag.py` - HellaSwag benchmark evaluation script for testing model performance

## Installation

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```bash
# local dev
python train_gpt2.py
# or on multiple GPUs
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

### Dataset Preparation
```bash
python fineweb.py
```

### Evaluation
```bash
python hellaswag.py 
```

## Acknowledgments

Thanks to Andrej Karpathy for the educational content and clear explanations that made this implementation possible. His approach to teaching complex machine learning concepts through practical implementation is invaluable.

## License

This project is intended for educational purposes. Please refer to the original video and any associated repositories for licensing information regarding the base implementation.
