# GAME: Gold-Standard-Agnostic Metric for Evaluation of Code-Mixed Sentences

This repository contains the implementation of GAME, introduced in our paper:

**Multilingual Controlled Generation And Gold-Standard-Agnostic Evaluation of Code-Mixed Sentences**
Ayushman Gupta, Aryan Bhogal, Kripa Ghosh
arXiv: https://arxiv.org/abs/2410.10580

## Citation

If you use this code, please cite:
```bibtex
@misc{gupta2024multilingualcontrolledgenerationgoldstandardagnostic,
      title={Multilingual Controlled Generation And Gold-Standard-Agnostic Evaluation of Code-Mixed Sentences}, 
      author={Ayushman Gupta and Akhil Bhogal and Kripabandhu Ghosh},
      year={2024},
      eprint={2410.10580},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.10580}, 
}
```

## Usage

The codebase is straightforward to use:

1. **Model settings** (API keys, LLM choice, generation parameters) can be changed in `config.py`
2. **Input sentences** (reference sentence and code-mixed sentence to evaluate) can be set in `main.py`

Then simply run:
```bash
python main.py
```

## Setup
```bash
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key:
```
API_KEY=your_key_here
```
