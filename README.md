# vitaLITy: creating GloVe and Specter document embeddings

## Requirements

- `Python 3.9` - [Link](https://www.python.org/) - tested on Python 3.9 on MacOSX Sonoma
- `pip` - [Link](https://pypi.org/project/pip/) - package installer for Python
- `venv` - [Link](https://docs.python.org/3/library/venv.html) - Serves files in virtual environment

## Setup
- Create and activate a Python virtual environment. We have tested using Python3.9.
- `brew install gcc`
- `export CC=/opt/homebrew/Cellar/gcc/14.1.0_2/bin/g++-14` This will be different for different users/systems.
- `export CFLAGS="-Wa,-q"`
- `pip install --upgrade pip setuptools wheel`
- `python -m pip install -r requirements.txt`
- `python -m spacy download en_core_web_sm`
- `python -m nltk.downloader popular`

Note: `pip cache purge` might be needed sometimes to start fresh installation.

## Run

- Configure file paths in `config.py`. Sample data files are provided:
   - `data/sample-dataset-sans-embeddings.tsv` - the output file from the [scraper](https://github.com/vitality-vis/scraper) module as the input file to compute embeddings.
   - `data/sample-dataset-with-embeddings.tsv` - the output file with computed embeddings.
- Run `python embed.py`


### Credits
vitaLITy was created by 
<a target="_blank" href="https://arpitnarechania.github.io">Arpit Narechania</a>, <a target="_blank" href="https://www.karduni.com/">Alireza Karduni</a>, <a target="_blank" href="https://wesslen.netlify.app/">Ryan Wesslen</a>, and <a target="_blank" href="https://emilywall.github.io/">Emily Wall</a>.


### Citation
```bibTeX
@article{narechania2021vitality,
  title={vitaLITy: Promoting Serendipitous Discovery of Academic Literature with Transformers \& Visual Analytics},
  author={Narechania, Arpit and Karduni, Alireza and Wesslen, Ryan and Wall, Emily},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2022},
  doi={10.1109/TVCG.2021.3114820},
  publisher={IEEE}
}
```

### License
The software is available under the [MIT License](https://github.com/vitality-vis/embed/blob/master/LICENSE).


### Contact
If you have any questions, feel free to [open an issue](https://github.com/vitality-vis/embed/issues/new/choose) or contact [Arpit Narechania](https://arpitnarechania.github.io).
