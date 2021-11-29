# vitaLITy: creating GloVe and Specter document embeddings

## Requirements

- `Python 3.8` - [Link](https://www.python.org/) - tested on Python 3.8.11
- `pip` - [Link](https://pypi.org/project/pip/) - package installer for Python
- `venv` - [Link](https://docs.python.org/3/library/venv.html) - Serves files in virtual environment

## Setup

- Open the command line/terminal on your machine and navigate to this project's top-level directory (i.e. where this file is)

### Windows

1. `py -3.8 -m venv venv_embed` - create a python3 virtual environment called _venv\_embed_ in the current directory
2. `venv_embed\Scripts\activate.bat` - enters the virtual environment
   - **FROM THIS POINT ON: only use `python` command to invoke interpeter, avoid using global command `py`!!**
3. `python -m pip install -r requirements.txt` - installs required libraries local to this project environment

### MacOS/Linux
1. `python3.8 -m venv venv_embed` / `virtualenv --python=python3.8 venv_embed` - create a python3 virtual environment called venv_embed
2. `source venv_embed/bin/activate` - enters the virtual environment
   - **FROM THIS POINT ON: only use `python` command to invoke interpeter, avoid using global command `python3.8`!!**
3. `brew install gcc`
4. `export CC=/usr/local/Cellar/gcc/11.2.0/bin/g++-11` This will be different for different users/systems.
5. `export CFLAGS="-Wa,-q"`
6. `python -m pip install -r requirements.txt` - installs required libraries local to this project environment
7. `python -m spacy download en_core_web_sm`
8. `python -m nltk.downloader popular`

## Run

1. Configure file paths in `config.py`. Sample data files are provided:
   - `data/sample-dataset-sans-embeddings.tsv` - the output file from the [scraper](https://github.com/vitality-vis/scraper) module as the input file to compute embeddings.
   - `data/sample-dataset-with-embeddings.tsv` - the output file with computed embeddings.
2. Run `python embed.py`
...
3. `deactivate` - exits the virtual environment


### Credits
vitaLITy was created by 
<a target="_blank" href="https://www.cc.gatech.edu/~anarechania3">Arpit Narechania</a>, <a target="_blank" href="https://www.karduni.com/">Alireza Karduni</a>, <a target="_blank" href="https://wesslen.netlify.app/">Ryan Wesslen</a>, and <a target="_blank" href="https://emilywall.github.io/">Emily Wall</a>.

##### Citation
```bibTeX
@article{narechania2021vitality,
  title={vitaLITy: Promoting Serendipitous Discovery of Academic Literature with Transformers \& Visual Analytics},
  author={Narechania, Arpit and Karduni, Alireza and Wesslen, Ryan and Wall, Emily},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2021},
  doi={10.1109/TVCG.2021.3114820},
  publisher={IEEE}
}
```

### License
The software is available under the [MIT License](https://github.com/vitality-vis/embed/blob/master/LICENSE).

### Contact
If you have any questions, feel free to [open an issue](https://github.com/vitality-vis/embed/issues/new/choose) or contact [Arpit Narechania](https://www.cc.gatech.edu/~anarechania3).
