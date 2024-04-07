For documentation: install sphinx in conda otherwise it does not find madupite

To build madupite
```bash
conda env create -f environment.yml
conda activate madupiteenv
pip install .
```

to test: `python example/idm.py` and `mpirun -n 4 python example/idm.py`


In case the python example doesn't work (probably some linking error):
* try to run `pip install .` again
* or manually set path for linker, sth. along these lines: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/robin/repos/madupite/madupite`
