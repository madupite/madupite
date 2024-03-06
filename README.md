For documentation: install sphinx in conda otherwise it does not find madupite

To build madupite
```bash
conda env create -f environment.yml
conda activate madupiteenv
pip install .
```

to test: `python example/idm.py` and `mpirun -n 4 python example/idm.py`