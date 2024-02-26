install sphinx in conda otherwise it does not find madupite


To build madupite
```bash
rm -rf cmake-build
mkdir cmake-build
cd cmake-build
cmake ..
make
cp lib/libmadupite.so ../madupite/libmadupite.so
cd ..
pip install .
```

to test: `python example/idm.py`
