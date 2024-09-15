This folder contains scripts to run `madupite` on Euler, ETH Zurich's HPC cluster. Refer to the [official documentation](https://scicomp.ethz.ch/wiki/Euler) for more information on how to use Euler. Read the [`madupite` documentation](https://madupite.github.io/install.html) for more information on how to install `madupite` on Euler.

In brief:
```bash
git clone https://github.com/madupite/madupite.git
cd madupite
source examples/euler/euler-load-modules.sh
pip install .
sbatch examples/euler/euler-launch.sh
```