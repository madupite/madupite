* This directory contains a sample configuration of VS Code.
* If you don't have your preferred configuration in place yet, you can just
    ```
    cp -nrvT . ../.vscode
    ```
    - Remove `-n` to overwrite existing files;
    - but merging using a diff tool is recommended.
* `launch.json` adds a debugging configuration
* `tasks.json` adds a couple of tasks:
    - `Run make` runs just `make all`.
    - `Run make with selected command` gives a menu of make target to select from.
* `run_make.sh` is a wrapper script used by `tasks.json`. In particular, it
    - loads the PETSc environment using the env file (preset to `${fileWorkspaceFolder}/petsc-env`),
    - makes sure that `compile_commands.json` for Clangd is produced.
* `settings.json` contains a set of useful editor settings.
