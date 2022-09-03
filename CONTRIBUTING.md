# How to contribute

## Dependencies

The easiest way to have the environment setup is using conda.
Conda is the most recommended package manager for scientific
and data python stack.

If you don't have it installed, you can install it via mambaforge:
https://github.com/conda-forge/miniforge#mambaforge

Mamba is a fastest layer on top of conda that will fast the way
conda works.

When you have the conda installed, you can proceed and create
a new conda environment for epigraphhub.

```bash
mamba env create --file conda/dev.yaml
```

This command will create a new conda environment with some development
dependencies for epigraphhub. In order to proceed, you will need first
activate your environment:

```bash
conda activate epigraphhubpy
```

Now, you can install the epigraphhub dependencies with:

```bash
poetry install
```

## Configuration file

`epigraphhub_py` needs a configuration file in order to access the database.

After you have the library installed, you can run the following command
(just an example):

```bash
epigraphhub-config \
  --db-host localhost \
  --db-port 25432 \
  --db-credential "epigraph_public:dev_epigraphhub/dev_epigraph/dev_epigraph" \
  --db-credential "epigraph_private:dev_privatehub/dev_epigraph/dev_epigraph" \
  --db-credential "epigraph_sandbox:dev_sandbox/dev_epigraph/dev_epigraph"
```

We need to have connection for all these 3 databases:
  - epigraphhub_public,
  - epigraphhub_private
  - epigraphhub_sandbox

If you don't have this databases yet, you can create that locally using docker:

```bash
```

## Codestyle

After installation you may execute code formatting.

```bash
make codestyle
```

### Checks

Many checks are configured for this project. Command `make check-codestyle` will check black, isort and darglint.
The `make check-safety` command will look at the security of your code.

Comand `make lint` applies all checks.

### Before submitting

Before submitting your code please do the following steps:

1. Add any changes you want
1. Add tests for the new changes
1. Edit documentation if you have changed something significant
1. Run `make codestyle` to format your changes.
1. Run `make lint` to ensure that types, security and docstrings are okay.

## Other help

You can contribute by spreading a word about this library.
It would also be a huge contribution to write
a short article on how you are using this project.
You can also share your best practices with us.
