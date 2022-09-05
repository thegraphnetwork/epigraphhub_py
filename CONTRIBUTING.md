# How to contribute

## Dependencies

The easiest way to have the environment setup is using conda.
Conda is the most recommended package manager for scientific
and data python stack.

If you don't have it installed, you can install it via mambaforge:
https://github.com/conda-forge/miniforge#mambaforge

Mamba is a fast layer on top of conda that will speed up the way that
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
  --db-credential "epigraphhub:dev_epigraphhub/dev_epigraph/dev_epigraph"
```
We can gave any number of credentials or connections, it depends on the
database configuration you have. The name for the credential doesn't
matter, but this same name should be used when you are making a connection
to the database.

For development, if you are using the postgres container we have
prepared here, you can run this, instead:

```bash
make create-config
```

This depends on a environment variables file (`.env`) at `docker` folder.
You can create one based on `docker/.env.tpl`. Or you can use this:

```
POSTGRES_HOST=localhost
POSTGRES_PORT=25432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_EPIGRAPH_USER=dev_epigraph
POSTGRES_EPIGRAPH_PASSWORD=dev_epigraph
POSTGRES_EPIGRAPH_DB=dev_epigraphhub
```

This is the default environment variables used by development.

## Database for development

If you don't have the databases for development yet,
you can create one locally using docker:

```bash
make docker-compose build
make docker-compose start
```

## Code Style

After installation you may execute code formatting.

```bash
make linter
```

### Checks

The `make check-safety` command will look at the security of your code.

Comand `make lint` applies all checks.

### Before submitting

Before submitting your code please do the following steps:

1. Run `pre-commit install`.
1. Add any changes you want
1. Add tests for the new changes
1. Edit documentation if you have changed something significant
1. Now, any new commit will run some tools to check and format the code.


## Other help

You can contribute by spreading a word about this library.
It would also be a huge contribution to write
a short article on how you are using this project.
You can also share your best practices with us.
