# epigraphhub_py

<div align="center">

[![Build status](https://github.com/thegraphnetwork/epigraphhub_py/workflows/build/badge.svg?branch=master&event=push)](https://github.com/thegraphnetwork/epigraphhub_py/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/epigraphhub_py.svg)](https://pypi.org/project/epigraphhub_py/)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/thegraphnetwork/epigraphhub_py/releases)
[![License](https://img.shields.io/github/license/thegraphnetwork/epigraphhub_py)](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/LICENSE)
![Coverage Report](assets/images/coverage.svg)

Epigraphhub Python package

</div>

## Very first steps

### Initialize your code

1. Initialize `git` inside your repo:

```bash
cd epigraphhub && git init
```

2. If you don't have `Poetry` installed run:

```bash
make poetry-download
```

3. Initialize poetry and install `pre-commit` hooks:

```bash
make install
make pre-commit-install
```

4. Run the codestyle:

```bash
make codestyle
```

5. Upload initial code to GitHub:

```bash
git add .
git commit -m ":tada: Initial commit"
git branch -M main
git remote add origin https://github.com/thegraphnetwork/epigraphhub_py.git
git push -u origin main
```

### Set up bots

- Set up [Stale bot](https://github.com/apps/stale) for automatic issue closing.

### Poetry

Want to know more about Poetry? Check [its documentation](https://python-poetry.org/docs/).

<details>
<summary>Details about Poetry</summary>
<p>

Poetry's [commands](https://python-poetry.org/docs/cli/#commands) are very intuitive and easy to learn, like:

- `poetry add numpy@latest`
- `poetry run pytest`
- `poetry publish --build`

etc
</p>
</details>

### Building and releasing your package

Building a new version of the application contains steps:

- Bump the version of your package `poetry version <version>`. You can pass the new version explicitly, or a rule such as `major`, `minor`, or `patch`. For more details, refer to the [Semantic Versions](https://semver.org/) standard.
- Make a commit to `GitHub`.
- Create a `GitHub release`.
- And... publish 🙂 `poetry publish --build`

## 🎯 What's next

Well, that's up to you 💪🏻. I can only recommend the packages and articles that helped me.

- [`Typer`](https://github.com/tiangolo/typer) is great for creating CLI applications.
- [`Rich`](https://github.com/willmcgugan/rich) makes it easy to add beautiful formatting in the terminal.
- [`Pydantic`](https://github.com/samuelcolvin/pydantic/) – data validation and settings management using Python type hinting.
- [`Loguru`](https://github.com/Delgan/loguru) makes logging (stupidly) simple.
- [`tqdm`](https://github.com/tqdm/tqdm) – fast, extensible progress bar for Python and CLI.
- [`IceCream`](https://github.com/gruns/icecream) is a little library for sweet and creamy debugging.
- [`orjson`](https://github.com/ijl/orjson) – ultra fast JSON parsing library.
- [`Returns`](https://github.com/dry-python/returns) makes you function's output meaningful, typed, and safe!
- [`Hydra`](https://github.com/facebookresearch/hydra) is a framework for elegantly configuring complex applications.
- [`FastAPI`](https://github.com/tiangolo/fastapi) is a type-driven asynchronous web framework.

Articles:

- [Open Source Guides](https://opensource.guide/).
- [A handy guide to financial support for open source](https://github.com/nayafia/lemonade-stand)
- [GitHub Actions Documentation](https://help.github.com/en/actions).
- Maybe you would like to add [gitmoji](https://gitmoji.carloscuesta.me/) to commit names. This is really funny. 😄

## 🚀 Features

### Development features

- Supports for `Python 3.9` and higher.
- [`Poetry`](https://python-poetry.org/) as the dependencies manager. See configuration in [`pyproject.toml`](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/pyproject.toml) and [`setup.cfg`](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/setup.cfg).
- Automatic codestyle with [`black`](https://github.com/psf/black), [`isort`](https://github.com/timothycrosley/isort) and [`pyupgrade`](https://github.com/asottile/pyupgrade).
- Ready-to-use [`pre-commit`](https://pre-commit.com/) hooks with code-formatting.
- Type checks with [`mypy`](https://mypy.readthedocs.io); docstring checks with [`darglint`](https://github.com/terrencepreilly/darglint); security checks with [`safety`](https://github.com/pyupio/safety) and [`bandit`](https://github.com/PyCQA/bandit)
- Testing with [`pytest`](https://docs.pytest.org/en/latest/).
- Ready-to-use [`.editorconfig`](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/.editorconfig), [`.dockerignore`](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/.dockerignore), and [`.gitignore`](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/.gitignore). You don't have to worry about those things.

### Deployment features

- `GitHub` integration: issue and pr templates.
- `Github Actions` with predefined [build workflow](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/.github/workflows/build.yml) as the default CI/CD.
- Everything is already set up for security checks, codestyle checks, code formatting, testing, linting, docker builds, etc with [`Makefile`](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/Makefile#L89). More details in [makefile-usage](#makefile-usage).
- [Dockerfile](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/docker/Dockerfile) for your package.
- Automatic drafts of new releases with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). You may see the list of labels in [`release-drafter.yml`](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/.github/release-drafter.yml). Works perfectly with [Semantic Versions](https://semver.org/) specification.

### Open source community features

- Ready-to-use [Pull Requests templates](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/.github/PULL_REQUEST_TEMPLATE.md) and several [Issue templates](https://github.com/thegraphnetwork/epigraphhub_py/tree/master/.github/ISSUE_TEMPLATE).
- Files such as: `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` are generated automatically.
- [`Stale bot`](https://github.com/apps/stale) that closes abandoned issues after a period of inactivity. (You will only [need to setup free plan](https://github.com/marketplace/stale)). Configuration is [here](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/.github/.stale.yml).
- [Semantic Versions](https://semver.org/) specification with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter).

## Installation

```bash
pip install -U epigraphhub
```

or install with `Poetry`

```bash
poetry add epigraphhub
```

Then you can run

```bash
epigraphhub --help
```

or with `Poetry`:

```bash
poetry run epigraphhub --help
```

### Makefile usage

[`Makefile`](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/Makefile) contains a lot of functions for faster development.

## Configuration file

`epigraphhub_py` needs a configuration file in order to access the database.

After you have the library installed, you can run the following command
(just an example):

```bash
epigraphhub-config \
  --db-host localhost \
  --db-port 25432 \
  --db-credential "public:dev_epigraphhub/dev_epigraph/dev_epigraph"
```
We need to have connection for all these 3 databases:
  - public,
  - private
  - sandbox

## 📈 Releases

You can see the list of available releases on the [GitHub Releases](https://github.com/thegraphnetwork/epigraphhub_py/releases) page.

We follow [Semantic Versions](https://semver.org/) specification.

We use [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). As pull requests are merged, a draft release is kept up-to-date listing the changes, ready to publish when you’re ready. With the categories option, you can categorize pull requests in release notes using labels.

### List of labels and corresponding titles

|               **Label**               |  **Title in Releases**  |
| :-----------------------------------: | :---------------------: |
|       `enhancement`, `feature`        |       🚀 Features       |
| `bug`, `refactoring`, `bugfix`, `fix` | 🔧 Fixes & Refactoring  |
|       `build`, `ci`, `testing`        | 📦 Build System & CI/CD |
|              `breaking`               |   💥 Breaking Changes   |
|            `documentation`            |    📝 Documentation     |
|            `dependencies`             | ⬆️ Dependencies updates |

You can update it in [`release-drafter.yml`](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/.github/release-drafter.yml).

GitHub creates the `bug`, `enhancement`, and `documentation` labels for you. Create the remaining labels on the Issues tab of your GitHub repository, when you need them.

## 🛡 License

[![License](https://img.shields.io/github/license/thegraphnetwork/epigraphhub_py)](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/LICENSE)

This project is licensed under the terms of the `GNU GPL v3.0` license. See [LICENSE](https://github.com/thegraphnetwork/epigraphhub_py/blob/master/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{epigraphhub_py,
  author = {thegraphnetwork},
  title = {Epigraphhub Python package},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/thegraphnetwork/epigraphhub_py}}
}
```

## Credits [![🚀 Your next Python package needs a bleeding-edge project structure.](https://img.shields.io/badge/python--package--template-%F0%9F%9A%80-brightgreen)](https://github.com/TezRomacH/python-package-template)

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template)
