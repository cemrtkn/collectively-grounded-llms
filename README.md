# Collectively-Grounded-LLMs

Generative AI systems like large language models (LLMs) are changing how we produce and share language. As these systems become more common, two main worries have emerged. One is the fear of global sameness, where centralized models erase cultural and linguistic differences. The other is the fear of extreme personalization, where AI adapts so much to each person that shared ways of speaking and understanding start to disappear.

This project explores a third, often overlooked possibility: using generative AI to support collective identity. **Instead of making models that speak the same way to everyone—or that tailor their responses to each individual—we suggest designing them to reflect social groups: communities, movements, and publics.** These groups have long shaped language and culture by offering shared values, perspectives, and ways of speaking.

We argue that collectives offer the right balance for AI design—not too general, like global models, and not too narrow, like personal ones. People are shaped by personal experiences, but it is through collectives that ideas grow, are tested, and refined through conversation and feedback.

Cultural evolution also depends on diverse and competing ways of thinking. This diversity often emerges through group-level processes: groups form shared identities, coordinate their behavior, and create new norms. Supporting a variety of collectives through AI could help keep culture dynamic, rather than making it more uniform.

While today’s LLMs can imitate social groups through prompting, these imitations are often based on outsider stereotypes rather than the group’s own language. **Our project takes a different approach: we fine-tune models on real data from specific communities and train the system to recognize special tokens that represent different collective identities.** This allows users to choose which group’s voice the chatbot should adopt.

By anchoring AI in real communities, we treat it not just as a neutral tool, but as a cultural medium—able to carry and express the values, styles, and ways of speaking of actual groups. This could lead to more context-aware and pluralistic AI systems that support, rather than erase, cultural diversity.

## Setup

### On Raven cluster
1. load python version `module load python-waterboa/2024.06`
2. Build a venv with a descriptive name (there are different requirement packages) `python -m venv <<descriptive_venv_name>>`
3. Find virtual environment: `source <<descriptive_venv_name>>/bin/activate`
5. Activate virtual environment: `source <<activate_file>>`
6. Install sft "src" directory as a package that can be imported from outside `python setup.py install`
7. Check different requirement packages at setup.py

## Run Project

The documentation on how to use this project is to be found in [doc](doc/) folder. Available guides are

- ([How to run an example sft](docs/fine_tuning.md))
- ([run_slurm.py explanation](docs/run_slurm.md))
- ([Interactive dev session on Raven](docs/get_ipython_shell_on_raven.md))


## Running Tests (not active atm)

Execute tests using pytest:
```bash
poetry run pytest
```
or if you want to get coverage report
```bash
poetry run coverage run --source=src -m pytest
&& poetry run coverage report
```

## Continuous Integration (not active atm)

The CI  system automatically runs code quality checks and tests on every push and pull request. It verifies code formatting, runs pre-commit hooks, executes the test suite, and ensures test coverage meets the 90% minimum requirement.

## Contributing

Please read [contributing.md](contributing.md) for guidelines on how to
contribute to this codebase.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE)
file.