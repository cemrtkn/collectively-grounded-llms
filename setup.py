from setuptools import find_packages, setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line]

extra_requirements = {}


def load_extra_requirements(filename, keyword):
    try:
        with open(path.join(here, filename), encoding="utf-8") as f:
            extra_requirements[keyword] = [line.strip() for line in f if line]
    except FileNotFoundError:
        print(f"Optional requirements file {filename} not found.")


load_extra_requirements("requirements_dev.txt", "dev")
load_extra_requirements("requirements_training_fsdp.txt", "training_fsdp")


# 'full' option to cover all the requirements
extra_requirements["full"] = [item for sublist in extra_requirements.values() for item in sublist]

setup(
    name="src",
    version="0.1.0",
    description="",
    author="Center for Humans and Machines",
    author_email="",
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extra_requirements,
)