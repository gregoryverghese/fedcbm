from setuptools import setup, find_packages

setup(
    name="fedcbm",
    version="0.1.0",
    description="Federated Concept Bottleneck Model for pan-cancer survival prediction",
    packages=find_packages(),
    install_requires=[
        # Add your requirements here
        # "torch>=1.10.0",
        # "pytorch-lightning>=1.5.0",
        # etc.
    ],
    python_requires=">=3.8",
)


