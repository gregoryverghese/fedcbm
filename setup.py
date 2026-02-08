from setuptools import setup, find_packages

setup(
    name="minotaur",
    version="0.1.0",
    description="Multimodal learnIng cliNical cOncepts Transparent pAn-cancer sURvival",
    packages=find_packages(),
    install_requires=[
        # Add your requirements here
        # "torch>=1.10.0",
        # "pytorch-lightning>=1.5.0",
        # etc.
    ],
    python_requires=">=3.8",
)


