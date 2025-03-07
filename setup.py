from setuptools import setup, find_packages

setup(
    name="tariff-analyzer",
    version="0.1.0",
    description="System for processing and analyzing tariff policy data",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "networkx>=2.6.0",
        "pyyaml>=6.0",
        "tqdm>=4.61.0",
    ],
    python_requires=">=3.8",
)
