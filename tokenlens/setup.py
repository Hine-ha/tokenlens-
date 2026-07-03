from setuptools import setup

setup(
    name="tokenlens",
    version="0.1.3",
    packages=["tokenlens"],
    package_dir={"tokenlens": "."},
    install_requires=["requests"],
)
