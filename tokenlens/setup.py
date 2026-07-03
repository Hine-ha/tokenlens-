from setuptools import setup

setup(
    name="tokenlens",
    version="0.1.5",
    packages=["tokenlens"],
    package_dir={"tokenlens": "."},
    install_requires=["requests"],
)
