from setuptools import setup, find_packages

setup(
    name="jaxrl2",
    packages=find_packages(include=["jaxrl2*", "dreamer_v3*"]),
)
