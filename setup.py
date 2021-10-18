import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="StochSynthSample",
    version="0.0.1",
    author="Eli Weinstein",
    author_email="eweinstein@g.harvard.edu",
    description="A package for designing stochastic synthesis procedures based on generative models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EWeinstein/stochastic-synthesis-samplers",
    packages=setuptools.find_packages()
)
