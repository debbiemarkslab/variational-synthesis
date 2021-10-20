import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VariationalSynthesis",
    version="0.0.1",
    author="Anonymous",
    author_email="",
    description="Optimizing stochastic synthesis protocols based on generative models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=['numpy>=1.21.1',
                      'matplotlib>=3.4.2',
                      'torch>=1.9.0',
                      'pyro-ppl>=1.7.0',
                      'pytest>=6.2.4'],
    extras_require={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        'Topic :: Scientific/Engineering :: Computational Biology',
    ],
    python_requires='>=3.9',
    keywords=('biological-sequences proteins probabilistic-models ' +
              'machine-learning'),
)
