from setuptools import find_packages, setup

setup(
    name="highfrequency",
    version="0.1.0",
    packages=find_packages(),
    description="Implementation of the HighFrequency R package in Python.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Patrick Lucescu",
    author_email="patricklucescu@outlook.com",
    license="MIT",
    install_requires=["polars", "pytest", "pandas", "numpy"],
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
    ],
    python_requires=">=3.12",
)
