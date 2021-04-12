import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fred", # Replace with your own username
    version="0.0",
    author="Maarten J. van den Broek",
    author_email="m.j.vandenbroek@tudelft.nl",
    description="FRED",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    # project_urls={
    #     "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "fred"},
    packages=setuptools.find_packages(where="fred"),
    python_requires=">=3.6",
)