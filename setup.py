from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()
    
with open("requirements.txt", "r") as f:
    INSTALL_REQUIRES = [line.strip() for line in f.readlines()]

setup(
    name='taxicab_st',
    version='0.0.13',
    author='Nathan A. Rooy, Raven van Ewijk',
    author_email='nathanrooy@gmail.com, ravenvanewijk1@gmail.com',
    url='https://github.com/ravenvanewijk/taxicab-st',
    description='Accurate time based routing for Open Street Maps and OSMnx',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ]
)
