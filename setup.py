from setuptools import find_packages, setup


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


reqs = parse_requirements("requirements.txt")


with open("README.md") as f:
    long_description = f.read()


setup(
    name="onnx2keras",
    version="0.0.24",
    description="The deep learning models converter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gmalivenko/onnx2keras",
    author="Grigory Malivenko",
    author_email="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="machine-learning deep-learning pytorch keras neuralnetwork vgg resnet "
    "densenet drn dpn darknet squeezenet mobilenet",
    license="MIT",
    packages=find_packages(),
    install_requires=reqs,
    zip_safe=False,
)
