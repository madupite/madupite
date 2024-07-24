from setuptools import find_packages, setup

# from skbuild import setup

setup(
    name="madupite",
    version="1.0.0",
    description="Python bindings for madupite library",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    cmake_install_dir="madupite",
    cmake_args=["-DCMAKE_BUILD_TYPE=Release"],
    include_package_data=True,
    python_requires=">=3.8",
)
