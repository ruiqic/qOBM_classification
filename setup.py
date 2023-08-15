from setuptools import setup, find_packages

def get_requirements(path: str):
    return [l.strip() for l in open(path)]

setup(
    name='qOBM_classification',
    version='0.1.1',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
