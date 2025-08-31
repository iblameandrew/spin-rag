from setuptools import setup, find_packages
import os

# Function to read the contents of the requirements file
def load_requirements(filename='requirements.txt'):
    with open(filename, 'r') as f:
        return f.read().splitlines()

# Read the contents of your README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='spinrag',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A demonstration of the SpinRAG concept with a Dash web interface.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/spinrag',  # Replace with your project's URL
    py_modules=['spinlm'],  # This tells setuptools to include spinlm.py
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Framework :: Dash',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=load_requirements(),
    include_package_data=True,
)