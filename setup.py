from setuptools import setup, find_packages
import os


def load_requirements(filename: str = "requirements.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f.read().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]


with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
    encoding="utf-8",
) as f:
    long_description = f.read()


setup(
    name="spin-rag",
    version="0.1.0a1",
    author="iblameandrew",
    author_email="iblameandrew@users.noreply.github.com",
    description=(
        "SpinRAG: an evolving knowledge-graph RAG that restores incomplete or "
        "damaged documents into self-contained definitions."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iblameandrew/spin-rag",
    project_urls={
        "Source": "https://github.com/iblameandrew/spin-rag",
        "Issues": "https://github.com/iblameandrew/spin-rag/issues",
    },
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*")),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Framework :: Dash",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords=[
        "rag",
        "retrieval-augmented-generation",
        "knowledge-graph",
        "document-restoration",
        "ollama",
        "langchain",
        "llm",
    ],
    python_requires=">=3.8",
    install_requires=load_requirements(),
    include_package_data=True,
    zip_safe=False,
)
