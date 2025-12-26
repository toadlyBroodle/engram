#!/usr/bin/env python3
"""
Setup script for Engram

A passive memory layer for AI conversations with vector-based storage and context-aware retrieval.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="engram",
    version="2.0.0",
    author="Engram Team",
    author_email="engram@ai.memory",
    description="Engram - Passive memory layer for AI conversations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/engram",

    packages=find_packages(exclude=['test', 'test.*', 'docs']),
    include_package_data=True,

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    python_requires=">=3.8",

    install_requires=read_requirements('requirements.txt'),

    extras_require={
        "dev": [
            "pytest>=7.0.0,<8.0.0",
            "black>=22.0.0,<23.0.0",
            "mypy>=1.0.0,<2.0.0",
        ],
        "gpu": [
            "torch[cu118]>=1.11.0,<3.0.0",
            "faiss-gpu>=1.7.0,<2.0.0",
        ],
    },

    entry_points={
        "console_scripts": [
            "engram=engram_pkg.cli:main",
        ],
    },

    keywords="ai memory llm vector-database embeddings machine-learning",

    project_urls={
        "Bug Reports": "https://github.com/your-repo/engram/issues",
        "Source": "https://github.com/your-repo/engram",
        "Documentation": "https://github.com/your-repo/engram#readme",
    },

    zip_safe=False,
)
