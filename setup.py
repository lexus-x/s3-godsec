"""
SE(3)-VLA: Riemannian Flow Matching on Lie Groups for Vision-Language-Action Models.

Install:
    pip install -e .

    # With SmolVLA support:
    pip install -e ".[smolvla]"

    # With benchmark support:
    pip install -e ".[benchmarks]"

    # Full development install:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages

setup(
    name="se3-vla",
    version="0.3.0",
    description="SE(3) Flow Matching Action Head for Vision-Language-Action Models",
    author="SE(3)-VLA Team",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "geoopt>=0.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "smolvla": [
            "lerobot @ git+https://github.com/huggingface/lerobot.git",
        ],
        "benchmarks": [
            "libero",
            "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "tensorboard>=2.13.0",
            "seaborn>=0.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
