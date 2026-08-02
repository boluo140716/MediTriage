"""
Setup script for MediTriage agent（可编辑安装为 meditriage 包）
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="meditriage-agent",
    version="0.1.0",
    author="MediTriage Team",
    description="Multi-agent medical assistant system based on MediX-R1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/...",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={"test": ["pytest"]},
)
