"""
Setup script for Hermes Agent Upgraded
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hermes-agent-upgraded",
    version="2.1.0",
    author="Hermes Agent Team",
    author_email="team@hermes-agent.dev",
    description="Enhanced Hermes Agent with processing power and efficiency modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Everaldtah/hermes-agent-orchestration-brain-core-upgrade",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.16.0",
        ],
        "openrouter": [
            "aiohttp>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hermes-agent=core.hermes_core:main",
        ],
    },
)
