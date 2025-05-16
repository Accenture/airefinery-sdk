"""setup.py: setuptools control."""

from setuptools import find_packages, setup

VERSION = "1.8.0"


setup(
    name="airefinery-sdk",
    packages=find_packages(),
    install_requires=[
        "omegaconf>=2.2.3",
        "requests>=2.32.2",
        "asyncpg>=0.30.0",
        "aiohttp[speedups]>=3.11.0",
        "pillow>=11.0.0",
        "websockets>=13.0",
        "openai>=1.57.0",
        "fastapi>=0.115.0",
        "tenacity>=9.0.0",
        "uvicorn>=0.32.0",
        "setuptools>=75.2.0",
        "numpy>=2.1.2",
        "pandas>=2.2.3",
        "google-genai>=1.9.0",
        "google-cloud-aiplatform[agent_engines]>=1.87.0",
        "azure-identity>=1.19.0",
        "azure-ai-projects==1.0.0b8",
        "mcp>=1.6.0",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "air = air.main:main",
        ]
    },
    version=VERSION,
    description="AI Refinery SDK",
    author="AI Refinery",
)
