from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-agent-swarm",
    version="0.1.0",
    author="Gabe",
    description="A system for managing and coordinating multiple AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabubu-dev/ai-agent-swarm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.5.3",
        "openai>=1.10.0",
        "anthropic>=0.18.1",
        "redis>=5.0.1",
        "aioredis>=2.0.1",
        "python-dotenv>=1.0.0",
        "httpx>=0.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-asyncio>=0.23.3",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.12.1",
            "ruff>=0.1.11",
            "mypy>=1.8.0",
        ],
    },
)
