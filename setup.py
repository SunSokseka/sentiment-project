from setuptools import setup, find_packages

setup(
    name="sentiment_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, e.g.:
        # "numpy",
        # "pandas",
    ],
    include_package_data=True,
    description="A sentiment analysis project.",
    author="Your Name",
    author_email="your.email@example.com",
)