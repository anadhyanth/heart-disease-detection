from setuptools import setup, find_packages

setup(
    name="heart-disease-detection",
    version="1.0.0",
    description="Heart Disease Detection using Machine Learning",
    author="B. Anadhyanth",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "joblib"
    ]
)