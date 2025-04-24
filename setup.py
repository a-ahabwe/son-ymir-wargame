from setuptools import setup, find_packages

setup(
    name="veto-game",
    version="0.1.0",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.19.0',
        'pygame>=2.0.0',
        'torch>=1.7.0',
        'stable-baselines3>=1.0',
        'joblib>=1.0.0',
    ],
)
