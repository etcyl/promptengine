from setuptools import setup, find_packages

setup(
    name='promptengine',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch==1.9.0',
        'transformers==4.9.2',
        'pandas==1.3.1',
        'tqdm==4.61.2',
        'psutil==5.8.0',
        'numpy<2.0.0',
        'dash==2.9.3',
        'dash-daq==0.5.0',
        'dash-uploader==0.5.0',
        'plotly==5.3.1',
        'requests==2.26.0',
        'werkzeug==2.0.3',
        'scikit-learn==0.24.2',
        'optuna==2.9.1',
    ],
    entry_points={
        'console_scripts': [
            'promptengine=promptengine.cli:main',
        ],
    },
)
