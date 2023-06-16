from setuptools import find_packages, setup


long_description = ''

setup(
    name='micelle_scattering', 
    version='1.0', 
    description='Dilute Micelle Scattering Data Generation', 
    package_dir={'': 'app'}, 
    packages=find_packages(where='app'), 
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    author='Yun Jan', 
    author_email='yunjan0001@gmail.com', 
    url='https://www.alexmarras.com/', 
    license='MIT', 
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent', 
        'Programming Language :: Python :: 3.10'], 
    install_requires=['numpy>=1.20'], 
    extras_require={}, 
    python_requires='>=3.10'
)
