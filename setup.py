from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Lattice descent algorithms in JAX'
LONG_DESCRIPTION = 'Lattice descent algorithms in JAX'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="lattDesc",
        version=VERSION,
        author="Diego Marcondes",
        author_email="<dmarcondes@ime.usp.br>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['jax','alive_progress','IPython','plotly','Pillow'], # add any additional packages that
        # needs to be installed along with your package. Eg: 'caer'

        keywords=['python', 'JAX', 'lattice descent'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Researchers",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux :: Ubuntu",
        ]
)
