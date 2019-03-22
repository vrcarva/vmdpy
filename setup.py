import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'vmdpy',         
  packages=setuptools.find_packages(), 
  version = '0.1',        
  description = 'Variational Mode Decomposition (VMD) algorithm',   
  url='http://github.com/vrcarva/vmdpy',
  author='Vinicius Rezende Carvalho',
  author_email='vrcarva@ufmg.br',
  keywords = ['VMD', 'variational', 'decomposition'],   
  long_description=long_description,
  long_description_content_type="text/markdown",  
  classifiers=[
    'Development Status :: 3 - Alpha',     
    'Intended Audience :: Science/Research',     
    'Topic :: Scientific/Engineering :: Mathematics',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',    
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)