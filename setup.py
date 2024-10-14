from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = ' -e .'


def get_requirements(file_path:str)->List[str]:
    '''
    this function will return list of requirements
    '''
    requirement=[]
    with open(file_path) as file:
        requirement = file.readlines()
        requirement = [req.replace('\n',"") for req in requirement]
        #print(requirement)
        if HYPEN_E_DOT in requirement:
            requirement.remove(HYPEN_E_DOT)
        '''with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('-e')]'''

        
    return requirement

 



setup(
    name='mlproject',
    version='0.0.1',
    author='soni',
    author_email='sonigujjula2004@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
