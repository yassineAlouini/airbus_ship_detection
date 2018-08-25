from setuptools import find_packages, setup

NAME = 'airbus_ship_detection'
VERSION = '0.0.1'
AUTHOR = 'Yassine Alouini'
DESCRIPTION = """The repo for the Airbus ship detection challenge."""
EMAIL = "yassinealouini@outlook.com"
URL = ""

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    # Some metadata
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    url=URL,
    license="MIT",
    keywords="kaggle machine-learning computer vision",
)
