# Author: Ty Andrews
# Date: 2023-03-27

FROM python:3.10-slim-buster

USER root
WORKDIR /fortunato-wheels-engine

# ensure packages are up to date on the system
RUN apt-get update

# install the required ptyhon packages from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# pull in src and data/scraping-tracking folders
COPY src/ /fortunato-wheels-engine/src/
COPY data/scraping-tracking/ /fortunato-wheels-engine/data/scraping-tracking/

ENTRYPOINT ["python", "/fortunato-wheels-engine/src/scraping/scraping.py"]