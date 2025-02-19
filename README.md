[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![codecov][codecov-shield]][codecov-url]

![](assets/fortunato-wheels-logo-blue-white.png)
# Fortunato Wheels Engine

This is the data engine for the Fortunato Wheels project. It manages the data and machine learning used to power the [Fortunato wheels website](https://fortunatowheels.com/).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Setup

To get started first setup the environments:

Create environment with Conda from environment.yml file:
```
conda env create -f environment.yml
```
Or with pip requirements.txt:
```
pip install -r requirements.txt
```

### Downloading & Preprocessing Data

There are two primary data sources, one is the Fortunato Wheels database hosted in a 
MongoDB instance on Azure. The other is the open source dataset of carguru.com ads originally used from Kaggle [here](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset).

1. Fortunato Wheels Database
    1. Get the connection string from the project owner.
    2. Put the connection string into the .env file in the root of the project.
    3. To get all car ads from the database use:
        ```
        from src.data.upload_to_db import connect_to_db
        client, db, collection = connect_to_db()
        all_ads_raw = pd.DataFrame(collection.find())
        ```
2. Carguru.com Ads: Download Processed Data
   1. To download an already processed version of the dataset download the parquet file from:  
    [Link to Processed Cargurus Data Download (~3GB)](https://drive.google.com/file/d/1rGJALL3xeGaakTp3BfH5Fd8uI3EP_4x8/view?usp=share_link)
    2. Move the downloaded file to saved to `data/processed/`

### Start Working with the Data

To start working with the data, make a copy of the `notebooks/00-getting-started.ipynb` notebook and start working from there. It imports all the necessary libraries and sets up the data into a single dataframe in a `CarAds` object and shows how to get data loaded into a dataframe.

Here is what the sample data looks like for Subaru Outback from 2008-2012:

![](assets/getting-started-nb-plot.png)

## Running the tests

Pytest is used to manage the tests. To run the tests in the `tests` folder use the following command:

```
pytest tests
```

## Deployment

TODO: Add additional notes about how to deploy this on a live system


## Contributing

Please read [CONTRIBUTING.md](https://github.com/tieandrews/fortunato-wheels-engine/blob/main/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the tags on this repository.

## Authors

* **Ty Andrews** - *Initial work* - [LinkedIn](https://www.linkedin.com/in/ty-andrews/)


## License

This project is licensed under the MIT License - see the [LICENSE.md]([LICENSE.md](https://github.com/tieandrews/fortunato-wheels-engine/blob/main/LICENSE.md)) file for details

[contributors-shield]: https://img.shields.io/github/contributors/tieandrews/fortunato-wheels-engine.svg?style=for-the-badge
[contributors-url]: https://github.com/tieandrews/fortunato-wheels-engine/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/tieandrews/fortunato-wheels-engine.svg?style=for-the-badge
[forks-url]: https://github.com/tieandrews/fortunato-wheels-engine/network/members
[stars-shield]: https://img.shields.io/github/stars/tieandrews/fortunato-wheels-engine.svg?style=for-the-badge
[stars-url]: https://github.com/tieandrews/fortunato-wheels-engine/stargazers
[issues-shield]: https://img.shields.io/github/issues/tieandrews/fortunato-wheels-engine.svg?style=for-the-badge
[issues-url]: https://github.com/tieandrews/fortunato-wheels-engine/issues
[license-shield]: https://img.shields.io/github/license/tieandrews/fortunato-wheels-engine.svg?style=for-the-badge
[license-url]: https://github.com/tieandrews/fortunato-wheels-engine/blob/master/LICENSE.txt
[codecov-shield]: https://img.shields.io/codecov/c/github/tieandrews/fortunato-wheels-engine?style=for-the-badge
[codecov-url]: https://codecov.io/gh/tieandrews/fortunato-wheels-engine
