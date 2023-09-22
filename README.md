# ML power predictor  

<div align="center">

![Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/SverreNystad/power-predictor/python-package.yml)
[![codecov.io](https://codecov.io/github/SverreNystad/power-predictor/coverage.svg?branch=main)](https://codecov.io/github/SverreNystad/power-predictor?branch=main)
![ML power predictor top language](https://img.shields.io/github/languages/top/SverreNystad/power-predictor)
![GitHub language count](https://img.shields.io/github/languages/count/SverreNystad/power-predictor)
[![Project Version](https://img.shields.io/badge/version-0.0.1-blue)](https://img.shields.io/badge/version-0.0.1-blue)

</div>

<details>
  <summary> <b> Table of Contents </b> </summary>
  <ol>
    <li>
    <a href="#standard_python_application"> PROJECT NAME </a>
    </li>
    <li>
      <a href="#Introduction">Introduction</a>
    </li>
    </li>
    <li><a href="#Usage">Usage</a></li>
    <li><a href="#Installation">Installation</a>
      <ul>
        <li><a href="#Prerequisites">Prerequisites</a></li>
        <li><a href="#Setup">Setup</a></li>
      </ul>
    </li>
    <li><a href="#Tests">Tests</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## Introduction



## Usage
After installing the project, run the jupyter notebook in the exploration directory of the project.

## Installation
To install the PROJECT NAME, one needs to have all the prerequisites installed and set up, and follow the setup guild. The following sections will guide you through the process.
### Prerequisites
- Python 3.9 or higher
- Jupyter Notebook
  

### Setup
1. Clone the repository
```bash
git clone https://github.com/USERNAME/REPONAME.git
cd PROJECT NAME
```
2. Create a virtual environment
    #### On Windows:
    ```bash
    python3 -m venv venv
    source venv/Scripts/activate
    ```
    #### On macOS and Linux: 
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages
```bash
pip install -r requirements.txt
```

4. Create a file called `.env` in the root directory of the project. Add the following lines to the file:
```bash
touch .env
echo "SECRET=SECRET_VALUE" > .env # Remember to change SECRET_VALUE to your actual key
```

## Tests
To run all the tests, run the following command in the root directory of the project:
```bash
pytest --cov
coverage html # To generate a coverage report
```
If you do not want to run api tests, run the following command instead:
```bash
pytest -m "not apitest" --cov
```

## License
Licensed under the [MIT License](LICENSE). Because this is a template repository, you need to change the license if you want to use it for your own project.

## Folder Structure

### **Data:** All data used for the project.
* **data/raw:** Original, immutable data dump.
* **data/processed:** Cleaned and pre-processed data used for modeling.
* **data/external:** External data sources or additional datasets.
* **data/interim:** Intermediate data that has been transformed.

### **Exploration:** Jupyter notebooks for exploratory data analysis.


### **Results:** Generated analysis as HTML, PDF, LaTeX, etc.

### **src:** Source code for use in this project.
* **src/data:** Scripts to download or generate data. From Data/raw or Data/processed to object that can be worked with.
* **src/features:** Scripts to turn raw data into features for modeling.
* **src/models:** Scripts to train models and then use trained models to make predictions.
* **src/visualization:** Scripts to create exploratory and results oriented visualizations.

### **tests:** Unit tests for the project source code.