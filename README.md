# ML power predictor  

<div align="center">

![Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/SverreNystad/power-predictor/python-package.yml)
[![codecov.io](https://codecov.io/github/SverreNystad/power-predictor/coverage.svg?branch=main)](https://codecov.io/github/SverreNystad/power-predictor?branch=main)
![ML power predictor top language](https://img.shields.io/github/languages/top/SverreNystad/power-predictor)
![GitHub language count](https://img.shields.io/github/languages/count/SverreNystad/power-predictor)
[![Project Version](https://img.shields.io/badge/version-1.0.0-blue)](https://img.shields.io/badge/version-1.0.0-blue)

</div>

<details>
  <summary> <b> Table of Contents </b> </summary>
  <ol>
    <li>
    <a href="#ML power predictor"> Power Predictor </a>
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

![Alt text](docs/model_image.png)

This is an machine learning program made for the subject TDT4173 Machine learning. The task was to find the how much solar power measured in Photovoltaic (PV) systems, which convert sunlight into electricity. This dataset provided data for evaluating solar production dayahead forecasting methods. The data provider is [ANEO](https://www.aneo.com/). With information about all weather features [here](https://www.meteomatics.com/en/api/available-parameters/alphabetic-list/). These Data was all collected in Trondheim. The data was collected from 2019-01-01 to 2023-07-31. The data was collected every 15 minutes. The data was collected from 3 different locations. These locations were not equal. The power output of location A was 6 times larger then B and C. Location A also had solar panals that were differently angled then B and C. Making it much more trick to make a model learn.
There was also much noise in the data, with outages and times during the night with zero sunlight that there was reported solar production, and also times at day were there was no power production due to external factors.

## Usage
After cloning the project, Look at the final submission folder to see the feature engineering and the model training. The final model is saved in the models folder. To use the model, run the following code in the root directory of the project.


The task was setup so we fight Machine learning algorithms that the professor Ruslan Khalitov have made.

[![Goslightning Talking to Students at start of semester](docs/bots/goslightning.png)](docs/supergosling_lowres.mp4)
NB: Press the image to see the video of Goslightning Talking to Students at start of semester.

## Our journey
This has been a great task and we have learned a lot. We have learned how to use machine learning to solve a real world problem.
We have tried so many things, worked so many late nights and had a lot of fun, and many frustrations.
**In the end we managed to beat all the bots, and we are very proud of our work. This giving us the best grade possible: A**
The two bots that where the hardest to beat was Ryleena and Shao-RyKhan. This seems a bit strange as Goslightning is the best bot in the entire tournament, but the reason for this was that most of the project we had tried to to predictions on Kaggle (where we were graded) with bugs in the way we got the test data. This made us think that the bots were better than they actually were. We worked so hard on data that was flawed. It is very impressive that we climbed so high with several flaws in our test data. After fixing that we defeated Ryleena. 

We learned that simpler models better models, as we had models so complex that they required to be ran for more then 24 hours before completion. We also learned that the data is the most important part of the project. We spent so much time on feature engineering and data cleaning. We also learned that it is important to have a good workflow, and that it is important to have a good structure of the project. We learned that cloud computing is very powerful and quite easy to setup.
<details>
  <summary><b> We have beaten the following bots: </b></summary>

  ![gosborg](docs/bots/gosborg.png)
  
The Gosborg 2049 VT was random guessing between 0 and max pv measurement.
  ![kenshi](docs/bots/kenshi.png)
  
The Kenshi VT was using Linear Regression, with no feature engineering or other preprocessing.
  ![quan-gos-chill](docs/bots/quan-gos-chill.png)
  
Quan Gos Chill was Average for each location at the specified hour.
  ![gosipon](docs/bots/gospion.png)
  
Gospion was using Random Forest with minial feature engineering.
  ![ryleena](docs/bots/ryleena.png)
  
Frostling was using an AutoML solution using H2O, the VT had some feature engineering and random split.
  ![frostling](docs/bots/frostling.png)
  
Frostling used CatBoost with good feature engineering and good hyperparams.
  ![La La Lizard](docs/bots/la-la-lizard.png)
  
La La Lizard was the avereage of two teaching assistans models 
  ![KEN-O](docs/bots/ken-o.png)
  
Keno used a single LightGBM with with change target and extensive hyperparameters search. It used one model for all 3 locations.
  ![Shao RyKhan](docs/bots/shao-rykhan.png)
  
Shao TyKhan was made by using the best teaching assistants models, then averageing 10 different CatBoost models, having great hyper parameters and good feature engineering. But different to the other Virtual Teams was that it used one model for each location. 
  ![Alt text](docs/bots/goslightning.png)
  
  Goslightning was the best model that the professor made. This model had extended time to be finished. It used Geometric mean of 10 models from the best teaching assistanst, 1 model averaged from other teaching assistanst solutions, 2 LightGBM models with finetuning from the professor. This was the hardest bot in the compotition. 
</details>


## Installation
To install the Power Predictor, one needs to have all the prerequisites installed and set up, and follow the setup guild. The following sections will guide you through the process.
### Prerequisites
- Python 3.9 or higher
- Jupyter Notebook
  
## Setup
To setup the project, one needs to have all the prerequisites installed. Then one needs to clone the repository, setup a virtual environment, and install the dependencies. This is described in more detail below.

### Prerequisites
- Ensure Python 3.9 or newer is installed on your machine. [Download Python](https://www.python.org/downloads/)
- Familiarity with basic Python package management and virtual environments is beneficial.

### Clone the repository
```bash
git clone https://github.com/SverreNystad/power-predictor.git
cd power-predictor
```

### Virtual Environment (Recommended)

<details> 
<summary><strong>🚀 A better way to set up repositories </strong></summary>

A virtual environment in Python is a self-contained directory that contains a Python installation for a particular version of Python, plus a number of additional packages. Using a virtual environment for your project ensures that the project's dependencies are isolated from the system-wide Python and other Python projects. This is especially useful when working on multiple projects with differing dependencies, as it prevents potential conflicts between packages and allows for easy management of requirements.

1.  **To set up and use a virtual environment for Power Predictor:**
    First, install the virtualenv package using pip. This tool helps create isolated Python environments.
    ```bash
    pip install virtualenv
    ```

2. **Create virtual environment**
    Next, create a new virtual environment in the project directory. This environment is a directory containing a complete Python environment (interpreter and other necessary files).
    ```bash
    python -m venv venv
    ```

4. **Activate virtual environment**
    To activate the environment, run the following command:
    * For windows:
        ```cmd
        source ./venv/Scripts/activate
        ```

    * For Linux / MacOS:
        ```bash
        source venv/bin/activate
        ```
</details>

### Install dependencies
With the virtual environment activated, install the project dependencies:
```bash
pip install -r requirements.txt
```
The requirements.txt file contains a list of packages necessary to run Power Predictor. Installing them in an activated virtual environment ensures they are available to the project without affecting other Python projects or system settings.


## Tests
To run all the tests, run the following command in the root directory of the project:
```bash
pytest
```

## License
Licensed under the [MIT License](LICENSE). Because sharing is caring

## Folder Structure

### **Data:** All data used for the project.
* **data/raw:** Original, immutable data dump.
* **data/processed:** Cleaned and pre-processed data used for modeling.
* **data/interim:** Intermediate data that has been transformed.

### **Results:** Figures and solutions
* **results/figures:** Generated analysis as HTML, PNG, PDF, LaTeX, etc.
* **results/output:** Contains different solutions generated by models.

### **src:** Source code for use in this project.
* **src/data:** Scripts to download or generate data. From Data/raw or Data/processed to object that can be worked with.
* **src/features:** Scripts to turn raw data into features for modeling.
* **src/models:** Scripts to train models and then use trained models to make predictions.
* **src/visualization:** Scripts to create exploratory and results oriented visualizations.

### **tests:** Unit tests for the project source code.

### **final_submission:** Contains the two attempts Short_notebook_1 and Short_notebook_2 that has our two allowed attempts at the private leaderboard.

## Contributors
Three brave students that applied their knowledge of Machine Learning to beat the bots.


<table>
    <td align="center">
        <a href="https://github.com/Gunnar2908">
            <img src="https://github.com/Gunnar2908.png?size=100" width="100px;"/><br />
            <sub><b>Gunnar Nystad</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/pskoland">
            <img src="https://github.com/pskoland.png?size=100" width="100px;"/><br />
            <sub><b>Peter Skoland</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/SverreNystad">
            <img src="https://github.com/SverreNystad.png?size=100" width="100px;"/><br />
            <sub><b>Sverre Nystad</b></sub>
        </a>
    </td>
  
  </tr>
</table>

# Thanks to

* [Ruslan Khalitov](https://github.com/RuslanKhalitov) for the task and the bots. This task has been amazing and we have learned a lot.
* Thanks to the group members for the great work and the good collaboration.
* Thanks to our amazing Professor [Zhirong Yang](https://www.ntnu.no/ansatte/yangzh)https://www.ntnu.no/ansatte/yangzh for great lectures.
