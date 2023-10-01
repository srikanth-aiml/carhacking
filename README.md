# carhacking
Repository for carhacking experiments. Overall this project loosely follows cookie cutter data science template, without actually relying on it

## Pre-requisite: 
* Refer to this document for setting up python environment and anaconda. The main thing is that python is not a binary managed by OS, but instead managed by anaconda. 
  - https://docs.google.com/document/d/1QSC0p-lB5XOfVdKdlnZdMpYvp2GI4NG5GJNOSIRsQrQ/
* If you have not cloned this repo, please do it first from https://github.com/srikanth-aiml/carhacking
* I have cloned it to C:/
* When using Anaconda powershell prompts, it is case sensitive. C:/ and c:/ are different. And thinks they are different - even though it silently resolves both of them to C:/
* The above catch becomes very important when doing conda environments and updating Jupyter kernel in step 4 below 

* Create a conda environment 
  * `conda create --name carhacking python=3.9`
* `conda activate carhacking`
* `conda install pip`
* `conda install -p C:/Users/srikanth/anaconda3/envs/carhacking ipykernel --update-deps --force-reinstall`
* `cd C:/carhacking`
* `pip install -r requirements.txt`
* `conda install scikit-learn-intelex`

## VS Code specific changes
* On Windows, use Powershell
* `touch .env`
* Add this line to .env file `PYTHONPATH=C:/carhacking;C:/carhacking/src`
* .env is not saved to git. gitignore has this setting

## .vscode
* .vscode folder is committed to git. Although this is against general git guidelines, it is done because as of now these files do not contain anything specific to individual developer folder structure
* The folder contains settings.json and launch.json. launch.json is for debugging py file execution in VS Code

## data folder
* Put the data files under data. data will not be saved to git. .gitignore has this setting

## How to execute
![Execution](https://raw.githubusercontent.com/srikanth-aiml/carhacking/main/docs/images/vscode_setup.png)