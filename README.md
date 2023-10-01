# carhacking
Repository for carhacking experiments

## Pre-requisite: 
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
