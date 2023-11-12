To execute python source codes, we need to install Python 3 environment. The current version has been tested on 
Python 3.10.7. 

It is recommended to install Python virtual environment for the tool

```shell
$ python3.10 -m venv env
$ source env/bin/activate
```

With the following packages: 

* numpy==1.23.5
* pandas==1.3.5
* scikit-learn==1.3.0
* scipy==1.9.3
* tensorflow==2.14.0
* keras==2.14.0


```shell
$ pip install -r requirements.txt
```

After installing the required packages, we can execute "inferDataPrecondition.py" to obtain data precondition 
with a trained DNN model and dataset from specific directory. 
In order to get the unseen data prediction with DeepInfer, we need to execute (unseenPrediction.py) python file.
