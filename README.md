# To LLM or to not LLM?

## Intro 
As a Part of the coursework of CS421 at UIC, I am comparing the results of Pretrained Large Language Models, Finetuned LLMs, and traditional Machine Learning Algorithms in an effort to identify use cases where the expenditure of excess computational cost is not justified by the meagre increase in the model's performance.

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)
[![Made with Pytoch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Share to Community](https://huggingface.co/datasets/huggingface/badges/resolve/main/powered-by-huggingface-dark.svg)](https://huggingface.co)
## Get started

I would reccomend using conda.

Activate a python environment where pytorch is installed or create a new environment from scratch.

You can install PyTorch from the requirements.txt however, I would advise against it as the installation of PyTorch requires various personal modification based on your System's specs, GPU, etc.

Install it from here: https://pytorch.org/get-started/locally/

Once Pytorch is installed, or if you decide to install PyTorch from the requirements file, run the following:

If you have already installed Pytorch:

```
pip install -r requirements.txt
```

If you haven't installed pytorch and your systems has a GPU:
```
pip install -r requirements-torch.txt
```

If you haven't installed pytorch and your systems doesn't have a GPU:
```
pip install -r requirements-torch-cpu.txt
```

once the requirements are set, you are free to run any of the notebooks.

## Running the Files

#### Pre-Trained Zeroshot model
The PreTrainedZeroShot is free to run without any complications and can be run anytime although it's runtime is quite heavy.

***

Before running the other, make sure to create a folder names 'Models' in order to store the models without any hassle and to run the entire notebook without interruptions

For the remaining two, in order to avoid repeated calculations and computations, there are two verions of each file.
***

#### FineTunedBertModel

For the Fine-tined Bert base model, there the [FineTuningBertBase.ipynb](https://github.com/thecoderenroute/to-llm-or-to-not-llm/blob/main/FineTuningBertBase.ipynb) which can be run to train the model which will automatically be stored in the created Models folder.

You need to run this notebook atleast once, and it will take approx 2-3* hours to run inclusing tokenization, training and testing.

Once this notebook runs succesfully, you will notice a .pth file in the Models folder, then you can use the [ImportAndEvaluateFBert.ipynb](https://github.com/thecoderenroute/to-llm-or-to-not-llm/blob/main/ImportAndEvaluateFBert.ipynb) which will directly IMport

#### Logistic Regression

While the Logistic regreesion model training in this case does not take a lot of time, the Doc2Vec model may end up taking 10-15 minutes to train. In order to avoid the repeated computations, you need to run the [LogisticRegressionTrainVec.ipynb](https://github.com/thecoderenroute/to-llm-or-to-not-llm/blob/main/LogisticRegressionTrainVec.ipynb) atleast once, after which a file titles d2v will have created itslef in the Models folder.

Henceforth, you can directly run the second file - [LogisticRegressionLoadVec.ipynb](https://github.com/thecoderenroute/to-llm-or-to-not-llm/blob/main/LogisticRegressionLoadVec.ipynb) which will directly import the loaded vectors.

## Possible Issue

You may run into issues regarding the file Paths, or library imports. Both of which can be solved with ease.