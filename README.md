# To LLM or to not LLM?

As a Part of the coursework of CS421 at UIC, I am comparing the results of Pretrained Large Language Models, Finetuned LLMs, and traditional Machine Learning Algorithms in an effort to identify use cases where the expenditure of excess computational cost is not justified by the meagre increase in the model's performance.

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

The PreTrainedZeroShot is free to run without any complications and can be run anytime although it's runtime is quite heavy.

Before running the other, make sure to create a folder names 'Models' in order to store the models without any hassle

For the remaining two, in order to avoid repeated calculations and computations, there are two variations of each file.

For the Fine-tined Bert base model, there the base file which can be run to train the model which will automaticallt be stored in 