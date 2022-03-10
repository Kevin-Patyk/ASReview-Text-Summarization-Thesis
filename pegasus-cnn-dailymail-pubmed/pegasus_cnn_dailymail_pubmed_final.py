{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "X4cRE8IbIrIV"
   },
   "source": [
    "If you're opening this Notebook on colab, you will probably need to install 🤗 `Transformers` and 🤗 `Datasets` as well as other dependencies. \n",
    "\n",
    "* `datasets`\n",
    "* `transformers`\n",
    "* `rogue-score`\n",
    "* `nltk`\n",
    "* `pytorch`\n",
    "* `ipywidgets`\n",
    "\n",
    "*Note*: Since we are using the GPU to optimize the performance of the deep learning algorithms, `CUDA` needs to be installed on the device."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "MOsHUjgdIrIW"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: datasets in ./miniconda3/envs/fastai/lib/python3.8/site-packages (1.18.4)\n",
      "Requirement already satisfied: transformers in ./miniconda3/envs/fastai/lib/python3.8/site-packages (4.17.0)\n",
      "Requirement already satisfied: rouge-score in ./miniconda3/envs/fastai/lib/python3.8/site-packages (0.0.4)\n",
      "Requirement already satisfied: nltk in ./miniconda3/envs/fastai/lib/python3.8/site-packages (3.7)\n",
      "Requirement already satisfied: ipywidgets in ./miniconda3/envs/fastai/lib/python3.8/site-packages (7.6.4)\n",
      "Requirement already satisfied: multiprocess in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (0.70.12.2)\n",
      "Requirement already satisfied: requests>=2.19.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (2.26.0)\n",
      "Requirement already satisfied: pandas in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (1.3.2)\n",
      "Requirement already satisfied: dill in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (0.3.4)\n",
      "Requirement already satisfied: pyarrow!=4.0.0,>=3.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (7.0.0)\n",
      "Requirement already satisfied: tqdm>=4.62.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (4.62.2)\n",
      "Requirement already satisfied: fsspec[http]>=2021.05.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (2022.2.0)\n",
      "Requirement already satisfied: aiohttp in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (3.7.4.post0)\n",
      "Requirement already satisfied: packaging in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (21.0)\n",
      "Requirement already satisfied: huggingface-hub<1.0.0,>=0.1.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (0.4.0)\n",
      "Requirement already satisfied: xxhash in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (3.0.0)\n",
      "Requirement already satisfied: numpy>=1.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (1.20.3)\n",
      "Requirement already satisfied: responses<0.19 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (0.18.0)\n",
      "Requirement already satisfied: regex!=2019.12.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (2022.3.2)\n",
      "Requirement already satisfied: sacremoses in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (0.0.47)\n",
      "Requirement already satisfied: pyyaml>=5.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (5.4.1)\n",
      "Requirement already satisfied: tokenizers!=0.11.3,>=0.11.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (0.11.6)\n",
      "Requirement already satisfied: filelock in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (3.6.0)\n",
      "Requirement already satisfied: six>=1.14.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from rouge-score) (1.16.0)\n",
      "Requirement already satisfied: absl-py in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from rouge-score) (1.0.0)\n",
      "Requirement already satisfied: click in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nltk) (8.0.1)\n",
      "Requirement already satisfied: joblib in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nltk) (1.0.1)\n",
      "Requirement already satisfied: nbformat>=4.2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (5.1.3)\n",
      "Requirement already satisfied: traitlets>=4.3.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (5.1.0)\n",
      "Requirement already satisfied: ipython-genutils~=0.2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (0.2.0)\n",
      "Requirement already satisfied: ipykernel>=4.5.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (6.2.0)\n",
      "Requirement already satisfied: widgetsnbextension~=3.5.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (3.5.1)\n",
      "Requirement already satisfied: ipython>=4.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (7.27.0)\n",
      "Requirement already satisfied: jupyterlab-widgets>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (1.0.0)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (3.10.0.2)\n",
      "Requirement already satisfied: matplotlib-inline<0.2.0,>=0.1.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (0.1.2)\n",
      "Requirement already satisfied: jupyter-client<8.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.1.12)\n",
      "Requirement already satisfied: tornado<7.0,>=4.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.1)\n",
      "Requirement already satisfied: debugpy<2.0,>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (1.4.1)\n",
      "Requirement already satisfied: pygments in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (2.10.0)\n",
      "Requirement already satisfied: decorator in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (5.0.9)\n",
      "Requirement already satisfied: pexpect>4.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (4.8.0)\n",
      "Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (3.0.17)\n",
      "Requirement already satisfied: pickleshare in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.7.5)\n",
      "Requirement already satisfied: jedi>=0.16 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.18.0)\n",
      "Requirement already satisfied: backcall in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.2.0)\n",
      "Requirement already satisfied: setuptools>=18.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (58.0.4)\n",
      "Requirement already satisfied: parso<0.9.0,>=0.8.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jedi>=0.16->ipython>=4.0.0->ipywidgets) (0.8.2)\n",
      "Requirement already satisfied: jupyter-core>=4.6.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jupyter-client<8.0->ipykernel>=4.5.1->ipywidgets) (4.7.1)\n",
      "Requirement already satisfied: pyzmq>=13 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jupyter-client<8.0->ipykernel>=4.5.1->ipywidgets) (22.2.1)\n",
      "Requirement already satisfied: python-dateutil>=2.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jupyter-client<8.0->ipykernel>=4.5.1->ipywidgets) (2.8.2)\n",
      "Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbformat>=4.2.0->ipywidgets) (3.2.0)\n",
      "Requirement already satisfied: attrs>=17.4.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (21.2.0)\n",
      "Requirement already satisfied: pyrsistent>=0.14.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (0.17.3)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from packaging->datasets) (2.4.7)\n",
      "Requirement already satisfied: ptyprocess>=0.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from pexpect>4.3->ipython>=4.0.0->ipywidgets) (0.7.0)\n",
      "Requirement already satisfied: wcwidth in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=4.0.0->ipywidgets) (0.2.5)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (1.26.6)\n",
      "Requirement already satisfied: idna<4,>=2.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (3.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (2021.5.30)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (2.0.4)\n",
      "Requirement already satisfied: notebook>=4.4.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from widgetsnbextension~=3.5.0->ipywidgets) (6.4.3)\n",
      "Requirement already satisfied: Send2Trash>=1.5.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.8.0)\n",
      "Requirement already satisfied: prometheus-client in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.11.0)\n",
      "Requirement already satisfied: nbconvert in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (5.6.1)\n",
      "Requirement already satisfied: argon2-cffi in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (20.1.0)\n",
      "Requirement already satisfied: terminado>=0.8.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.9.4)\n",
      "Requirement already satisfied: jinja2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (3.0.1)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (5.1.0)\n",
      "Requirement already satisfied: async-timeout<4.0,>=3.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (3.0.1)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (1.5.1)\n",
      "Requirement already satisfied: chardet<5.0,>=2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (4.0.0)\n",
      "Requirement already satisfied: cffi>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.14.0)\n",
      "Requirement already satisfied: pycparser in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from cffi>=1.0.0->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.20)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jinja2->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.0.1)\n",
      "Requirement already satisfied: mistune<2,>=0.8.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.8.4)\n",
      "Requirement already satisfied: entrypoints>=0.2.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.3)\n",
      "Requirement already satisfied: pandocfilters>=1.4.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.4.3)\n",
      "Requirement already satisfied: bleach in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (4.0.0)\n",
      "Requirement already satisfied: testpath in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.5.0)\n",
      "Requirement already satisfied: defusedxml in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.7.1)\n",
      "Requirement already satisfied: webencodings in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.5.1)\n",
      "Requirement already satisfied: pytz>=2017.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from pandas->datasets) (2021.1)\n"
     ]
    }
   ],
   "source": [
    "! pip install datasets transformers rouge-score nltk ipywidgets"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "PfR1c-t7j9d3"
   },
   "source": [
    "When using `nltk`, `punkt` also needs to be installed. I guess it is not installed automatically. Not having `punkt` will result in an error during the analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "T_4jdf_Gj-Cu"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to /home/user/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('punkt')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "KyNmCPBvir6z"
   },
   "source": [
    "If you're opening this notebook locally, make sure your environment has an install from the last version of those libraries.\n",
    "\n",
    "To be able to share your model with the community and generate results like the one shown in the picture below via the inference API, there are a few more steps to follow.\n",
    "\n",
    "First you have to store your authentication token from the Hugging Face website (sign up [here](https://huggingface.co/join) if you haven't already!) then execute the following cell and input your username and password:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "id": "9weGF83Sir6z"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d27544ee9efc42f78e2965189f0430a5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(HTML(value='<center>\\n<img src=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from huggingface_hub import notebook_login\n",
    "\n",
    "notebook_login()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2KWkoABDir6z"
   },
   "source": [
    "Then you need to install `Git-LFS`.\n",
    "\n",
    "If you are not using `Google Colab`, you may need to install `Git-LFS` manually, since the code below may not work and depending on your operating system. You can read about `Git-LFS` and how to install it [here](https://git-lfs.github.com/)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "id": "FNO0eWs7ir6z"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Reading package lists... Done\n",
      "Building dependency tree       \n",
      "Reading state information... Done\n",
      "git-lfs is already the newest version (2.9.2-1).\n",
      "0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.\n"
     ]
    }
   ],
   "source": [
    "! sudo apt install git-lfs"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "tiYMAMuair60"
   },
   "source": [
    "Make sure your version of `Transformers` is at least 4.11.0 since the functionality was introduced in that version:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "3y-w6Uyhir60"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4.17.0\n"
     ]
    }
   ],
   "source": [
    "import transformers\n",
    "\n",
    "print(transformers.__version__)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HFASsisvIrIb"
   },
   "source": [
    "You can find a script version of this notebook to fine-tune your model in a distributed fashion using multiple GPUs or TPUs [here](https://github.com/huggingface/transformers/tree/master/examples/seq2seq)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "rEJBSTyZIrIb"
   },
   "source": [
    "# Fine-tuning a model on a summarization task"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kTCFado4IrIc"
   },
   "source": [
    "In this notebook, we will see how to fine-tune one of the [🤗`Transformers`](https://github.com/huggingface/transformers) model for a summarization task. We will use the [PubMed Summarization dataset](https://huggingface.co/datasets/ccdv/pubmed-summarization) which contains PubMed articles accompanied with abstracts.\n",
    "\n",
    "![Widget inference on a summarization task](https://github.com/huggingface/notebooks/blob/master/examples/images/summarization.png?raw=1)\n",
    "\n",
    "We will see how to easily load the dataset for this task using 🤗 `Datasets` and how to fine-tune a model on it using the `Trainer` API."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "voXBC93bir61"
   },
   "outputs": [],
   "source": [
    "model_checkpoint = \"google/pegasus-cnn_dailymail\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4RRkXuteIrIh"
   },
   "source": [
    "This notebook is built to run  with any model checkpoint from the [Model Hub](https://huggingface.co/models) as long as that model has a sequence-to-sequence version in the Transformers library. Here we picked the [`google/pegasus-cnn_dailymail`](https://huggingface.co/google/pegasus-cnn_dailymail) checkpoint. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "whPRbBNbIrIl"
   },
   "source": [
    "## Loading the dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "W7QYTpxXIrIl"
   },
   "source": [
    "We will use the [🤗 `Datasets`](https://github.com/huggingface/datasets) library to download the data and get the metric we need to use for evaluation (to compare our model to the benchmark). This can be easily done with the functions `load_dataset` and `load_metric`.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "IreSlFmlIrIm"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No config specified, defaulting to: pub_med_summarization_dataset/document\n",
      "Reusing dataset pub_med_summarization_dataset (/home/user/.cache/huggingface/datasets/ccdv___pub_med_summarization_dataset/document/1.0.0/5792402f4d618f2f4e81ee177769870f365599daa729652338bac579552fec30)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "655cf79213854b699ef9b639b3dedefd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/3 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from datasets import load_dataset, load_metric\n",
    "\n",
    "raw_datasets = load_dataset(\"ccdv/pubmed-summarization\")\n",
    "metric = load_metric(\"rouge\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "RzfPtOMoIrIu"
   },
   "source": [
    "The `dataset` object itself is [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which contains one key for the training, validation and test set:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "GWiVUF0jIrIv"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DatasetDict({\n",
       "    train: Dataset({\n",
       "        features: ['article', 'abstract'],\n",
       "        num_rows: 119924\n",
       "    })\n",
       "    validation: Dataset({\n",
       "        features: ['article', 'abstract'],\n",
       "        num_rows: 6633\n",
       "    })\n",
       "    test: Dataset({\n",
       "        features: ['article', 'abstract'],\n",
       "        num_rows: 6658\n",
       "    })\n",
       "})"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "raw_datasets"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "u3EtYfeHIrIz"
   },
   "source": [
    "To access an actual element, you need to select a split first, then give an index:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "X6HrpprwIrIz"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'article': \"a recent systematic analysis showed that in 2011 , 314 ( 296 - 331 ) million children younger than 5 years were mildly , moderately or severely stunted and 258 ( 240 - 274 ) million were mildly , moderately or severely underweight in the developing countries . in iran a study among 752 high school girls in sistan and baluchestan showed prevalence of 16.2% , 8.6% and 1.5% , for underweight , overweight and obesity , respectively . the prevalence of malnutrition among elementary school aged children in tehran varied from 6% to 16% . anthropometric study of elementary school students in shiraz revealed that 16% of them suffer from malnutrition and low body weight . snack should have 300 - 400 kcal energy and could provide 5 - 10 g of protein / day . nowadays , school nutrition programs are running as the national programs , world - wide . national school lunch program in the united states there are also some reports regarding school feeding programs in developing countries . in vietnam , school base program showed an improvement in nutrient intakes . in iran a national free food program ( nffp ) is implemented in elementary schools of deprived areas to cover all poor students . however , this program is not conducted in slums and poor areas of the big cities so many malnourished children with low socio - economic situation are not covered by nffp . although the rate of poverty in areas known as deprived is higher than other areas , many students in deprived areas are not actually poor and can afford food . hence , nutritional value of the nffp is lower than the scientific recommended snacks for this age group . furthermore , lack of variety of food packages has decreased the tendency of children toward nffp . on the other hand , the most important one is ministry of education ( moe ) of iran , which is responsible for selecting and providing the packages for targeted schools . the ministry of health ( moh ) is supervising the health situation of students and their health needs . welfare organizations , along with charities , have the indirect effect on nutritional status of students by financial support of their family . provincial governors have also the role of coordinating and supervising all activities of these organizations . parent - teacher association is a community - based institution that participates in school 's policy such as nffp . in addition to these organizations , nutritional literacy of students , their parents and teachers , is a very important issue , which could affect nutritional status of school age children . therefore , the present study was conducted with the aim of improving the nffp , so that by its resources all poor children will be covered even in big cities . moreover , all food packages were replaced by nutritious and diverse packages that were accessible for non - poor children . according to the aim of this study and multiple factors that could affect the problem , public health advocacy has been chosen as the best strategy to deal with this issue . therefore , the present study determines the effects of nutrition intervention in an advocacy process model on the prevalence of underweight in school aged children in the poor area of shiraz , iran . this interventional study has been carried out between 2009 and 2010 in shiraz , iran . this survey was approved by the research committee of shiraz university of medical sciences . in coordination with education organization of fars province two elementary schools and one middle school in the third region of the urban area of shiraz were selected randomly . in those schools all students ( 2897 , 7 - 13 years old ) were screened based on their body mass index ( bmi ) by nutritionists . according to convenience method all students divided to two groups based on their economic situation ; family revenue and head of household 's job and nutrition situation ; the first group were poor and malnourished students and the other group were well nourished or well - off students . for this report , the children 's height and weight were entered into center for disease control and prevention ( cdc ) to calculate bmi and bmi - for - age z - scores based on cdc for diseases control and prevention and growth standards . the significance of the difference between proportions was calculated using two - tailed z - tests for independent proportions . for implementing the interventions , the advocacy process model weight was to the nearest 0.1 kg on a balance scale ( model # seca scale ) . standing height was measured to the nearest 0.1 cm with a wall - mounted stadiometer . advocacy group formation : this step was started with stakeholder analysis and identifying the stakeholders . the team was formed with representatives of all stakeholders include ; education organization , welfare organization , deputy for health of shiraz university , food and cosmetic product supervisory office and several non - governmental organizations and charities . situation analysis : this was carried out by use of existing data such as formal report of organizations , literature review and focus group with experts . the prevalence of malnutrition and its related factors among students was determined and weaknesses and strengths of the nffp were analyzed . accordingly , three sub - groups were established : research and evaluation , education and justification and executive group . designing the strategies : three strategies were identified ; education and justification campaign , nutritional intervention ( providing nutritious , safe and diverse snacks ) and networking . performing the interventions : interventions that were implementing in selected schools were providing a diverse and nutritious snack package along with nutrition education for both groups while the first group ( poor and malnourished students ) was utilized the package free of charge . education and justification intervention : regarding the literature review and expert opinion , an educational group affiliated with the advocacy team has prepared educational booklets about nutritional information for each level ( degree ) . accordingly , education of these booklets has been integrated into regular education of students and they educated and justified for better nutrition life - style . it leads the educational group to hold several meeting with the student 's parents to justify them about the project and its benefit for their children . after these meetings , parental desire for participation in the project illustrated the effectiveness of the justification meeting with them . for educate fifteen talk show programs in tv and radio , 12 published papers in the local newspaper , have implemented to mobilize the community and gain their support . healthy diet , the importance of breakfast and snack in adolescence , wrong food habits among school age children , role of the family to improve food habit of children were the main topics , in which media campaign has focused on . nutritional intervention : the snack basket of the students was replaced with traditional , nutritious and diverse foods . in general , the new snack package in average has provided 380 kcal energy , 15 g protein along with sufficient calcium and iron . low economic and malnourished children were supported by executive group affiliated with advocacy team and the rest of them prepare their snack by themselves . research and evaluation : in this step , the literacy and anthropometric indices ( bmi ) of students were assessed before and after the interventions . the reference for anthropometric measures was the world health organization / national center for health statistics ( who / nchs ) standards and the cut - offs were - two standard deviations ( sd ) from the mean . each student that was malnourished and poor has been taken into account for free food and nutritious snacks . demographic information , height , weight and knowledge of the students were measured by use of a validated and reliable ( cronbach 's alpha was 0.61 ) questionnaire . this project is granted by shiraz university of medical sciences , charities and welfare organization and education organization of fars province . statistical analyses were performed using the statistical package for the social sciences ( spss ) software , version 17.0 ( spss inc . , the results are expressed as mean  sd and proportions as appropriated . in order to determine the effective variables on the malnutrition status paired t test was used to compare the end values with baseline ones in each group . in this project , the who z - score cut - offs used were as follow : using bmi - for - age z - scores ; overweight : > + 1 sd , i.e. , z - score > 1 ( equivalent to bmi 25 kg / m ) , obesity : > + 2 sd ( equivalent to bmi 30 kg / m ) , thinness : < 2 sd and severe thinness : < 3 sd . this interventional study has been carried out between 2009 and 2010 in shiraz , iran . this survey was approved by the research committee of shiraz university of medical sciences . in coordination with education organization of fars province two elementary schools and one middle school in the third region of the urban area of shiraz were selected randomly . in those schools all students ( 2897 , 7 - 13 years old ) were screened based on their body mass index ( bmi ) by nutritionists . according to convenience method all students divided to two groups based on their economic situation ; family revenue and head of household 's job and nutrition situation ; the first group were poor and malnourished students and the other group were well nourished or well - off students . for this report , the children 's height and weight were entered into center for disease control and prevention ( cdc ) to calculate bmi and bmi - for - age z - scores based on cdc for diseases control and prevention and growth standards . the significance of the difference between proportions was calculated using two - tailed z - tests for independent proportions . for implementing the interventions , weight was to the nearest 0.1 kg on a balance scale ( model # seca scale ) . standing height was measured to the nearest 0.1 cm with a wall - mounted stadiometer . advocacy group formation : this step was started with stakeholder analysis and identifying the stakeholders . the team was formed with representatives of all stakeholders include ; education organization , welfare organization , deputy for health of shiraz university , food and cosmetic product supervisory office and several non - governmental organizations and charities . situation analysis : this was carried out by use of existing data such as formal report of organizations , literature review and focus group with experts . the prevalence of malnutrition and its related factors among students was determined and weaknesses and strengths of the nffp were analyzed . accordingly , three sub - groups were established : research and evaluation , education and justification and executive group . designing the strategies : three strategies were identified ; education and justification campaign , nutritional intervention ( providing nutritious , safe and diverse snacks ) and networking . performing the interventions : interventions that were implementing in selected schools were providing a diverse and nutritious snack package along with nutrition education for both groups while the first group ( poor and malnourished students ) was utilized the package free of charge . duration of intervention was 6 months . education and justification intervention : regarding the literature review and expert opinion , an educational group affiliated with the advocacy team has prepared educational booklets about nutritional information for each level ( degree ) . accordingly , education of these booklets has been integrated into regular education of students and they educated and justified for better nutrition life - style . obviously , student 's families had remarkable effect on children 's food habit . it leads the educational group to hold several meeting with the student 's parents to justify them about the project and its benefit for their children . after these meetings , parental desire for participation in the project illustrated the effectiveness of the justification meeting with them . educate fifteen talk show programs in tv and radio , 12 published papers in the local newspaper , have implemented to mobilize the community and gain their support . healthy diet , the importance of breakfast and snack in adolescence , wrong food habits among school age children , role of the family to improve food habit of children were the main topics , in which media campaign has focused on . nutritional intervention : the snack basket of the students was replaced with traditional , nutritious and diverse foods . in general , the new snack package in average has provided 380 kcal energy , 15 g protein along with sufficient calcium and iron . low economic and malnourished children were supported by executive group affiliated with advocacy team and the rest of them prepare their snack by themselves . research and evaluation : in this step , the literacy and anthropometric indices ( bmi ) of students were assessed before and after the interventions . the reference for anthropometric measures was the world health organization / national center for health statistics ( who / nchs ) standards and the cut - offs were - two standard deviations ( sd ) from the mean . each student that was malnourished and poor has been taken into account for free food and nutritious snacks . demographic information , height , weight and knowledge of the students were measured by use of a validated and reliable ( cronbach 's alpha was 0.61 ) questionnaire . this project is granted by shiraz university of medical sciences , charities and welfare organization and education organization of fars province . advocacy group formation : this step was started with stakeholder analysis and identifying the stakeholders . the team was formed with representatives of all stakeholders include ; education organization , welfare organization , deputy for health of shiraz university , food and cosmetic product supervisory office and several non - governmental organizations and charities . situation analysis : this was carried out by use of existing data such as formal report of organizations , literature review and focus group with experts . the prevalence of malnutrition and its related factors among students was determined and weaknesses and strengths of the nffp were analyzed . accordingly , three sub - groups were established : research and evaluation , education and justification and executive group . designing the strategies : three strategies were identified ; education and justification campaign , nutritional intervention ( providing nutritious , safe and diverse snacks ) and networking . performing the interventions : interventions that were implementing in selected schools were providing a diverse and nutritious snack package along with nutrition education for both groups while the first group ( poor and malnourished students ) was utilized the package free of charge . education and justification intervention : regarding the literature review and expert opinion , an educational group affiliated with the advocacy team has prepared educational booklets about nutritional information for each level ( degree ) . accordingly , education of these booklets has been integrated into regular education of students and they educated and justified for better nutrition life - style . obviously , student 's families had remarkable effect on children 's food habit . it leads the educational group to hold several meeting with the student 's parents to justify them about the project and its benefit for their children . after these meetings , parental desire for participation in the project illustrated the effectiveness of the justification meeting with them . educate fifteen talk show programs in tv and radio , 12 published papers in the local newspaper , have implemented to mobilize the community and gain their support . healthy diet , the importance of breakfast and snack in adolescence , wrong food habits among school age children , role of the family to improve food habit of children were the main topics , in which media campaign has focused on . nutritional intervention : the snack basket of the students was replaced with traditional , nutritious and diverse foods . in general , the new snack package in average has provided 380 kcal energy , 15 g protein along with sufficient calcium and iron . low economic and malnourished children were supported by executive group affiliated with advocacy team and the rest of them prepare their snack by themselves . research and evaluation : in this step , the literacy and anthropometric indices ( bmi ) of students were assessed before and after the interventions . the reference for anthropometric measures was the world health organization / national center for health statistics ( who / nchs ) standards and the cut - offs were - two standard deviations ( sd ) from the mean . each student that was malnourished and poor has been taken into account for free food and nutritious snacks . demographic information , height , weight and knowledge of the students were measured by use of a validated and reliable ( cronbach 's alpha was 0.61 ) questionnaire . this project is granted by shiraz university of medical sciences , charities and welfare organization and education organization of fars province . statistical analyses were performed using the statistical package for the social sciences ( spss ) software , version 17.0 ( spss inc . , chicago , il , usa ) . the results are expressed as mean  sd and proportions as appropriated . in order to determine the effective variables on the malnutrition status paired t test was used to compare the end values with baseline ones in each group . two - sided p < 0.05 was considered to be statistically significant . in this project , the who z - score cut - offs used were as follow : using bmi - for - age z - scores ; overweight : > + 1 sd , i.e. , z - score > 1 ( equivalent to bmi 25 kg / m ) , obesity : > + 2 sd ( equivalent to bmi 30 kg / m ) , thinness : < 2 sd and severe thinness : < 3 sd . study population contains 2897 children ; 70.8% were primary school students and 29.2% were secondary school students . 2336 ( 80.5% ) out of total students were well - off and 561 children ( 19.5% ) were indigent . 19.5% of subjects were in case group ( n = 561 ) and 80.5% were in the control group ( n = 2336 ) . the mean of age in welfare group was 10.0  2.3 and 10.5  2.5 in non - welfare group . demographic characteristics of school aged children in shiraz , iran table 2 shows the frequency of subjects in different categories of bmi for age in non - welfare and welfare groups of school aged children separately among boys and girls before and after a nutrition intervention based on advocacy process model in shiraz , iran . the frequency of subjects with bmi lower than < 2 sd decreased significantly after intervention among non - welfare girls ( p < 0.01 ) . however , there were no significant decreases in the frequency of subjects with bmi lower than < 2 sd boys . when we assess the effect of intervention in total population without separating by sex groups , we found no significant change in this population [ table 3 ] . bmi for age for iranian students aged 7 - 14 years based on gender according to who growth standards 2007 bmi for age for iranian students aged 7 - 14 years according to who growth standards 2007 in non - welfare and welfare groups of total population table 4 has shown the prevalence of normal bmi , mild , moderate and severe malnutrition in non - welfare and welfare groups of school aged children separately among boys and girls before and after a nutrition intervention based on advocacy process model . according to this table there were no significant differences in the prevalence of mild , moderate and severe malnutrition among girls and boys . table 4 also shows the mean of all anthropometric indices changed significantly after intervention both among girls and boys . the pre- and post - test education assessment in both groups showed that the student 's average knowledge score has been significantly increased from 12.5  3.2 to 16.8  4.3 ( p < 0.0001 ) . bmi , height and weight in non - welfare and welfare groups of school aged children separately in males and females before and after a nutrition intervention based on advocacy process model in shiraz , iran according to study 's finding the odds ratio ( or ) of sever thinness and thinness in non - welfare compared with welfare is 3.5 ( or = 3.5 , confidence interval [ ci ] = 2.5 - 3.9 , p < 0.001 ) . furthermore , the finding showed or of overweight and obesity in welfare compared to non - welfare is 19.3 ( or = 19.3 , ci = 2.5 - 3.9 , p = 0.04 ) . the result of this community intervention study revealed that nutrition intervention based on advocacy program had been successful to reduce the prevalence of underweight among poor girls . this study shows determinant factor of nutritional status of school age children was their socio - economic level . according to our knowledge , this is the first study , which determines the effect of a community intervention based on advocacy process on the malnutrition indices in a big city ( shiraz ) in iran . the other program in iran ( nffp ) is specified to deprived area and is not conducted in big cities . allocating millions of dollars to nffp by government , selecting the malnourished students through an active screening system at primary and middle schools , paying attention of policy makers to student 's nutrition have provided the opportunity to combat the problem . however , negligence of under - poverty line , providing poor snacks in terms of nutritional value and lack of variety are the main defects of this program . advocacy by definition is a blending of science , ethics and politics for comprehensive approaching health issues . by using advocacy program in california among the high school students for improving their nutrition and physical activity angeles unified school district participants emphasized on nutrition classes for families as well as students in addition to other interventions . in the present study another study revealed that evaluability assessment gave stakeholders the opportunity to reflect on the project and its implementation issues . it seems that in iran , free food program among the students not only is needed in deprived areas , but also it should be performed in big cities such as shiraz . at baseline , no significant difference was founded among wealthy students between the pre- and post - nutritional status intervention . in contrast , the numbers of students who have malnutrition decreased from 44% to 39.4% , which was identified as a significant among impecunious girls students . there was also a significant increase in the proportion of children with bmi that was normal for age ( 2 to + 1 sd ) most of the published community interventions showed better results among females compared with males . this difference in the impact of nutritional interventions between male and female might be related to the different age of puberty in the female population compared to the male population . in the age range of the present study female although , there is no nffp in big cities of iran , there are some programs for improving the nutritional status such as providing free milk in schools . a recent publication has shown that school feeding programs focus on milk supplementation had beneficial effects on the physical function and school performances specifically among girls in iran . the results of the mentioned study showed an improvement in the weight of children , psychological test 's scores and the grade - point average following this school feeding program . the intervention in the present study had focused on the snack intake in the school time . there are some reports regarding the nutrition transition in iran , which shows the importance of nutrition intervention to provide more healthy eating dietary habits among welfare groups of adolescents . hence , nutrition intervention especially in the form of nutrition education is needed in big cities and among welfare children and adolescents . although a study among iranian adolescents showed that dietary behavior of adolescents does not accord to their knowledge , which emphasize on the necessity of community intervention programs . a recent study regarding the major dietary pattern among iranian children showed the presence of four major dietary patterns , in which fast food pattern and sweet pattern as two major dietary patterns can be mentioned among iranian children . in advocacy program audience 's analysis accordingly , one of the prominent strategies in this study was working with media and was meeting with parent - teacher association that both of them were secondary target audiences . we also took into account policy makers in different levels , from national to local as primary audiences . advocacy team had several meetings with management and planning organization at national level and education organization of the fars province as well as principal of the targeted schools . providing nutritious snacks need contribution of private sector such as food industries or factories , but their benefits should be warranted . another choice was community involvement ; which can be achieved by female health volunteers who are working with the health system . advocacy team by using the support of charities and female health volunteers could establish a local factory that produced student 's snacks based on the new definition . however , there are some challenges on the way of expanding this program . mass production of the proposed snacks according to different desires and cultures and getting involvement of food industries with respect to marketing issues is one of those challenges . moreover , providing a supportive environment in order to change the food habits of the students and their parents among the wide range of the population require a sustainable and continuous inter - sector collaboration . although in a limited number of schools , in our study , interventions and advocacy program was successful , expanding this model to another areas around the country depends on convincing the policy makers at national level . in this regard , advocacy team should prepare evidenced based profile and transitional planning to convince the policy makers for improving the rule and regulation of nffp . the same as this study in other studies have also emphasized that there must be efforts to strengthen the capacity within the schools to deal with the nutritional problems either overweight , obesity or malnutrition by using of educational and nutritional intervention . assessing the dietary adherence is very important in nutrition intervention among population . as this population was children and adolescents we had a limitation in the blood sample collection to assess the subject 's dietary adherence . furthermore , this intervention was only focused on the intake of snack in school time and we did not have comprehensive information on the dietary intake of children and adolescents after school all over the day . the investigators propose further investigation in different areas of the country based on socio - cultural differences in order to make necessary modification and adapt this model to other areas . regarding the nutritional needs of the school age children , provision of a good platform for implementing and expanding this efficient model to the whole country based upon the socio - economic situation of each region is advisable to the moh and the moe . community nutrition intervention based on the advocacy process model is effective on reducing the prevalence of underweight specifically among female school aged children .\",\n",
       " 'abstract': \"<S> background : the present study was carried out to assess the effects of community nutrition intervention based on advocacy approach on malnutrition status among school - aged children in shiraz , iran.materials and methods : this case - control nutritional intervention has been done between 2008 and 2009 on 2897 primary and secondary school boys and girls ( 7 - 13 years old ) based on advocacy approach in shiraz , iran . </S> <S> the project provided nutritious snacks in public schools over a 2-year period along with advocacy oriented actions in order to implement and promote nutritional intervention . for evaluation of effectiveness of the intervention growth monitoring indices of pre- and post - intervention were statistically compared.results:the frequency of subjects with body mass index lower than 5% decreased significantly after intervention among girls ( p = 0.02 ) . </S> <S> however , there were no significant changes among boys or total population . </S> <S> the mean of all anthropometric indices changed significantly after intervention both among girls and boys as well as in total population . </S> <S> the pre- and post - test education assessment in both groups showed that the student 's average knowledge score has been significantly increased from 12.5  3.2 to 16.8  4.3 ( p < 0.0001).conclusion : this study demonstrates the potential success and scalability of school feeding programs in iran . </S> <S> community nutrition intervention based on the advocacy process model is effective on reducing the prevalence of underweight specifically among female school aged children . </S>\"}"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "raw_datasets[\"train\"][0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0uxehJnZlnWv"
   },
   "source": [
    "Since the `pubmed` data is extremely large, we are going to remove rows so that we have a training set of 8,000, a validation set of 2,000, and a test set of 2,000. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "id": "88RJWRuylqjh"
   },
   "outputs": [],
   "source": [
    "raw_datasets[\"train\"] = raw_datasets[\"train\"].select(range(1, 2001))\n",
    "raw_datasets[\"validation\"] = raw_datasets[\"validation\"].select(range(1, 501))\n",
    "raw_datasets[\"test\"] = raw_datasets[\"test\"].select(range(1, 501))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "WHUmphG3IrI3"
   },
   "source": [
    "To get a sense of what the data looks like, the following function will show some examples picked randomly in the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "id": "i3j8APAoIrI3"
   },
   "outputs": [],
   "source": [
    "import datasets\n",
    "import random\n",
    "import pandas as pd\n",
    "from IPython.display import display, HTML\n",
    "\n",
    "def show_random_elements(dataset, num_examples=5):\n",
    "    assert num_examples <= len(dataset), \"Can't pick more elements than there are in the dataset.\"\n",
    "    picks = []\n",
    "    for _ in range(num_examples):\n",
    "        pick = random.randint(0, len(dataset)-1)\n",
    "        while pick in picks:\n",
    "            pick = random.randint(0, len(dataset)-1)\n",
    "        picks.append(pick)\n",
    "    \n",
    "    df = pd.DataFrame(dataset[picks])\n",
    "    for column, typ in dataset.features.items():\n",
    "        if isinstance(typ, datasets.ClassLabel):\n",
    "            df[column] = df[column].transform(lambda i: typ.names[i])\n",
    "    display(HTML(df.to_html()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "SZy5tRB_IrI7"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>article</th>\n",
       "      <th>abstract</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>fatty acids ( fas ) , the components of phospholipids in organelle and cellular membranes , play important biological roles by maintaining or processing membrane protein function or fluidity.1 in addition , fas modulate vascular inflammation , a key mechanism of atherosclerosis , cerebral small vessel pathologies , and stroke , by altering intracellular signal transduction or controlling lipid mediators such as prostaglandins , thromboxanes , or leukotrienes.2 among fas , 3-polyunsaturated fas ( 3-pufas ) , such as eicosapentaenoic acid ( epa ) and docosahexaenoic acid ( dha ) , are potent anti - inflammatory molecules . epa and dha decrease expression of receptors for chemoattractants on blood inflammatory cells and prohibit migration of neutrophils or monocytes . therefore , 3-pufas may protect from atherosclerotic changes.3,4 several clinical studies have emphasized the role of fas in the risk or occurrence of stroke5 or cardiovascular disease6 ; however , the effects due to the composition of fas on stroke or cardiovascular disease remain controversial . high - dose 3-pufas has been reported to have beneficial effects on cardiac or sudden death.7 high levels of plasma 3-pufas can decrease the risk of myocardial infarction.8 in terms of stroke , low levels of circulating 3-pufas in the blood is a risk factor for ischemic and hemorrhagic stroke.9 a decreased proportion of linoleic acid is also associated with ischemic stroke.10 compared to normal controls , stroke patients with moderate - to - severe intracranial arterial stenosis or occlusion had decreased levels of dha.11 on the other hand , a recent meta - analysis revealed that the evidence for the beneficial effects of 3-pufas is insufficient in adults with peripheral arterial disease associated with poor cardiovascular outcome.12 to date , little information is available on the relationship between the composition of fas and prognosis of stroke . therefore , we investigated whether the composition of fas was associated with stroke severity on hospital admission and functional outcomes at 3-months follow - up of patients with acute non - cardiogenic ischemic stroke . between september 2007 and may 2010 , we prospectively enrolled patients diagnosed with a first - episode of ischemic stroke and admitted to our hospital within 7 days after onset of symptoms . the patient 's demographic information as well as past medical , medication and familial history ; brain imaging studies ( computerized tomography [ ct ] and/or magnetic resonance imaging [ mri ] ) ; vascular imaging studies ( digital subtraction angiography , ct angiography , or mr angiography ) ; chest radiography ; 12-lead electrocardiography ; electrocardiography monitoring during a median time period of 3 days at the intensive stroke care unit ; transthoracic echocardiography ; and routine blood test data were collected.13 patients were preferentially excluded if they did not agree to provide blood samples for our study , or if they were taking lipid - lowering agents such as statins , niacin , fenofibrate , or 3-pufas as components of health foods and supplements.11 of a total of 401 subjects were enrolled , however those , who on the basis of the trial of org 10,172 in acute stroke treatment classification system,14 had moderate or high risk cardiac sources of embolism ( n=127 ) were excluded . patients who had undetermined stroke subtype ( n=45 : negative evaluation , n=19 : two or more causes identified ) or those who had rare causes of stroke subtype ( n=10 ) , such as moyamoya disease , arterial dissection , or venous thrombosis were also excluded from this study . moreover , patients who received incomplete vascular imaging work - up ( n=2 ) and those who had transient ischemic attacks with negative diffusion - weighted images ( n=42 ) were not enrolled . for extracranial arteries , the degree of arterial stenosis was measured according to the published method used in the north american symptomatic carotid endarterectomy trial.15 for intracranial arteries , the degree of arterial stenosis was measured based on the methods used in the warfarin - aspirin symptomatic intracranial disease study.16 vascular images were evaluated by two independent vascular neurologists , who were blinded to the clinical information . inter - observer agreement on the presence of more than 50% stenosis and/or occlusion was excellent ( kappa=0.95 ) . the severity of neurologic deficits was determined using the national institutes of health stroke scale ( nihss ) on admission.17 a recurrent ischemic stroke was considered in the presence of acute onset of focal neurological signs of more than 24 hours duration with evidence of a new ischemic lesion on ct or mri scan , or when new lesions were absent but the clinical syndrome was consistent with stroke from admission to 3 months after the index stroke . functional outcomes were assessed using the modified rankin scale ( mrs ) 3 months after the index stroke . blood samples for lipid profiles were collected from the patients within 24 hours of admission and at a fasting state of more than 12 hours . they were centrifuged to separate plasma or serum from the whole blood and then were stored at -70 until analysis could be performed . the methods for measuring fa composition have been described previously.11 briefly , plasma total lipids were extracted according to the folch method18 and the phospholipid fraction was isolated by thin layer chromatography using a development solvent composed of hexane , diethyl ether , and acetic acid ( 80:20:2 ) . the phospholipid fractions were then methylated to fa methyl esters ( fames ) by the lepage and roy method.19 the fames of individual fas of phospholipids were separated by gas chromatography using a model 6890 apparatus ( agilent technologies , palo alto , ca , usa ) and a 30 m omegawaz tm 250 capillary column ( supelco , bellefonte , pa , usa).20 peak retention times were obtained by comparison with known standards ( 37 component fame mix and pufa-2 , supelco ; glc37 , nucheck prep , elysian , mn , usa ) and analyzed with chemstation software ( agilent technologies ) . the average of duplicate measurements for each sample was calculated.11,21 plasma phospholipid fas were expressed as the percentage of total fas . the sum (  ) of 16:0 palmitic acid , 18:0 stearic acid , 20:0 arachidonic acid , and 22:0 behenic acid was defined as the  saturated fatty acids ; 16:1 palmitoleic acid , 18:1 9 oleic acid , and 22:1 erucic acid were defined as the  monounsaturated fatty acids ; 18:2 6 linolenic acid , 18:3 6 -linolenic acid , 20:3 6 dihomo--linolenic acid , and 20:4 6 arachidonic acid were defined as the  6-pufas ; and 18:3 3 -linolenic acid , 20:3 3 5 - 8 - 11-eicosatrienoic acid , 20:5 3 epa , and 22:6 3 dha were reported as the  3-pufas . hypertension was defined as having resting systolic blood pressure 140 mmhg or diastolic blood pressure 90 mmhg on repeated measurements , or receiving treatment with anti - hypertensive medications . diabetes mellitus was diagnosed when the patient had fasting blood glucose level 7.0 mmol / l , or was being treated with oral hypoglycemic agents or insulin . patients were defined as smokers , if they were smokers at the stroke event or if they had stopped smoking within 1 year before the stroke event . body mass index was estimated by dividing body weight by height ( kg / m ) . the presence of coronary artery disease was defined as a patient history of unstable angina , myocardial infarction , or angiographically confirmed coronary artery disease . all statistical analyses were performed using the windows spss software package ( version 18.0 , chicago , il , usa ) . independent t test , mann - whitney u test , one - way analysis of variance ( anova ) with bonferroni post hoc analysis , and kruskal - wallis test were used to compare the continuous variables . categorical variables were compared using the chi - square test or fisher 's exact test . continuous variables were expressed as meansstandard deviations ( sd ) or as medians and interquartile ranges ( iqr ) . univariate and multivariate linear regression analyses were performed to determine the factors associated with nihss at admission . functional outcome was dichotomized into good outcome ( mrs &lt;3 ) or poor outcome ( mrs 3 ) . univariate and multivariate binary logistic regression analyses were performed to determine the predictive factors for functional outcome . to assess the goodness of fit of the logistic regression model , cox &amp; snell r square was calculated and the hosmer - lemeshow test was performed . between september 2007 and may 2010 , we prospectively enrolled patients diagnosed with a first - episode of ischemic stroke and admitted to our hospital within 7 days after onset of symptoms . the patient 's demographic information as well as past medical , medication and familial history ; brain imaging studies ( computerized tomography [ ct ] and/or magnetic resonance imaging [ mri ] ) ; vascular imaging studies ( digital subtraction angiography , ct angiography , or mr angiography ) ; chest radiography ; 12-lead electrocardiography ; electrocardiography monitoring during a median time period of 3 days at the intensive stroke care unit ; transthoracic echocardiography ; and routine blood test data were collected.13 patients were preferentially excluded if they did not agree to provide blood samples for our study , or if they were taking lipid - lowering agents such as statins , niacin , fenofibrate , or 3-pufas as components of health foods and supplements.11 of a total of 401 subjects were enrolled , however those , who on the basis of the trial of org 10,172 in acute stroke treatment classification system,14 had moderate or high risk cardiac sources of embolism ( n=127 ) were excluded . patients who had undetermined stroke subtype ( n=45 : negative evaluation , n=19 : two or more causes identified ) or those who had rare causes of stroke subtype ( n=10 ) , such as moyamoya disease , arterial dissection , or venous thrombosis were also excluded from this study . moreover , patients who received incomplete vascular imaging work - up ( n=2 ) and those who had transient ischemic attacks with negative diffusion - weighted images ( n=42 ) were not enrolled . for extracranial arteries , the degree of arterial stenosis was measured according to the published method used in the north american symptomatic carotid endarterectomy trial.15 for intracranial arteries , the degree of arterial stenosis was measured based on the methods used in the warfarin - aspirin symptomatic intracranial disease study.16 vascular images were evaluated by two independent vascular neurologists , who were blinded to the clinical information . inter - observer agreement on the presence of more than 50% stenosis and/or occlusion was excellent ( kappa=0.95 ) . the severity of neurologic deficits was determined using the national institutes of health stroke scale ( nihss ) on admission.17 a recurrent ischemic stroke was considered in the presence of acute onset of focal neurological signs of more than 24 hours duration with evidence of a new ischemic lesion on ct or mri scan , or when new lesions were absent but the clinical syndrome was consistent with stroke from admission to 3 months after the index stroke . functional outcomes were assessed using the modified rankin scale ( mrs ) 3 months after the index stroke . blood samples for lipid profiles were collected from the patients within 24 hours of admission and at a fasting state of more than 12 hours . they were centrifuged to separate plasma or serum from the whole blood and then were stored at -70 until analysis could be performed . the methods for measuring fa composition have been described previously.11 briefly , plasma total lipids were extracted according to the folch method18 and the phospholipid fraction was isolated by thin layer chromatography using a development solvent composed of hexane , diethyl ether , and acetic acid ( 80:20:2 ) . the phospholipid fractions were then methylated to fa methyl esters ( fames ) by the lepage and roy method.19 the fames of individual fas of phospholipids were separated by gas chromatography using a model 6890 apparatus ( agilent technologies , palo alto , ca , usa ) and a 30 m omegawaz tm 250 capillary column ( supelco , bellefonte , pa , usa).20 peak retention times were obtained by comparison with known standards ( 37 component fame mix and pufa-2 , supelco ; glc37 , nucheck prep , elysian , mn , usa ) and analyzed with chemstation software ( agilent technologies ) . the average of duplicate measurements for each sample was calculated.11,21 plasma phospholipid fas were expressed as the percentage of total fas . the sum (  ) of 16:0 palmitic acid , 18:0 stearic acid , 20:0 arachidonic acid , and 22:0 behenic acid was defined as the  saturated fatty acids ; 16:1 palmitoleic acid , 18:1 9 oleic acid , and 22:1 erucic acid were defined as the  monounsaturated fatty acids ; 18:2 6 linolenic acid , 18:3 6 -linolenic acid , 20:3 6 dihomo--linolenic acid , and 20:4 6 arachidonic acid were defined as the  6-pufas ; and 18:3 3 -linolenic acid , 20:3 3 5 - 8 - 11-eicosatrienoic acid , 20:5 3 epa , and 22:6 3 dha were reported as the  3-pufas . hypertension was defined as having resting systolic blood pressure 140 mmhg or diastolic blood pressure 90 mmhg on repeated measurements , or receiving treatment with anti - hypertensive medications . diabetes mellitus was diagnosed when the patient had fasting blood glucose level 7.0 mmol / l , or was being treated with oral hypoglycemic agents or insulin . patients were defined as smokers , if they were smokers at the stroke event or if they had stopped smoking within 1 year before the stroke event . body mass index was estimated by dividing body weight by height ( kg / m ) . the presence of coronary artery disease was defined as a patient history of unstable angina , myocardial infarction , or angiographically confirmed coronary artery disease . all statistical analyses were performed using the windows spss software package ( version 18.0 , chicago , il , usa ) . independent t test , mann - whitney u test , one - way analysis of variance ( anova ) with bonferroni post hoc analysis , and kruskal - wallis test were used to compare the continuous variables . categorical variables were compared using the chi - square test or fisher 's exact test . continuous variables were expressed as meansstandard deviations ( sd ) or as medians and interquartile ranges ( iqr ) . univariate and multivariate linear regression analyses were performed to determine the factors associated with nihss at admission . functional outcome was dichotomized into good outcome ( mrs &lt;3 ) or poor outcome ( mrs 3 ) . univariate and multivariate binary logistic regression analyses were performed to determine the predictive factors for functional outcome . to assess the goodness of fit of the logistic regression model , cox &amp; snell r square was calculated and the hosmer - lemeshow test was performed . the demographic data of study subjects and comparative analysis according to functional outcome at 3 months after index stroke are summarized in table 1 . of all patients , there were 60 ( 38.5% ) patients with large artery atherosclerosis stroke subtype and 96 ( 61.5% ) patients with small vessel occlusions . the meanssd of proportions of epa and dha were 2.00.7 and 8.91.4 , respectively . in the case of  3-pufa , considering stroke subtypes , there was no difference between large artery atherosclerosis and small vessel occlusion stroke subtypes in terms of the proportion of epa , dha , or  3-pufas ( supplementary table 1 ) . of the 156 patients , 122 ( 78.2% ) patients had good functional outcome with meanssd of epa and dha proportions of 2.10.7 and 9.11.3 , respectively . the remaining 34 ( 21.8% ) patients had poor outcome with a relatively smaller proportion of epa ( 1.80.6 , p=0.032 ) and dha ( 8.11.3 , p=0.001 ) . the proportion of  3-pufa in patients with poor outcome was significantly lower than that in patients with good outcome ( 10.81.6 vs. 12.21.9 , p=0.001 ) ( table 1 ) . after adjusting for factors including age , sex , and variables with p&lt;0.1 in the univariate analysis ( stroke subtypes , hemoglobin , high density lipoprotein , high sensitivity c - reactive protein , fasting glucose , 16:0 palmitic acid , and  saturated fatty acids ) , lower proportion of epa and dha were independently associated with stroke severity on admission (  : -0.751 , standard error ( se ) : 0.376 , p=0.048 for epa ,  : -0.610 , se : 0.215 , p=0.005 for dha ) . moreover , the  3-pufa was significantly associated with stroke severity on admission (  : -0.462 , se : 0.156 , p=0.004 ) ( table 2 ) ( fig . 1 ) . considering stroke subtypes , dha and  3-pufa were correlated with stroke severity on admission , in both large artery atherosclerosis ( even though it showed tendency for  3-pufa , p=0.065 ) and small vessel occlusion , however epa ( supplementary table 2 ) did not appear to be correlated in multivariate linear regression analysis . there were six recurrent stroke cases and events , and epa was relatively lower in the recurrent group compared to the non - recurrent group ( 1.60.2 vs. 2.00.7 , p=0.006 ) . however , the proportions of dha and  3-pufas were not different between the two groups ( supplementary table 3 ) . regarding functional outcome at three months after index stroke , a lower proportion of dha ( odds ratio ( or ) : 0.20 , 95% confidence interval ( ci ) : 0.04 - 0.88 , p=0.033 ) and  3-pufa ( or : 0.22 , 95% ci : 0.05 - 0.84 , p=0.028 ) showed a significant relationship with poor functional outcome . however , epa was not independently associated with poor functional outcome ( or : 0.60 , 95% ci : 0.12 -2.93 , p=0.533 ) in multivariate analysis after adjusting for age , sex , smoking status , nihss score , stroke subtypes , or 16:0 palmitic acid ( table 3 ) . considering stroke subtypes , a lower proportion of dha and  3-pufas was independently associated with poor functional outcome in both the large artery atherosclerosis subtype ( or : 0.62 , 95% ci : 0.42 - 0.93 , p=0.023 for dha , or : 0.65 , 95% ci : 0.47 - 0.90 , p=0.011 for  3-pufa ) and the small vessel occlusion subtype ( or : 0.49 , 95% ci : 0.28 - 0.85 , p=0.012 for dha , or : 0.64 , 95% ci : 0.42 - 0.98 , p=0.044 for  3-pufa ) ( supplementary table 4 ) . the demographic data of study subjects and comparative analysis according to functional outcome at 3 months after index stroke are summarized in table 1 . of all patients , there were 60 ( 38.5% ) patients with large artery atherosclerosis stroke subtype and 96 ( 61.5% ) patients with small vessel occlusions . the meanssd of proportions of epa and dha were 2.00.7 and 8.91.4 , respectively . in the case of  3-pufa , considering stroke subtypes , there was no difference between large artery atherosclerosis and small vessel occlusion stroke subtypes in terms of the proportion of epa , dha , or  3-pufas ( supplementary table 1 ) . of the 156 patients , 122 ( 78.2% ) patients had good functional outcome with meanssd of epa and dha proportions of 2.10.7 and 9.11.3 , respectively . the remaining 34 ( 21.8% ) patients had poor outcome with a relatively smaller proportion of epa ( 1.80.6 , p=0.032 ) and dha ( 8.11.3 , p=0.001 ) . the proportion of  3-pufa in patients with poor outcome was significantly lower than that in patients with good outcome ( 10.81.6 vs. 12.21.9 , p=0.001 ) ( table 1 ) . after adjusting for factors including age , sex , and variables with p&lt;0.1 in the univariate analysis ( stroke subtypes , hemoglobin , high density lipoprotein , high sensitivity c - reactive protein , fasting glucose , 16:0 palmitic acid , and  saturated fatty acids ) , lower proportion of epa and dha were independently associated with stroke severity on admission (  : -0.751 , standard error ( se ) : 0.376 , p=0.048 for epa ,  : -0.610 , se : 0.215 , p=0.005 for dha ) . moreover , the  3-pufa was significantly associated with stroke severity on admission (  : -0.462 , se : 0.156 , p=0.004 ) ( table 2 ) ( fig . 1 ) . considering stroke subtypes , dha and  3-pufa were correlated with stroke severity on admission , in both large artery atherosclerosis ( even though it showed tendency for  3-pufa , p=0.065 ) and small vessel occlusion , however epa ( supplementary table 2 ) did not appear to be correlated in multivariate linear regression analysis . there were six recurrent stroke cases and events , and epa was relatively lower in the recurrent group compared to the non - recurrent group ( 1.60.2 vs. 2.00.7 , p=0.006 ) . however , the proportions of dha and  3-pufas were not different between the two groups ( supplementary table 3 ) . regarding functional outcome at three months after index stroke , a lower proportion of dha ( odds ratio ( or ) : 0.20 , 95% confidence interval ( ci ) : 0.04 - 0.88 , p=0.033 ) and  3-pufa ( or : 0.22 , 95% ci : 0.05 - 0.84 , p=0.028 ) showed a significant relationship with poor functional outcome . however , epa was not independently associated with poor functional outcome ( or : 0.60 , 95% ci : 0.12 -2.93 , p=0.533 ) in multivariate analysis after adjusting for age , sex , smoking status , nihss score , stroke subtypes , or 16:0 palmitic acid ( table 3 ) . considering stroke subtypes , a lower proportion of dha and  3-pufas was independently associated with poor functional outcome in both the large artery atherosclerosis subtype ( or : 0.62 , 95% ci : 0.42 - 0.93 , p=0.023 for dha , or : 0.65 , 95% ci : 0.47 - 0.90 , p=0.011 for  3-pufa ) and the small vessel occlusion subtype ( or : 0.49 , 95% ci : 0.28 - 0.85 , p=0.012 for dha , or : 0.64 , 95% ci : 0.42 - 0.98 , p=0.044 for  3-pufa ) ( supplementary table 4 ) . our study revealed that 3-pufas , especially dha , were associated with stroke severity on hospital admission and poor functional outcome even after adjusting for the nihss score , which is considered a strong predictive factor for stroke outcome . for example , treatment with dha - albumin complex in animal studies decreased brain injury after a transient and permanent focal cerebral ischaemia.22 dha showed anti - inflammatory and neuroprotective effects by decreasing oxidative stress via activation of nuclear factor e2-related factor 2 and heme oxygenase-1 expression and by attenuating c - jun phosphorylation , or the activating protein-1 signaling pathway.23 consistently , studies in humans have produced similar results . one previous study in 281 japanese patients with acute ischemic stroke diagnosed within 24 hours of onset , showed that the epa / arachidonic acid ( aa ) ratio , dha / aa ratio , and the epa+dha / aa ratio were independently and negatively associated with early neurological deterioration.24 furthermore , a population - based study in the u.s.a , which included 2,692 elderly adults without prevalence of stroke or cardiovascular disease , revealed that higher plasma 3-polyunsaturated fas ( epa , dha , and total 3 fas ) were associated with lower mortalities.25 overall , these previous studies showed that 3-pufas ( especially epa and dha ) were related to vascular outcome , which is in line with our study results . however , the reason why epa was not independently associated with poor functional outcome in our study population remains to be elucidated . dha exerts vasodilating properties by stimulating nitric oxide release in the vascular endothelium , and hence may also be responsible for decreasing heart rate variability and potentially for dyslipidemia improvement , effects not observed by epa.26 loss of nitric oxide control , heart rate variation , and dyslipidemia may be related to poor vascular disease outcome,27 and our study results can be explained by the correlation observed with dha , and not epa , and with functional outcome . in addition , the difference in race and staple food of the study population could represent another possible explanation . our results demonstrated a relationship between 3-pufas and stroke severity on hospital admission and poor functional outcomes , in both the large artery atherosclerosis and the small vessel occlusion stroke subtypes . 3-pufas give rise to anti - inflammatory molecules ( resolvins and protectins ) through lipoxygenase or cyclo - oxygenase pathways.28,29 resolvins or protectins play an important role in the resolution of inflammation , which decreases atherosclerotic changes and tissue injuries . because cerebral small vessel pathologies such as lacunar infarction , white matter changes , and cerebral microbleeds are linked to inflammatory reactions30 and increase arterial stiffness31 caused by progressive atherosclerosis , these anti - inflammatory and anti - atherogenic effects of 3-pufas could explain the results observed in our study . moreover , because progressive cerebral atherosclerosis is associated with a poor stroke outcome,32 our finding relative to the association between 3-pufas and stroke prognosis may be valid . in addition , the atherosclerotic plaque stabilization effect of 3-pufas may be an important mechanism . a previous study performed on patients awaiting carotid endarterectomy showed that plaques from patients treated with fish oil ( 1.4 g 3-polyunsaturated fas per day ) had a well - formed thick fibrous cap ( i.e. less vulnerable ) atheroma compared to that in patients of the placebo group.33 pathologically , infiltration of macrophages was less severe in patients treated with fish oil.33 furthermore , another prospective study confirmed that patients treated with fish oil had lower plaque inflammation and instability.34 because the vulnerability of the atherosclerotic plaque is a very important determinant of thrombosis - related stroke , as well as the degree of arterial stenosis , our results relative to the relationship between stroke outcome and 3-pufas is within expectation . lastly , a recent study revealed that 3-pufas enhanced cerebral angiogenesis in animal models,35 and because increased angiogenesis could augment brain repair and improve long - term functional recovery after cerebral infarction , these findings may support our study.35 one limitation of our study is that the blood samples were obtained from acute stroke patients on admission . therefore , the fatty acids and their composition were not serially assessed during the time course of the stroke . moreover , even though we prospectively enrolled our study subjects , our study design is mainly cross - sectional . in addition , the short - term of observation with a small sample size is another limitation of this study . further studies with a long - term follow - up and larger population size are needed . finally , our study design was not that of a randomized control study . furthermore , because stroke severity on admission correlated with both lower proportion of 3-pufas and functional outcome at 3 months , there could be a bias due to the interaction between severity on admission and 3-pufas . that is , there may be a possibility that the functional outcome at 3 months is only weakly associated with 3-pufas . our results demonstrate that 3-pufa levels correlate with stroke severity at admission and functional outcomes at 3 months . 3-pufas may be considered potential blood biomarkers for prognosis of acute non - cardiogenic ischemic stroke patients . comparison of proportions of 3-pufas according to stroke subtypes the relationship between fatty acids composition and stroke severity on admission according to stroke subtypes comparison of functional outcome based on occurrence of recurrent stroke the relationship between fatty acids composition and poor functional outcome according to stroke subtype</td>\n",
       "      <td>&lt;S&gt; background and purposealterations in blood fatty acid ( fa ) composition are associated with cardiovascular diseases . &lt;/S&gt; &lt;S&gt; we investigated whether plasma fa composition was related to stroke severity and functional outcome in acute ischemic stroke patients.methodswe prospectively enrolled 156 patients with first - episode cerebral infarction , within 7 days of symptom onset . &lt;/S&gt; &lt;S&gt; the proportion of fas was analyzed using gas chromatography , and the summation of the omega-3 polyunsaturated fatty acids ( 3-pufa ) , 18:3 3 -linolenic acid , 20:3 3 eicosatrienoic acid , 20:5 3 eicosapentaenoic acid ( epa ) , and 22:6 3 docosahexaenoic acid ( dha ) was reported as 3-pufas . &lt;/S&gt; &lt;S&gt; stroke severity was assessed using the national institutes of health stroke scale ( nihss ) score on admission . &lt;/S&gt; &lt;S&gt; poor functional outcome was defined by modified rankin scale ( mrs ) 3 at three months after the index stroke.resultslower proportions of epa ( =-0.751 ) , dha ( =-0.610 ) , and 3-pufas ( =-0.462 ) were independently associated with higher nihss score , after adjusting for stroke subtype , hemoglobin , high density lipoprotein , high sensitivity c - reactive protein , fasting glucose , 16:0 palmitic acid , and saturated fatty acids . moreover &lt;/S&gt; &lt;S&gt; , a lower proportion of dha ( odds ratio [ or ] : 0.20 , 95% confidence interval [ ci ] : 0.04 - 0.88 ) , and 3-pufas ( or : 0.22 , 95% ci : 0.05 - 0.84 ) showed an independent relationship with poor functional outcome after adjusting for age , sex , smoking status , nihss score , stroke subtype , and 16:0 palmitic acid.conclusionsour results demonstrate that 3-pufas correlated with stroke severity on admission and functional outcomes at 3 months . &lt;/S&gt; &lt;S&gt; 3-pufas are potential blood biomarkers for prognosis of acute non - cardiogenic ischemic stroke patients . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>lignocellulosic materials represent an abundant and inexpensive source of sugars and can be microbiologically converted to industrial products . xylitol ( c5h12o5 ) , a sugar alcohol obtained from xylose , is generated during the metabolism of carbohydrates in animals and humans . xylitol was present in fruits and vegetables , at low concentration , which makes its production from these sources economically unfeasible . its sweetening power was comparable to that of sucrose and is higher than that of sorbitol and mannitol . in addition to reducing dental caries , xylitol also promotes tooth enamel remineralization by reversing small lesions . this happens because , when in contact with xylitol , the saliva seems to be favorably influenced ; the chemical composition of xylitol induces the calcium ions and phosphate . for these characteristics , xylitol was a feed stock of great interest to food , odontological , and pharmaceutical industries . however it was expensive and it requires several steps of xylose purification before the chemical reaction [ 4 , 9 , 10 ] . xylitol production through bioconversion has been proposed as for utilizing microorganism such as yeast , bacteria , and fungi [ 11 , 12 ] . among these , yeast has shown to possess desirable properties for xylitol production [ 13 , 14 ] . therefore , for the present study , yeast strain pachysolen tannophilus was selected for xylitol production . furthermore studies have shown that nutritional factors including sources of carbon and nitrogen can influence xylitol production .   corncob is a large volume solid waste for using sweet corn processing industry in india . they are currently used as animal feed or returned to the harvested field for land application . corncob contains approximately over 40% of the dry matter in residues   and thus has value has a raw material for production of xylose , xylitol , arabinose , xylobiose , and xylo oligosaccharides . these carbohydrates mainly consist of the xylose and other minor pentose [ 1820 ] . among various agricultural wastes , corncob was regarded as promising agricultural resources for microbial xylitol production . in microbial production of xylitol from corncob , the cobs were first hydrolysed to produce from hemicelluloses by acid hydrolysis and the corncob hydrolysate is then used as the medium for xylitol production . the bioconversion of xylitol is influenced by the concentration of various ingredients in culture medium . this study also investigates the effect of process variables such as ph , temperature , substrate concentration , inoculum size , and agitation speed on xylitol yield . response surface methodology ( rsm ) is a mathematical and statistical analysis , which is useful for the modeling and analysis problems that the response of interest is influenced by several variables . rsm was utilized extensively for optimizing different biotechnological process [ 22 , 23 ] .   in the present study , the screening and optimization of medium composition and process variables for xylitol production by pachysolen tannophilus using plackett - burman and rsm were reported . the plackett - burman screening design was applied for knowing the most significant nutrients enhancing xylitol production . then , box - behnken design and central composite design ( ccd ) were applied to determine the optimum level of significant nutrients and process variables , respectively . the yeast strain pachysolen tannophilus ( mtcc 1077 ) was collected from microbial type culture collection and gene bank , chandigarh . the lyophilized stock cultures were maintained at 4c on culture medium supplemented with 20  g of agar . the medium composition ( g / l ) was compressed of the following : malt extract : 3.0 ; yeast extract : 3.0 ; peptone : 5.0 ; glucose : 10.0 at ph : 7 . corncob was collected from perambalur farms , tamil nadu , india , and was dried in sunlight for 2 days , crushed , and sieved for different mesh size ranging from 0.45  mm to 0.9  mm ( 2040 mesh ) and used for further studies . the composition of the corncob ( g / l ) : xylose : 28.7 , glucose : 5.4 , arabinose : 3.7 , cellobiose : 0.5 , galactose : 0.7 , mannose : 0.4 , acetic acid : 2 , furfural : 0.8 , hydroxymethyl furfural : 0.2 was used for xylitol production .   2  g of corncobs at a solid loading of 10% ( w / w ) was mixed with dilute sulfuric acid ( 0.1% ( w / v ) ) and pretreated in an autoclave at 120c with residence time of 1 hour . the liquid fraction was separated by filtration and the unhydrolysed solid residue was washed with warm water ( 60c ) . hemicellulose acid hydrolysate was heated at 100c for 15  min to reduce the volatile components . the hydrolysate was overlined with solid ca(oh)2 up to ph 10 , in combination with 0.1% sodium sulfite , and filtered to remove the insoluble materials . activated charcoal treatment was an efficient and economic method of reduction in the amount of phenolic compounds , acetic acid , aromatic compounds , furfural , and hydroxymethylfurfural normally found in hemicellulosic hydrolysates . after centrifugation , the solutions were mixed with powdered charcoal at 5% ( w / v ) for 30 and stirred ( 100  rpm ) at 30c . the liquor was recovered by filtration , chemically characterized , and used for culture media . fermentation was carried out in 250  ml erlenmeyer flasks with 100  ml of pretreated corncob hemicelluloses hydrolysate is adjusted to ph 7 with 2  m h2so4 or 3  m naoh and supplemented with different nutrients concentration for tests according to the selected factorial design , were used for fermentation medium and sterilized at 120c for 20  mins . the flasks were maintained at 30c for agitation at 200  rpm for 48 hours . after the optimization of medium composition , the fermentation was carried out with different parameter levels ( table 5 ) with the optimized media for tests according to the selected factorial design . during the preliminary screening , the experiments were carried out for 5 days and the maximum production was obtained in 48 hours . sugar and sugar alcohol in the culture broth were measured by high - performance liquid chromatography ( hplc ) , model lc-10-ad ( shimadzu , tokyo , japan ) equipped with a refractive index ( ri ) detector . the chromatography column used was a aminex hpx-87h ( 300  7.8  mm ) column at 80c with 5  mm h2so4 as mobile phase at a flow rate of 0.4  ml / min , and the injected sample volume was 20  l . the rsm has several classes of designs , with its own properties and characteristics . central composite design ( ccd ) , box - behnken design , and three - level factorial design are the most popular designs applied by the researchers . a prior knowledge with understanding of the related bioprocesses is necessary for a realistic modeling approach . it assumes that there are no interactions between the different variables in the range under consideration . plackett - burman experimental design is a fractional factorial design and the main effects of such a design may be simply calculated as the difference between the average of measurements made at the high level ( + 1 ) of the factor and the average of measurements at the low level ( 1 ) . to determine which variables significantly affect xylitol production , nine variables were screened in 12 experimental runs ( table 1 ) , and insignificant ones are eliminated in order to obtain a smaller , manageable set of factors . the low level ( 1 ) and high level ( + 1 ) of each factor ( 1 , + 1 ) g / l ) : k2hpo4 ( 6.6 , 7 ) , yeast extract ( 1.5 , 5 ) , peptone ( 2 , 5 ) , kh2po4 ( 1.2 , 3.6 ) , xylose ( 9.8 , 10.2 ) , ( nh4)2so4 ( 1 , 4 ) , mgso47h2o ( 0.7 , 1.3 ) , malt ( 2.8 , 3.2 ) , and glucose ( 9.8 , 10.2 ) , and they were coded with a , b , c , d , e , f , g , h , i , respectively . the statistical software package  minitab 16  is used for analyzing the experimental data . once the critical factors are identified through the screening , the box - behnken design was used to obtain a quadratic model after the central composite design ( ccd ) was used to optimize the process variables and obtain a quadratic model . the box - behnken design and ccd was used to study the effects of the variables towards their responses and subsequently in the optimization studies . this method was suitable for fitting a quadratic surface , and it helps to optimize the effective parameters with a minimum number of experiments , as well as to analyze the interaction between the parameters . in order to determine the relationship between the factors and response variables , a regression design was employed to model a response as a mathematical function ( either known or empirical ) for few continuous factors , and good model parameter estimates are desired . the coded values of the process parameters are determined by the following equation : \\n ( 1)xi = xix0x , \\n where xi is coded value of the ith variable , xi is uncoded value of the ith test variable , and x0 is uncoded value of the ith test variable at center point . the regression analysis is performed to estimate the response function as a second - order polynomial : \\n ( 2)y=0+i=1kixi+i=1kiixi2i=1 ,  i &lt; jk1  j=2kijxixj , \\n where y is the predicted response , 0 constant , and i , ii , ij are coefficients estimated from regression . they represent the linear , quadratic , and cross - products of xi and xj on response . the regression and graphical analysis are carried out using design - expert software ( version 7.1.5 , stat - ease , inc . , the adequacy of the models is further justified through analysis of variance ( anova ) . lack - of - fit is a special diagnostic test for adequacy of a model that compares the pure error , based on the replicate measurements to the other lack of fit , based on the model performance . f value , calculated as the ratio between the lack - of - fit mean square and the pure error mean square , is the statistic parameter used to determine whether the lack - of - fit is significant or not , at a significance level . the statistical model was validated with respect to xylitol production under the conditions predicted by the model in shake - flask level . samples were drawn at the desired intervals and xylitol production was determined as described above . plackett - burman experiments ( table 1 ) showed a wide variation in xylitol production . this variation reflected the importance of optimization to attain higher productivity . from the pareto chart shown in figure 1 the variables , namely , peptone , xylose , mgso47h2o , and yeast extract , were selected for further optimization to attain a maximum response . the levels of factors and the effect of their interactions on xylitol production were determined by box - behnken design of rsm . the design matrix of experimental results by tests was planned according to the 29 full factorial designs . twenty - nine experiments were performed at different combinations of the factors shown in table 2 , and the central point was repeated five times . the predicted and observed responses along with design matrix are presented in table 3 , and the results were analyzed by anova . the second - order regression equation provided the levels of xylitol production as a function of peptone , xylose , mgso47h2o , and yeast extract , which can be presented in terms of coded factors as in the following equation : \\n ( 3)y=0.70 + 0.053a+0.018b+0.057c+0.054d  + 0.092ab(2.500e003)ac  + ( 1.000e002)ad+0.028bc+0.077bd  0.040cd0.083a20.16b20.076c20.11d2 , \\n where y is the xylitol yield ( g / g ) and a , b , c , and d were peptone , xylose , mgso47h2o , and yeast extract , respectively . there is only a 0.01% chance that a  model f - value  this large could occur due to noise . values of  prob &gt; f  less than 0.05 indicate that model terms are significant . values greater than 0.1 indicate that model terms are not significant . in the present work , linear terms of a , c , and  d and all the square effects of a , b , c , and d and the combination of ab , bd , and cd were significant for xylitol production . the coefficient of determination ( r ) for xylitol production was calculated as 0.9634 , which is very close to 1 and can explain up to 96.00% variability of the response . the predicted r value of 0.7898 was in reasonable agreement with the adjusted r value of 0.9267 . the adequate precision value of 16.010 indicates an adequate signal and suggests that the model can navigate the design space . the above model can be used to predict the xylitol production within the limits of the experimental factors that the actual response values agree well with the predicted response values . experimental conditions for optimization of the process variables for xylitol yield were determined by ccd of rsm . the design matrix of experimental results by tests was planned according to the 50 full factorial designs , and the central point was repeated eight times . the predicted and observed responses along with design matrix are presented in table 6 and the results were analyzed by anova . the second - order regression equation provided the levels of xylitol production as a function of temperature , substrate concentration , ph , agitation speed , and inoculums size , which can be presented in terms of coded factors as in the following equation : \\n ( 4)y=0.79 + 0.025a+0.043b+0.049c  + 0.030d+0.038e0.029ab(4.063e003)ac  + 0.018ad+(2.187e003)ae(9.688e003)bc  + 0.014bd+(5.312e003)be+(1.562e003)cd  + ( 5.312e003)ce(3.125e004)de0.040a2  0.041b20.046c20.027d20.034e2 , \\n where y was the xylitol yield ( g / g ) and a , b , c , d , and e are temperature , substrate concentration , ph , agitation speed , and inoculums size , respectively . there is only a 0.01% chance that a  model f - value  this large could occur due to noise . values of  prob &gt; f  less than 0.05 indicate that model terms are significant . values greater than 0.1 indicate that model terms are not significant . in the present work , linear terms and all the square effects of a , b , c , d , and e and the coefficient of determination ( r ) for xylitol production was calculated as 0.9148 , which is very close to 1 and can explain up to 91.00% variability of the response . the predicted r value of 0.6867 was in reasonable agreement with the adjusted r value of 0.8561 . the adequate precision value of 12.951 indicates an adequate signal and suggests that the model can navigate the design space . in both designs the interaction effects of variables on xylitol production were studied by plotting 3d surface curves against any two independent variables , while keeping another variable at its central ( 0 ) level . the 3d curves of the calculated response ( xylitol yield ) and contour plots from the interactions between the variables were obtained .   this evidence from above figures shows the dependency of xylose , mgso47h2o , yeast extract on xylitol production . the optimal operation conditions of peptone , xylose , mgso47h2o , and yeast extract for maximum xylitol production were determined by response surface analysis and also estimated by regression equation . the predicted values from the regression equation closely agreed with that obtained from experimental values . the xylitol production increased with increase in temperature to about 36c , and thereafter xylitol production decreased with further increase in temperature . this evidance from above figures shows the dependency of ph , substrate concentration , agitation speed , and inoculum size on xylitol production . the optimal operation conditions of temperature , substrate concentration , ph , agitation speed , and inoculum size for maximum xylitol production were determined by response surface analysis and also estimated by regression equation . the predicted values from the regression equation closely agreed with that obtained from experimental values . validation of the experimental model was tested by carrying out the batch experiment under optimal operation conditions which are ( g / l ) : peptone : 6.03 , xylose : 10.62 , mgso47h2o : 1.39 , yeast extract : 4.66 established by the regression model . under optimal process variables levels are temperature ( 36.56c ) , ph ( 7.27 ) , substrate concentration ( 3.55  g / l ) , inoculum size ( 3.69  ml ) , and agitation speed ( 194.44  rpm ) . the xylitol production ( 0.80  g / g ) obtained from experiments was very close to the actual response ( 0.78  g / g ) predicted by the regression model , which proved the validity of the model . in this work , plackett - burman design was used to test the relative importance of medium components on xylitol production . among the variables , peptone , xylose , mgso47h2o , and yeast g / l ) : peptone : 6.03 , xylose : 10.62 , mgso47h2o : 1.39 , and yeast extract : 4.66 . then the influence of various process variables , namely , temperature , ph , substrate concentration , agitation speed , and inoculum size on the xylitol production was evaluated by ccd . the optimum levels of process variables are temperature ( 36.56c ) , ph ( 7.27 ) , substrate concentration ( 3.55  g / l ) , inoculum size ( 3.69  ml ) , and agitation speed ( 194.44  rpm ) . this study showed that the corncob is a good source for the production of xylitol . using the optimized conditions , the xylitol yield reaches 0.80  g / g .</td>\n",
       "      <td>&lt;S&gt; optimization of the culture medium and process variables for xylitol production using corncob hemicellulose hydrolysate by pachysolen tannophilus ( mttc 1077 ) was performed with statistical methodology based on experimental designs . &lt;/S&gt; &lt;S&gt; the screening of nine nutrients for their influence on xylitol production was achieved using a plackett - burman design . &lt;/S&gt; &lt;S&gt; peptone , xylose , mgso47h2o , and yeast extract were selected based on their positive influence on xylitol production . &lt;/S&gt; &lt;S&gt; the selected components were optimized with box - behnken design using response surface methodology ( rsm ) . the optimum levels ( &lt;/S&gt; &lt;S&gt; g / l ) were peptone : 6.03 , xylose : 10.62 , mgso47h2o : 1.39 , yeast extract : 4.66 . &lt;/S&gt; &lt;S&gt; the influence of various process variables on the xylitol production was evaluated . &lt;/S&gt; &lt;S&gt; the optimal levels of these variables were quantified by the central composite design using rsm , for establishment of a significant mathematical model with a coefficient determination of r2 = 0.91 . &lt;/S&gt; &lt;S&gt; the validation experimental was consistent with the prediction model . &lt;/S&gt; &lt;S&gt; the optimum levels of process variables were temperature ( 36.56c ) , ph ( 7.27 ) , substrate concentration ( 3.55  &lt;/S&gt; &lt;S&gt; g / l ) , inoculum size ( 3.69  ml ) , and agitation speed ( 194.44  rpm ) . &lt;/S&gt; &lt;S&gt; these conditions were validated experimentally which revealed an enhanced xylitol yield of 0.80  g / g . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>conversion disorder has one or more symptoms that affect voluntary motor or sensory function suggesting a neurological or other medical condition , but they are inconsistent with known neurological or musculoskeletal pathologies . instead , the symptoms are due to an unconscious expression of a psychological conflict or need . the symptoms are often reinforced by social support from family and friends or by avoiding underlying emotional stress . the symptoms of patients with conversion disorder can be debilitating and include paralysis of one or more limbs , ataxia , tremors , tics , and dystonia . many other names are used to describe this disorder are functional gait disorder , hysterical paralysis , psychosomatic disorder , conversion reaction , and chronic neurosis . the disorder is more common in adolescence than in childhood . despite conversion disorders long - documented history , it is often confused with other psychological disorders conversion disorder , remain diagnostic challenges for the clinicians . a 17-year - old female coming from mses who was premorbidly maintaining well came with complaints of asymmetrical repetitive flickering like movement of the right hand which started on the day of her 12 grade board exams . she was observed to have reduced sleep since 1 week before her exams and had relatively less communication with family members . on the day of her exams , by the time she got the question paper her whole of her right arm started having repetitive flickering movement vigorously , and she had to support her right arm with the left to write the exam and had come out of exam hall without completing the exam . within a few days , the abnormal movements had progressed to her right leg . informant said that she use to have crying spells and appear sad most of the time as she had not given the exams . she was treated with promethazine and trihexyphenidyl neuroimaging was done which was found to be normal . she showed some improvement after 20 days , but she was not completely resolved , on the day before the day of admission , she developed shivering over her whole body and was admitted to the intensive care unit . it was not associated with loss of consciousness , no urine or fecal incontinence , no frothing from the mouth , no tongue biting , and no up rolling of eyeball . electroencephalogram computed tomography and magnetic resonance imaging brain were done and were found to be normal . , it was found that she was an above average student in her class and that her family had too much expectation from her . she also said that her younger sister was always given more attention by her mother . her episodes were provoked when asked to write or hold a pen with her right hand , also when she was asked to walk without assistance . she was also observed to flex her right toe while walking and during stay in the hospital she was observed to be having a sudden onset of asymmetrical repetitive jerky movements of bilateral legs . routine hemogram , renal function test , liver function test blood sugar , lipid profile , and thyroid function were found to be normal . the patient was prescribed diazepam 4 mg per days and after 2 days it was increased to 6 mg per day , she showed gradual improvement . on the initial days of sessions , her symptoms got aggravated during the sessions and session had to be stopped in between . after few attempts , the patient had ventilated to us how her mother gives less importance to her when compared to her younger sister who is 6 years younger to her . the patient had also said that from her toddler stage till 10 standard she was living with her paternal grandmother and father , and now she moved to a different house along with her parents . her parents were also included in the sessions and her issues with her mother were discussed . the patient had gradually started walking without difficulty and frequency of abnormal movements had reduced . diazepam was tapered and stopped within a week and had been stable at the time of discharge . patient came for a follow - up after 2 weeks , and she had been maintaining well . the clinical picture is indicative of dissociative motor disorder f44.4 according to icd 10 . after taking a detailed history , it was clear that her parents were giving her more pressure to attain high marks in board exam . during interview whenever a patient was asked to hold a pen or to write her symptoms increased . patient was started on therapy sessions as well as low - dose diazepam . in the sessions , possible causes of these symptoms were discussed , and she was encouraged to hold the pen and write . her parents were also psychoeducated about the psychosomatic nature of the symptoms and advised to encourage her for a symptom - free lifestyle . they were also given an instruction not to pay attention to her complaints of physical nature . , patient was explained to her that if there is any problem in her nerves , it will be recorded in that study , along with that dose of diazepam was also increased to 6 from 4 mg . after the study , patient showed a dramatic improvement in her symptoms . failure of treatment in dissociative disorders occurs mostly when we can not identify the primary stressor or gain . by taking proper history and early identification of the stressor there are case reports where dissociative disorders are managed alone with therapy sessions , and the patient is taught to how to deal with stressful situations . nerve conduction study is a diagnostic test to evaluate the function , i.e. the electrical conduction of motor or sensory nerve of the human body . it can also be done along with needle electromyography to measure both nerve and muscle function . in this , the study is performed by electrical stimulation of a peripheral nerve and recording of a muscle supplied by this nerve . the time taken for electrical impulses to travel from the stimulation to the recording site is measured . in this patient , this procedure was well explained to the patient , and she showed marked improvement . conversion disorder , somatoform disorder , and malingering always remain a diagnostic challenges for the clinicians . the prompt history taking , identification of stressors , use of appropriate and validated physical examination manoeuvres , and coordination of care and information exchange between all family members and medical team may facilitate the expeditious care of these patients in a cost - effective manner . the existing literature supports a multidisciplinary treatment approach , with specific interventions , such as cognitive behavior therapy for cognitive restructuring and psychodynamic therapy for addressing symptom connections to trauma and dissociation .</td>\n",
       "      <td>&lt;S&gt; conversion disorders are more prevalent in childhood and adolescence , especially in females . &lt;/S&gt; &lt;S&gt; they are usually associated with stressors and symptoms usually reflect a means to avoid the stressor , or also with a primary and secondary gain . this case report involves a similar situation where a young girl was treated successfully with diazepam , therapeutic nerve conduction study , and behavioral psychotherapy . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>diabetic neuropathy is a leading complication of diabetes mellitus , resulting in significant morbidity and mortality . although its exact pathogenesis is not fully understood , hyperglycemia does not appear to be the sole factor in the development of neuropathy in diabetic patients . enigmatically , recent reports have described that long - term tight glycemic control may be a major risk factor for the development of diabetic neuropathy [ 1 , 2 ] . neuropathy secondary to rapid normalization of chronic hyperglycemia in the setting of poorly controlled diabetes is also emerging as a new disease entity classified as an iatrogenic complication . symptoms in these patients are typically consistent with a distal sensory polyneuropathy which is appearing shortly after the initiation of intensive glycemic control and is referred to as  insulin neuritis  or treatment - induced neuropathy and is characterized by acute , severe pain . . however , the parallel worsening of neuropathy and retinopathy from a rapid tightening of glycemic control [ 4 , 5 ] suggests a common underlying pathophysiology . hypoglycemia , a potentially devastating neuronal insult , is usually the result of attempting tight control of blood glucose levels with insulin or other hypoglycemic agents [ 1 , 6 ] . currently , the only available method for preventing this hypoglycemia - induced neuronal injury in the clinical setting is the delivery of glucose , a treatment that paradoxically may exacerbate the insult . the objective of this present research was to study the molecular mechanisms of acute neuropathic pain induced by insulin and hypoglycemia in an animal model . the expression of c - fos protooncogene , a marker of nociceptive - induced neuronal activity in the spinal cord [ 7 , 8 ] , was also determined . additionally , the preventive effects of pretreatment with coenzyme q10 ( coq10 ) on hypoglycemia - induced neuropathic pain and stress - sensitive factor expression were explored . all experiments were carried out following the guidelines and protocols of the animal care and use committee of the university of miami , and the protocol was approved by the iacuc committee . c57bl/6j mice served as controls and cba / caj mice , which develop diabetes spontaneously , functioned as the treatment group ; they were both obtained from jackson laboratory ( bar harbor , maine , usa ) . all mice were approximately 12 to 14 weeks old which is comparable to young adult in humans . while cba / caj mice spontaneously develop mild hyperglycemia , these mice had not yet developed peripheral neuropathy at the commencement of the study , as assessed by mechanical testing . mice were housed in groups of five in plastic cages with soft bedding and free access to food and water under a 12  h/12  h light - dark cycle ( dark cycle : 7:00 pm7:00 am ) . blood from animals for glucose measurement was obtained via a tail tip snip . during collection , the initial blood expressed was discarded and a subsequent sample was analyzed with onetouch glucometer . to examine the effects of acute insulin - induced hypoglycemia on mechanical sensitivity , 1  unit / kg of insulin ( novolin , novo nordisk , 2880 bagsvrd , denmark ) was injected intraperitoneally in the treatment group , while control animals received equal volumes of normal saline . blood glucose levels and mechanical sensitivity were tested before injection and periodically throughout the study until blood glucose levels recovered to normal . to determine whether insulin itself or insulin - induced hypoglycemia was the cause of mechanical hypersensitivity , blood glucose levels were  clamped  in the normal range by the combined administration of insulin ( 1  unit / kg ) and glucose ( 3.2  g / kg ) in an intraperitoneal injection . the primary reason for not utilizing an intravenous infusion was the fact that the mechanical sensitivity measurement is an unrestricted behavior test and the presence of an intravenous access was felt to interfere with measurements . louis , mo , usa ) was dissolved in olive oil ( sigma - aldrich ) at a concentration of 30  mg / ml dosed at 100  mg / kg . this dose represents the human equivalent doses of 8  mg / kg , based on body surface area . twice at a volume of 100  l/30  g of body weight before 20  hr and 4  hr of the induction of hypoglycemia . the mechanical allodynia test was conducted with a touch - test sensory evaluator ( von frey filaments , north coast medical , inc . , the mouse was placed on a wire mesh platform and was covered with a transparent glass container and a period of 30 minutes was allowed for habituation . the observation of a positive response ( paw lifting ,  shaking ,  or licking ) within five seconds of the application of the filament was then followed by the application of a thinner filament ( or a thicker one if the response was negative ) . the paw withdrawal threshold was measured five times and was expressed as the tolerance level in grams . normal saline - injected control mice , mice with hypoglycemia induced by insulin , and hypoglycemic mice pretreated with coq10 were sacrificed via an overdose of nembutal and were then decapitated . part of the samples was fixed in 4% paraformaldehyde in phosphate buffered saline ( ph 7.4 ) overnight , cryoprotected in 0.1  m phosphate buffered saline containing 20% sucrose , and sectioned by cryostat into 15  m thick sections . sections were incubated overnight at 4c with the primary antibody , anti - c - fos ( sigma - aldrich , usa ) , followed by biotinylated secondary antibody ( vector lab , usa ) for one hr at 22c . to ensure the specificity of primary antibody , the primary antibody was replaced by the diluent of the antibody in one section in each set of stains so as to exclude nonspecific background staining . positive c - fos cells were counted in laminar i - ii area of 280  m of dorsal horn of lumbar spinal cord transverse section ( the laminar i - ii area is shown by the dotted line in figure 4 ) . the other half of the collected samples were fresh frozen in dry ice and stored at 80c . the levels of mrna of c - fos were evaluated by rt - pcr in the drg and spinal cord tissues . extraction of total rna was carried out with trizol ( invitrogen , grand island , ny , usa ) according to the manufacturer 's instructions . 1  g of rna was reverse transcribed with 200  u / sample superscript ii ( invitrogen ) and 250  ng / reaction of random primers ( promega , san luis obispo , ca , usa ) . the genes of c - fos were amplified from 0.1  g aliquots of cdna in a standard pcr buffer ( 50  mm kcl , 1.5  mm mgcl2 , and 10  mm tris - hcl , ph 8.3 ) containing 10  pmol of forward and reverse primers along with 0.5  u / sample of amplitaq dna polymerase ( applied biosystems , grand island , ny , usa ) . the sequences of primer pairs are the following : -actin forward : ctagacttcgagcaggagatg , reverse : caagaaggaaggctggaaaag , the product is 150  bp ; c - fos forward : ccagtcaagagcatcagcaa , reverse : aagtagtgcagcccggagta , the product is 247  bp . data are presented as mean  sem and analyzed using prism 4 software ( graphpad software inc . , san diego , ca ) . the behavior test data was analyzed with two - way analysis of variance with two repeated factors followed by tukey 's multiple comparison test . comparison between two groups was assessed by unpaired , two - tailed student 's t - test . compared to control animals , it appeared that decreased blood glucose levels correlated to increased pain in the insulin treatment group . both strains demonstrated significant differences in mechanical sensitivity 40 , 90 , and 150  min after insulin injection ( p &lt; 0.05 and p &lt; 0.001 ) .   figure 1 shows that decreased withdrawal thresholds ( mechanical hypersensitivity ) were associated with insulin - induced acute hypoglycemia in both strains of mice . a group of normal saline - injected mice served as a control and demonstrated no changes in blood glucose levels or mechanical sensitivity , indicating that handling and injection stress did not affect or confound results . to determine whether insulin alone induces hypersensitivity , blood glucose levels were clamped at normal levels by joint insulin and glucose injection .   table 1 demonstrated the blood glucose levels of two strains of mice in different situation : saline , insulin , or insulin combined with glucose . in the linked administration of insulin and glucose , blood glucose levels remained at an average of 123.33  8.55 and 165.93  10.60  mg / dl for the c57b/6j and cba / caj mice , respectively , and these mice subsequently demonstrated no significant change in hindpaw withdrawal thresholds .   figure 2 indicates that mechanical hypersensitivity did not develop when blood glucose levels remained in normal range after insulin was injected , suggesting that insulin itself is not involved in the hypoglycemia - induced mechanical hypersensitivity . coq10 has a critical role in producing energy and antioxidant protection for the body . for the scenario of insulin - induced hypoglycemia , we evaluated whether coq10 could play a protective role in the peripheral nerves .   figure 3 indicates that coq10 did not affect the blood glucose level decrease following insulin injection ; however , pretreatment with coq10 did prevent the development of mechanical hypersensitivity in insulin - induced hypoglycemic mice . levels of c - fos mrna and c - fos immunoreactivity within the spinal cord were evaluated in insulin - induced hypoglycemic mice . figure 4 shows that c - fos positive cells in the dorsal horn of the lumbar spinal cord after insulin injection increased significantly ( in cell - counted analysis , positive cells in the insulin - injected group were more numerous than those in the saline - injected group , p &lt; 0.01 ; in rt - pcr analysis , mrna level of c - fos in insulin - injected group is almost two times that in saline - injected group , p &lt; 0.001 ; student 's t - test ) . however , pretreatment with coq10 partially decreased c - fos expression in the spinal cord ( in rt - pcr analysis , c - fos mrna levels in the group pretreated with coq10 were significantly lower than those in the insulin - injected group , p &lt; 0.05 ) . studies have suggested that hypoglycemia - induced neuropathy may not simply be the result of glucose deprivation but rather a result of a multifactorial process involving oxidative stress and stress - sensitive factors . the results of the present study demonstrate that insulin - induced hypoglycemia may result in acute neuropathic pain and the increased mechanical sensitivity noted is the result of decreased glycemic levels rather than insulin itself . the immunohistological and rt - pcr results suggest that insulin - induced hypoglycemia results in an increased expression of the stress - sensitive and pain - related factor c - fos in nerve tissues . this in turn may be the mechanism by which acute pain is induced in the body . furthermore , our results demonstrated that pretreatment with coq10 can prevent hypoglycemia - induced mechanical hypersensitivity and decrease the expression of c - fos . results further suggest that the protective effects of coq10 on pain sensitivity may be related to a decrease in activation of spinal pathways mediated by the inhibition of oxidative stress and intracellular signaling , preventing neuronal injury . patients with diabetes may face the difficult situation where tight blood glucose control can reduce the risk of diabetic complications ; however , this degree of control may also increase the risk of dangerous hypoglycemic episodes . studies estimate 30% of diabetics experience serious hypoglycemic episodes annually   and hypoglycemia has potentially devastating effects on nervous tissues . clinicians have described acute severe painful neuropathy occurring during intensive treatment of patients with type 1 and type 2 diabetes treated with oral hypoglycemic agents or with insulin [ 1 , 11 ] . in 1933 , caravati described neuropathic pain resulting from insulin use ,  insulin neuritis  ; however , the mechanism remains unclear . trophic factors and cytokines , including vascular endothelial growth factor ( vegf ) , insulin growth factor ( igf ) , mitogenic cytokine , il-8 , il-6 , and tnf- , have been implicated in the pathogenesis of diabetic retinopathy , diabetic nephropathy , and diabetic neuropathy . it is hypothesized that upregulation of these trophic factors and cytokines is associated with intensive glycemic control and is responsible for the early worsening of retinopathy and acute pain . our data suggests that c - fos , an immediate early transcription factor , is involved in insulin - induced hypersensitivity . elevated cytokine levels , including interleukin-1 , interleukin-6 , and tumor necrosis factor- , have been associated with impaired autonomic function after experimental hypoglycemia . thus , acute treatment of diabetes - induced neuropathy and retinopathy notably after intensive glycemic control may have a common pathophysiological mechanism that involves upregulation of proinflammatory cytokines . this concept also suggests an additional hypoglycemia - related pathophysiological mechanism and provides potential targets for therapeutic intervention . our data demonstrated that when combined , glucose and insulin injections , without subsequent hypoglycemic episodes , do not result in acute painful neuropathy , suggesting that insulin itself does not induce hypoglycemia - induced mechanical hypersensitivity . thus , acute painful neuropathy is a concern not only for diabetics but also for normal subjects experiencing sudden hypoglycemic episodes . tight glucose control has been associated with numerous clinical benefits in diabetic patients , including the reduction of diabetic neuropathy ; however , this type of treatment significantly increases the risk of severe hypoglycemic episodes . as we have demonstrated , hypoglycemia itself may exacerbate neuropathy and currently the only available method for preventing this hypoglycemia - induced neuronal injury in the clinical setting is the delivery of glucose , a treatment that paradoxically may exacerbate the insult . this study has obvious limitations ; most notably , it was conducted solely in mice . it can be difficult to extrapolate data from lower mammals to humans ; pain has many complex elements that can be difficult to assess . autophagy occurs in hypoglycemic peripheral nerves in association with axonal degeneration and regeneration in rats models . hypoglycemia causes wallerian - type axonal degeneration of large myelinated nerve fibers in the peripheral nerve of insulin - treated diabetic animal models [ 19 , 20 ] . neuronal death resulting from hypoglycemia involves excitotoxicity and dna damage . by using cortical neuron cultures , researchers have found that application of poly(adp - ribose ) polymerase ( parp-1 ) , an endogenous caspase-3 substrate inhibitor , increases neuronal survival in glucose deprivation . additionally , rat models of insulin - induced hypoglycemia have shown the therapeutic potential of papd-1 inhibitors . other researches have demonstrated that coq10 inhibits high glucose - induced cleavage of papd-1   and suggest that coq10 prevents oxidative stress - induced apoptosis through inhibition of the mitochondria - dependent caspase-3 pathway . taken together , our present results indicate that pretreatment with coq10 can prevent hypoglycemia - induced mechanical hypersensitivity and decrease the expression of c - fos and chronic treatment with coq10 may scavenge free radicals instantly and prevent mitochondrial dysfunction in the transient hypoglycemia induced by tight glucose control in diabetics .</td>\n",
       "      <td>&lt;S&gt; diabetic neuropathic pain is reduced with tight glycemic control . &lt;/S&gt; &lt;S&gt; however , strict control increases the risk of hypoglycemic episodes , which are themselves linked to painful neuropathy . &lt;/S&gt; &lt;S&gt; this study explored the effects of hypoglycemia - related painful neuropathy . &lt;/S&gt; &lt;S&gt; pretreatment with coenzyme q10 ( coq10 ) was performed to explore the preventive effect of coq10 on hypoglycemia - related acute neuropathic pain . &lt;/S&gt; &lt;S&gt; two strains of mice were used and 1  unit / kg of insulin was given to induce hypoglycemia . &lt;/S&gt; &lt;S&gt; mechanical sensitivity of hindpaw withdrawal thresholds was measured using von frey filaments . &lt;/S&gt; &lt;S&gt; blood glucose levels were clamped at normal levels by joint insulin and glucose injection to test whether insulin itself induced hypersensitivity . &lt;/S&gt; &lt;S&gt; results suggest that the increased mechanical sensitivity after insulin injection is related to decreased blood glucose levels . &lt;/S&gt; &lt;S&gt; when blood glucose levels remained at a normal level by the linked administration of insulin and glucose , mice demonstrated no significant change in mechanical sensitivity . &lt;/S&gt; &lt;S&gt; pretreatment with coq10 prevented neuropathic pain and the expression of the stress factor c - fos . &lt;/S&gt; &lt;S&gt; these results support the concept that pain in the diabetic scenario can be the result of hypoglycemia and not insulin itself . additionally , pretreatment with coq10 &lt;/S&gt; &lt;S&gt; may be a potent preventive method for the development of neuropathic pain . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>urinary stones are rarely seen in the urethra and are usually encountered in men with urethral stricture or diverticulum . urethral diverticula can present in many ways , including recurrent urinary tract infections ( utis ) , dysuria , increased urinary frequency , urgency , and hematuria . in addition to causing the symptoms above , diverticula also can be complicated with stones or malignancy , both of which can lead to bladder outlet or urethral obstruction . stone formation has been reported to occur in 1% to 10% of patients with urethral diverticula . urinary stasis and chronic infection have been identified as the causes of calculi formation within urethral diverticula . the case of a woman with a giant calculus in a urethral diverticulum is reported . a 62-year - old woman was first seen in the urology clinic complaining of a 1-year history of constant lower abdominal pain , dysuria , and dyspareunia . she also noted an increase in the volume of the vaginal wall that coincided with an increase of pain in this area . written informed consent was obtained from the patient for the publication of this case report and any accompanying images . the general patient examination was normal , but a focused genital examination revealed a large mass of approximately 4  cm near the urethra . when compressing the urethra , leakage of purulent discharge from the meatal orifice was noted . a kidney , ureter , bladder x - ray showed a giant calculi ( figure 1 ) , and cystoscopy revealed an extrusion of the posterolateral distal urethra . on admission , vital signs were all normal and laboratory tests demonstrated microscopic pyuria ( 2030/high power field ) . the diverticulum was punctured by electrocautery and dissected with periurethral tissue , which allowed total removal of the calculi ( figure 2 ) . a tagged 3 - 0 with silk suture the foley catheter was kept in place for 7 days . oral antibiotic therapy with ciprofloxacin the estimated prevalence of urethral diverticula in adult women is between 0.6% and 6% , and associated stone formation is reported in 1.5% to 10% of cases . the cause of diverticula remains largely unknown and ranges from congenital to traumatic ( instrumentation , childbirth ) to infectious causes . the formation of abscesses and these may rupture into the urethral lumen , forming the diverticula . the quality of life of patients who have a diverticulum ( especially with calculi ) may be significantly disturbed because of complications such as dysuria , dyspareunia , uti , and postvoid dribbling . any patient with lower urinary tract symptoms that have proved to be unresponsive to traditional treatment should be suspected of having a urethral diverticulum . in patients with urethral diverticula other ways to confirm a diagnosis of urethral diverticula are voiding cystourethrogram , intravenous pyelography , and ultrasonography . presumably , a stone should also be visualized within the diverticula by one of these diagnostic modalities . the issues that remain focus on determining symptomatic relief by conservative therapy , assessing satisfactory long - term treatment of diverticulum , and determining the possible benefit from surgical excision . however , the confirmation of number , site , and size of the diverticulum is important before operation to prevent complications such as urethral stricture , urethro - vaginal fistula and incontinence due to injury of sphincter . diagnosis of a complicated diverticulum can be easily achieved if one possesses a high degree of clinical suspicion . thus , this diagnosis should be considered in the case of recurrent utis , hematuria , and dysuria , as well as in patients with masses felt on pelvic examination . surgical approach with litholapaxy followed by diverticulectomy</td>\n",
       "      <td>&lt;S&gt; abstracturethral diverticula with calculi have a low incidence as reported in the literature . &lt;/S&gt; &lt;S&gt; diverticulum of female urethra is rare , often discovered due to associated complications . &lt;/S&gt; &lt;S&gt; we report a case of diverticulum of the female urethra containing giant calculi in a 62-year - old multiparous woman . &lt;/S&gt; &lt;S&gt; she consulted with our office due to dysuria and a hard , painful periurethral mass in the anterior vagina wall . &lt;/S&gt; &lt;S&gt; the diverticulum was approached surgically by a vaginal route , and local extraction of the calculi and subsequent diverticulectomy successfully treated the condition.diagnosis of a complicated diverticulum can be easily achieved if one possesses a high degree of clinical symptoms . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "show_random_elements(raw_datasets[\"train\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lnjDIuQ3IrI-"
   },
   "source": [
    "The metric is an instance of [`datasets.Metric`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Metric):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "id": "5o4rUteaIrI_"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Metric(name: \"rouge\", features: {'predictions': Value(dtype='string', id='sequence'), 'references': Value(dtype='string', id='sequence')}, usage: \"\"\"\n",
       "Calculates average rouge scores for a list of hypotheses and references\n",
       "Args:\n",
       "    predictions: list of predictions to score. Each predictions\n",
       "        should be a string with tokens separated by spaces.\n",
       "    references: list of reference for each prediction. Each\n",
       "        reference should be a string with tokens separated by spaces.\n",
       "    rouge_types: A list of rouge types to calculate.\n",
       "        Valid names:\n",
       "        `\"rouge{n}\"` (e.g. `\"rouge1\"`, `\"rouge2\"`) where: {n} is the n-gram based scoring,\n",
       "        `\"rougeL\"`: Longest common subsequence based scoring.\n",
       "        `\"rougeLSum\"`: rougeLsum splits text using `\"\n",
       "\"`.\n",
       "        See details in https://github.com/huggingface/datasets/issues/617\n",
       "    use_stemmer: Bool indicating whether Porter stemmer should be used to strip word suffixes.\n",
       "    use_agregator: Return aggregates if this is set to True\n",
       "Returns:\n",
       "    rouge1: rouge_1 (precision, recall, f1),\n",
       "    rouge2: rouge_2 (precision, recall, f1),\n",
       "    rougeL: rouge_l (precision, recall, f1),\n",
       "    rougeLsum: rouge_lsum (precision, recall, f1)\n",
       "Examples:\n",
       "\n",
       "    >>> rouge = datasets.load_metric('rouge')\n",
       "    >>> predictions = [\"hello there\", \"general kenobi\"]\n",
       "    >>> references = [\"hello there\", \"general kenobi\"]\n",
       "    >>> results = rouge.compute(predictions=predictions, references=references)\n",
       "    >>> print(list(results.keys()))\n",
       "    ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']\n",
       "    >>> print(results[\"rouge1\"])\n",
       "    AggregateScore(low=Score(precision=1.0, recall=1.0, fmeasure=1.0), mid=Score(precision=1.0, recall=1.0, fmeasure=1.0), high=Score(precision=1.0, recall=1.0, fmeasure=1.0))\n",
       "    >>> print(results[\"rouge1\"].mid.fmeasure)\n",
       "    1.0\n",
       "\"\"\", stored examples: 0)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "metric"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "jAWdqcUBIrJC"
   },
   "source": [
    "You can call its `compute` method with your predictions and labels, which need to be list of decoded strings:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "id": "6XN1Rq0aIrJC"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'rouge1': AggregateScore(low=Score(precision=1.0, recall=1.0, fmeasure=1.0), mid=Score(precision=1.0, recall=1.0, fmeasure=1.0), high=Score(precision=1.0, recall=1.0, fmeasure=1.0)),\n",
       " 'rouge2': AggregateScore(low=Score(precision=1.0, recall=1.0, fmeasure=1.0), mid=Score(precision=1.0, recall=1.0, fmeasure=1.0), high=Score(precision=1.0, recall=1.0, fmeasure=1.0)),\n",
       " 'rougeL': AggregateScore(low=Score(precision=1.0, recall=1.0, fmeasure=1.0), mid=Score(precision=1.0, recall=1.0, fmeasure=1.0), high=Score(precision=1.0, recall=1.0, fmeasure=1.0)),\n",
       " 'rougeLsum': AggregateScore(low=Score(precision=1.0, recall=1.0, fmeasure=1.0), mid=Score(precision=1.0, recall=1.0, fmeasure=1.0), high=Score(precision=1.0, recall=1.0, fmeasure=1.0))}"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fake_preds = [\"hello there\", \"general kenobi\"]\n",
    "fake_labels = [\"hello there\", \"general kenobi\"]\n",
    "metric.compute(predictions=fake_preds, references=fake_labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "n9qywopnIrJH"
   },
   "source": [
    "## Preprocessing the data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "YVx71GdAIrJH"
   },
   "source": [
    "Before we can feed those texts to our model, we need to preprocess them. This is done by a 🤗 `Transformers` `Tokenizer` which will (as the name indicates) tokenize the inputs (including converting the tokens to their corresponding IDs in the pretrained vocabulary) and put it in a format the model expects, as well as generate the other inputs that the model requires.\n",
    "\n",
    "To do all of this, we instantiate our tokenizer with the `AutoTokenizer.from_pretrained` method, which will ensure:\n",
    "\n",
    "- we get a tokenizer that corresponds to the model architecture we want to use,\n",
    "- we download the vocabulary used when pretraining this specific checkpoint.\n",
    "\n",
    "That vocabulary will be cached, so it's not downloaded again the next time we run the cell.\n",
    "\n",
    "To tokenize the inputs for this particular model, we need to have `sentencepiece` installed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "roKGCrrSJgsF"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: sentencepiece in ./miniconda3/envs/fastai/lib/python3.8/site-packages (0.1.96)\n"
     ]
    }
   ],
   "source": [
    "! pip install sentencepiece"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NqQnTtd2JhZW"
   },
   "source": [
    "Now we can instantiate our tokenizer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "eXNLu_-nIrJI"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "879b2803d648435d8e9f04952b125699",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/88.0 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bbbc3e476203464a88aa05bc683888d0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/1.09k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "adb699cae3ec46b583b87714ed4b2469",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/1.82M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3a6712e59fc94ca38aff51e4407d9998",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/65.0 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from transformers import AutoTokenizer\n",
    "    \n",
    "tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Vl6IidfdIrJK"
   },
   "source": [
    "By default, the call above will use one of the fast tokenizers (backed by Rust) from the 🤗 `Tokenizers` library."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "rowT4iCLIrJK"
   },
   "source": [
    "You can directly call this tokenizer on one sentence or a pair of sentences:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "id": "a5hBlsrHIrJL"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'input_ids': [8087, 108, 136, 156, 5577, 147, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tokenizer(\"Hello, this one sentence!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "qo_0B1M2IrJM"
   },
   "source": [
    "Depending on the model you selected, you will see different keys in the dictionary returned by the cell above. They don't matter much for what we're doing here (just know they are required by the model we will instantiate later), you can learn more about them in [this tutorial](https://huggingface.co/transformers/preprocessing.html) if you're interested.\n",
    "\n",
    "Instead of one sentence, we can pass along a list of sentences:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "id": "b2EnCbkBir66"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'input_ids': [[8087, 108, 136, 156, 5577, 147, 1], [182, 117, 372, 5577, 107, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tokenizer([\"Hello, this one sentence!\", \"This is another sentence.\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "mXYe5s3Jir67"
   },
   "source": [
    "To prepare the targets for our model, we need to tokenize them inside the `as_target_tokenizer` context manager. This will make sure the tokenizer uses the special tokens corresponding to the targets:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "id": "m4JRmJxFir67"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'input_ids': [[8087, 108, 136, 156, 5577, 147, 1], [182, 117, 372, 5577, 107, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}\n"
     ]
    }
   ],
   "source": [
    "with tokenizer.as_target_tokenizer():\n",
    "    print(tokenizer([\"Hello, this one sentence!\", \"This is another sentence.\"]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2C0hcmp9IrJQ"
   },
   "source": [
    "If you are using one of the five T5 checkpoints we have to prefix the inputs with \"summarize:\" (the model can also translate and it needs the prefix to know which task it has to perform)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "id": "DAQBW23yir67"
   },
   "outputs": [],
   "source": [
    "if model_checkpoint in [\"t5-small\", \"t5-base\", \"t5-larg\", \"t5-3b\", \"t5-11b\"]:\n",
    "    prefix = \"summarize: \"\n",
    "else:\n",
    "    prefix = \"\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_hB_Pubqir68"
   },
   "source": [
    "We can then write the function that will preprocess our samples. We just feed them to the `tokenizer` with the argument `truncation=True`. This will ensure that an input longer that what the model selected can handle will be truncated to the maximum length accepted by the model. The padding will be dealt with later on (in a data collator) so we pad examples to the longest length in the batch and not the whole dataset.\n",
    "\n",
    "The max input length of `google/pegasus-cnn_dailymail` is 1024, so `max_input_length = 1024`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "vc0BSBLIIrJQ"
   },
   "outputs": [],
   "source": [
    "max_input_length = 1024\n",
    "max_target_length = 256\n",
    "\n",
    "def preprocess_function(examples):\n",
    "    inputs = [prefix + doc for doc in examples[\"article\"]]\n",
    "    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)\n",
    "\n",
    "    # Setup the tokenizer for targets\n",
    "    with tokenizer.as_target_tokenizer():\n",
    "        labels = tokenizer(examples[\"abstract\"], max_length=max_target_length, truncation=True)\n",
    "\n",
    "    model_inputs[\"labels\"] = labels[\"input_ids\"]\n",
    "    return model_inputs"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0lm8ozrJIrJR"
   },
   "source": [
    "This function works with one or several examples. In the case of several examples, the tokenizer will return a list of lists for each key:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "id": "-b70jh26IrJS"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'input_ids': [[126, 4403, 115, 154, 197, 4567, 113, 1044, 111, 218, 1111, 8895, 115, 878, 1020, 113, 15791, 110, 108, 704, 115, 1044, 12857, 16020, 111, 191, 490, 7755, 2495, 107, 740, 32680, 117, 3365, 130, 142, 14069, 22021, 476, 113, 58117, 143, 110, 55654, 110, 158, 143, 110, 55654, 110, 105, 665, 3957, 943, 110, 20815, 110, 158, 111, 218, 6860, 130, 114, 711, 113, 109, 5910, 1568, 110, 108, 11300, 110, 108, 2111, 5173, 110, 108, 16020, 110, 108, 132, 7755, 2495, 110, 107, 8823, 1683, 2298, 120, 5690, 111, 49159, 233, 2881, 562, 244, 7755, 2495, 110, 108, 704, 115, 693, 111, 3464, 15791, 110, 108, 218, 129, 12409, 141, 32680, 107, 6304, 32680, 432, 64142, 2775, 253, 130, 8466, 110, 108, 10353, 110, 108, 111, 35368, 1379, 28247, 110, 108, 111, 2297, 218, 133, 114, 2404, 1298, 124, 348, 113, 271, 143, 15593, 6045, 110, 158, 111, 637, 1932, 115, 1044, 122, 1695, 110, 107, 2297, 110, 108, 112, 927, 1312, 7233, 110, 108, 15593, 6045, 110, 108, 111, 32261, 115, 1044, 122, 1695, 110, 108, 126, 192, 129, 3048, 112, 248, 114, 10774, 1014, 115, 5987, 8149, 170, 217, 791, 118, 1695, 233, 1589, 32680, 143, 9073, 304, 110, 158, 111, 319, 5235, 603, 110, 107, 1458, 49334, 117, 142, 957, 230, 112, 2555, 29599, 110, 55654, 373, 114, 613, 908, 110, 108, 155, 109, 1298, 117, 110, 108, 6808, 110, 108, 4274, 111, 137, 1007, 1651, 9361, 3198, 111, 1562, 12439, 110, 107, 115, 24374, 2827, 6316, 115, 1044, 122, 9073, 304, 110, 108, 110, 8723, 34764, 20547, 34261, 233, 13568, 3073, 143, 110, 24397, 116, 110, 158, 1788, 1225, 3445, 115, 110, 55654, 476, 110, 108, 8408, 49334, 1096, 110, 108, 111, 2521, 15593, 6045, 107, 12061, 802, 110, 108, 7732, 26588, 113, 1044, 171, 146, 2847, 112, 253, 3073, 110, 107, 115, 663, 110, 108, 109, 207, 113, 110, 24397, 116, 432, 2791, 2991, 160, 3726, 9361, 7557, 107, 14323, 2000, 115, 500, 1683, 110, 108, 110, 24397, 116, 195, 374, 112, 22384, 1380, 5690, 166, 110, 108, 132, 166, 112, 10497, 8973, 115, 1044, 1843, 110, 55654, 476, 2455, 154, 197, 665, 3957, 943, 110, 20815, 110, 107, 219, 1683, 953, 1044, 122, 291, 1708, 15791, 110, 108, 253, 130, 4622, 110, 108, 9577, 110, 108, 693, 111, 3464, 110, 108, 73483, 110, 108, 111, 29516, 116, 107, 43381, 109, 1905, 113, 1407, 112, 110, 8723, 34764, 554, 1879, 10124, 16374, 115, 1044, 122, 1695, 117, 9417, 11589, 112, 109, 3819, 2889, 15642, 449, 110, 108, 115, 162, 109, 281, 872, 113, 110, 8723, 34764, 20547, 34261, 13957, 109, 1366, 113, 14778, 2889, 110, 108, 2409, 5630, 2889, 2062, 107, 3602, 4448, 2889, 15642, 110, 108, 115, 3945, 110, 108, 4403, 173, 2889, 1366, 117, 15771, 262, 2889, 2062, 127, 29599, 143, 13204, 110, 68545, 10124, 110, 108, 110, 105, 1061, 110, 4652, 943, 16342, 110, 206, 12908, 22700, 110, 108, 110, 105, 5658, 250, 5517, 178, 14135, 72118, 110, 108, 114, 34700, 11265, 1788, 141, 109, 6395, 110, 108, 117, 164, 233, 9400, 115, 4906, 17120, 1653, 330, 1695, 110, 107, 178, 14135, 72118, 41465, 2889, 2725, 482, 2201, 26545, 110, 108, 2297, 13621, 109, 9808, 113, 3600, 2889, 111, 110, 26889, 12126, 113, 9418, 2889, 110, 108, 964, 112, 142, 1562, 5099, 113, 2889, 233, 7162, 110, 8723, 34764, 20547, 34261, 107, 5700, 5357, 223, 24374, 6316, 8703, 109, 868, 113, 34357, 143, 53301, 110, 158, 2889, 115, 663, 112, 110, 24397, 116, 115, 109, 791, 113, 32680, 115, 1044, 122, 1695, 110, 107, 223, 113, 219, 1683, 2375, 2757, 115, 110, 24397, 1407, 110, 108, 166, 112, 35845, 1407, 110, 108, 3746, 115, 110, 24397, 5734, 110, 108, 111, 2757, 115, 15593, 6045, 5384, 143, 173, 5844, 110, 158, 115, 5089, 113, 109, 1852, 204, 110, 24397, 116, 1600, 110, 107, 109, 5221, 1280, 140, 1991, 113, 13757, 2889, 5384, 107, 6113, 7090, 156, 692, 374, 114, 43693, 3746, 115, 109, 344, 113, 1044, 7662, 69450, 107, 7090, 136, 4947, 692, 9068, 109, 14376, 111, 17890, 113, 53301, 2889, 11325, 19206, 115, 1044, 122, 1695, 170, 133, 32680, 111, 170, 127, 12857, 791, 122, 16020, 111, 191, 490, 7755, 2495, 347, 109, 207, 113, 110, 24397, 116, 110, 107, 1044, 915, 109, 692, 791, 118, 665, 899, 1734, 141, 114, 6220, 6479, 857, 233, 164, 908, 110, 107, 3352, 1044, 195, 134, 583, 1204, 231, 459, 110, 108, 160, 112, 388, 114, 2891, 113, 16020, 111, 191, 490, 7755, 2495, 373, 305, 396, 113, 7476, 110, 108, 111, 196, 114, 609, 3809, 326, 47945, 72673, 110, 108, 110, 55654, 1099, 113, 68335, 3957, 943, 110, 20815, 132, 478, 110, 108, 114, 271, 21570, 113, 154, 197, 1202, 899, 110, 108, 111, 142, 7257, 12595, 28007, 456, 637, 1932, 113, 11364, 110, 107, 1044, 195, 163, 656, 112, 133, 114, 13204, 110, 68545, 10124, 476, 113, 1061, 110, 4652, 943, 16342, 132, 902, 132, 1955, 10420, 22700, 143, 110, 144, 116, 2130, 110, 158, 1099, 113, 8667, 132, 902, 111, 112, 133, 915, 220, 110, 24397, 116, 132, 53301, 2889, 2495, 373, 677, 390, 111, 220, 4868, 2889, 2495, 143, 2684, 5111, 943, 242, 132, 154, 110, 158, 373, 624, 390, 269, 9280, 110, 107, 1044, 195, 12489, 118, 66980, 554, 8723, 307, 3882, 30012, 2288, 556, 124, 1458, 896, 110, 108, 16763, 54972, 110, 108, 110, 26889, 11300, 110, 108, 50787, 132, 5692, 3027, 3602, 15642, 110, 108, 9386, 13204, 110, 68545, 10124, 143, 10511, 110, 4652, 943, 16342, 110, 158, 132, 1955, 10420, 22700, 143, 110, 144, 116, 2130, 110, 158, 143, 20495, 110, 158, 1099, 110, 108, 5248, 132, 60193, 110, 108, 6395, 15624, 143, 2476, 280, 132, 902, 451, 124, 1146, 1695, 12189, 830, 20970, 4249, 110, 158, 110, 108, 20768, 15624, 143, 13204, 69842, 1099, 4586, 5111, 943, 110, 20815, 110, 158, 110, 108, 1371, 4917, 6279, 16673, 13448, 110, 108, 510, 132, 328, 689, 113, 178, 5196, 77785, 15219, 110, 108, 1229, 88392, 1133, 95716, 692, 4054, 110, 108, 110, 71911, 112, 53301, 2889, 110, 108, 1108, 1458, 2201, 49334, 373, 109, 289, 280, 899, 110, 108, 1], [15962, 39237, 35368, 144, 16865, 143, 110, 144, 252, 110, 158, 110, 108, 114, 57070, 477, 1298, 244, 895, 3411, 112, 76004, 116, 110, 108, 117, 10592, 141, 391, 132, 956, 110, 108, 8123, 110, 108, 42245, 26357, 113, 114, 3526, 132, 3526, 456, 110, 108, 122, 23495, 5573, 110, 108, 1813, 2283, 110, 108, 162, 218, 2384, 109, 18825, 110, 108, 11850, 110, 108, 3464, 110, 108, 132, 749, 110, 107, 110, 144, 252, 148, 174, 1673, 112, 1070, 115, 160, 15418, 113, 1044, 170, 133, 196, 300, 233, 1286, 3411, 112, 76004, 116, 110, 107, 110, 107, 109, 580, 887, 113, 110, 144, 252, 118, 50581, 76004, 116, 117, 666, 112, 711, 135, 153, 5404, 22685, 118, 34641, 22293, 110, 107, 1711, 122, 2953, 110, 108, 50581, 76004, 3073, 133, 114, 1626, 22685, 118, 33668, 6604, 24371, 522, 304, 197, 34641, 3138, 522, 22293, 110, 108, 122, 114, 580, 39147, 112, 20598, 110, 144, 252, 110, 107, 790, 136, 83931, 117, 666, 112, 133, 38492, 918, 134, 49419, 83531, 2288, 204, 19226, 20019, 60690, 9550, 34641, 20890, 2288, 12252, 111, 117, 110, 108, 1923, 110, 108, 1589, 122, 114, 221, 580, 15111, 113, 911, 9662, 40299, 14644, 16569, 143, 110, 42843, 110, 158, 110, 107, 21905, 110, 108, 114, 25754, 1382, 113, 3922, 1546, 12948, 6316, 3498, 120, 83931, 163, 7997, 66998, 2775, 113, 15962, 39237, 5573, 110, 107, 145, 731, 114, 437, 113, 9559, 1019, 233, 459, 3719, 110, 108, 10857, 112, 1074, 32021, 755, 110, 108, 8115, 164, 112, 280, 971, 110, 108, 1848, 122, 3726, 46532, 35368, 31897, 518, 22178, 3464, 5573, 1126, 1868, 305, 110, 1100, 110, 107, 3794, 689, 6031, 3264, 178, 140, 646, 110, 10885, 36924, 19464, 280, 5111, 77651, 118, 280, 590, 111, 237, 83931, 371, 5111, 118, 372, 384, 590, 110, 107, 1082, 113, 3464, 35368, 144, 16865, 113, 1532, 134, 1925, 231, 110, 108, 109, 1532, 1848, 122, 7635, 7630, 59854, 110, 108, 509, 40026, 124, 360, 2887, 111, 8231, 328, 549, 110, 108, 850, 429, 135, 238, 110, 108, 15114, 6902, 5746, 110, 108, 23632, 1759, 110, 108, 1756, 22041, 118, 280, 390, 110, 206, 162, 140, 73851, 244, 9817, 110, 107, 992, 112, 109, 1499, 110, 108, 156, 1151, 382, 133, 8926, 943, 266, 546, 113, 342, 589, 111, 244, 120, 1532, 3135, 313, 165, 113, 480, 110, 108, 111, 3597, 607, 2137, 2775, 110, 107, 136, 140, 15186, 130, 45949, 122, 446, 4464, 7457, 15866, 675, 110, 108, 111, 178, 140, 2839, 122, 110, 10885, 36924, 19464, 280, 5111, 943, 242, 118, 280, 590, 111, 237, 122, 83931, 371, 5111, 943, 242, 118, 384, 590, 110, 107, 115, 289, 228, 857, 233, 164, 116, 1532, 368, 146, 799, 1847, 110, 108, 111, 1499, 1668, 4694, 3464, 5573, 110, 108, 162, 195, 784, 130, 114, 297, 113, 169, 1380, 43022, 6671, 111, 146, 784, 3415, 110, 108, 6522, 24215, 3464, 5573, 1668, 195, 2987, 130, 297, 113, 22214, 2764, 743, 50497, 181, 6555, 115, 2959, 110, 107, 130, 3464, 35368, 144, 16865, 1562, 110, 108, 109, 1532, 196, 114, 3726, 5907, 130, 1532, 196, 112, 376, 169, 1233, 893, 169, 693, 118, 109, 337, 110, 107, 109, 1815, 192, 5148, 173, 109, 1532, 140, 7538, 308, 111, 140, 12001, 333, 1756, 110, 107, 178, 254, 3135, 646, 425, 640, 112, 3726, 3464, 5573, 395, 19728, 111, 33384, 1011, 110, 107, 169, 2755, 111, 616, 12112, 17193, 195, 1644, 110, 107, 333, 19496, 231, 113, 779, 1532, 140, 115, 44216, 16778, 111, 26364, 76629, 110, 107, 176, 11243, 195, 8115, 122, 2080, 1034, 116, 1393, 110, 108, 111, 1532, 140, 163, 1406, 112, 399, 110, 108, 155, 640, 112, 115, 65167, 111, 69433, 110, 108, 178, 368, 146, 1566, 280, 971, 244, 339, 5086, 110, 107, 178, 518, 109, 17176, 110, 107, 122, 1077, 2735, 7233, 111, 271, 766, 110, 108, 178, 947, 130, 142, 63134, 75762, 115, 109, 3153, 3067, 130, 114, 1360, 561, 110, 107, 110, 108, 178, 140, 374, 112, 129, 509, 204, 2611, 110, 108, 16373, 110, 108, 39801, 110, 108, 111, 613, 26597, 110, 107, 1254, 110, 108, 109, 1532, 196, 13167, 525, 6799, 110, 206, 118, 162, 169, 594, 266, 546, 113, 342, 110, 108, 111, 38543, 342, 110, 107, 124, 2287, 1932, 2843, 110, 108, 3337, 9051, 110, 108, 16331, 1434, 2749, 110, 108, 13401, 79150, 14506, 110, 108, 2617, 1579, 4712, 110, 108, 22085, 1026, 233, 20610, 110, 108, 7214, 20003, 7705, 2037, 195, 1644, 110, 107, 244, 4985, 12263, 81170, 110, 108, 110, 62801, 1034, 116, 1568, 111, 176, 4367, 2791, 113, 35368, 144, 16865, 195, 8258, 165, 110, 107, 109, 1532, 140, 2839, 122, 57745, 305, 5111, 916, 17722, 1907, 143, 110, 144, 252, 116, 110, 158, 110, 108, 35616, 15002, 43933, 1182, 5111, 110, 144, 252, 116, 110, 108, 9100, 55492, 457, 30209, 4915, 7760, 280, 5111, 23349, 6006, 143, 110, 28794, 110, 158, 110, 107, 244, 280, 590, 110, 108, 186, 140, 181, 2757, 113, 279, 7732, 110, 107, 13160, 33115, 22872, 377, 5111, 140, 717, 110, 206, 1562, 164, 112, 599, 5111, 110, 108, 9100, 55492, 415, 30209, 4915, 7760, 2785, 112, 280, 5111, 110, 107, 122, 332, 2757, 244, 384, 590, 113, 791, 118, 35368, 144, 16865, 110, 108, 7189, 53935, 32101, 1877, 439, 20301, 32101, 143, 1061, 1877, 1182, 110, 158, 140, 717, 141, 12263, 81170, 111, 1562, 164, 112, 4806, 110, 144, 252, 116, 111, 13160, 33115, 22872, 28902, 110, 107, 244, 665, 590, 113, 791, 110, 108, 1532, 148, 2521, 279, 8895, 122, 35616, 15002, 43933, 4781, 5111, 110, 108, 7189, 53935, 32101, 1877, 439, 20301, 32101, 143, 1061, 1877, 1182, 110, 158, 233, 4806, 110, 28794, 110, 108, 111, 57745, 305, 5111, 110, 28794, 110, 107, 1678, 437, 1574, 1668, 110, 144, 252, 1690, 122, 281, 233, 5734, 50581, 76004, 116, 253, 130, 83931, 599, 5111, 132, 114, 4278, 68441, 551, 54040, 738, 5111, 122, 895, 5400, 113, 3411, 113, 279, 665, 4262, 590, 115, 1614, 20140, 4222, 172, 28603, 132, 189, 176, 49451, 4222, 110, 107, 110, 42843, 115, 956, 111, 15962, 39237, 35368, 44121, 13856, 110, 108, 115, 970, 110, 108, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'labels': [[110, 105, 283, 2314, 1688, 1321, 23722, 115, 1044, 122, 1695, 170, 127, 12857, 1371, 2495, 117, 3732, 8674, 111, 218, 27811, 348, 113, 271, 115, 219, 1044, 110, 107, 109, 1298, 113, 1458, 49334, 117, 432, 4274, 111, 218, 129, 1589, 122, 1651, 9361, 702, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 110, 8723, 34764, 20547, 34261, 233, 13568, 3073, 127, 146, 957, 115, 7732, 26588, 113, 1044, 111, 218, 133, 114, 2404, 1298, 124, 1380, 5690, 107, 54867, 116, 497, 4676, 109, 14376, 111, 17890, 113, 34357, 2889, 2495, 115, 1044, 122, 1695, 170, 133, 609, 233, 2889, 233, 15642, 32680, 111, 170, 127, 12857, 791, 122, 16020, 347, 109, 207, 113, 110, 8723, 34764, 20547, 34261, 233, 13568, 3073, 107, 36005, 116, 30551, 1044, 122, 1907, 15791, 111, 609, 233, 2889, 233, 15642, 32680, 195, 953, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 28910, 11026, 74900, 134, 114, 5734, 113, 2416, 5111, 140, 634, 115, 613, 34357, 60760, 3309, 118, 114, 916, 113, 665, 899, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 58117, 476, 140, 5844, 134, 13757, 110, 108, 290, 296, 899, 110, 108, 111, 280, 899, 244, 109, 289, 2889, 20718, 143, 396, 1265, 110, 158, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 9361, 702, 985, 112, 34357, 2889, 195, 6667, 445, 1668, 107, 56131, 1313, 1182, 1044, 953, 110, 108, 1925, 143, 624, 95108, 110, 158, 1413, 134, 583, 339, 2889, 60760, 111, 1], [110, 105, 283, 2314, 15962, 39237, 35368, 144, 16865, 143, 110, 144, 252, 110, 158, 117, 114, 1651, 477, 1298, 113, 76004, 6098, 110, 108, 154, 122, 2953, 76004, 116, 110, 108, 120, 117, 3744, 38483, 115, 2790, 1044, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 1683, 403, 120, 7340, 50581, 76004, 116, 133, 114, 1074, 887, 113, 110, 144, 252, 110, 107, 130, 114, 711, 110, 108, 223, 17869, 218, 133, 1184, 114, 4797, 1083, 113, 750, 173, 31328, 219, 6098, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 145, 731, 114, 437, 113, 9559, 1019, 233, 459, 3719, 122, 8945, 92875, 1152, 28975, 111, 39205, 5272, 7233, 110, 108, 170, 1184, 3726, 110, 144, 252, 244, 580, 5734, 613, 5400, 3411, 112, 50581, 76004, 110, 10885, 36924, 19464, 111, 237, 83931, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 109, 1000, 113, 136, 800, 117, 112, 6034, 109, 3679, 112, 129, 69094, 111, 13740, 269, 303, 5048, 580, 5734, 453, 2233, 76004, 116, 115, 1532, 122, 220, 1962, 49451, 556, 110, 108, 8945, 92875, 1152, 28975, 110, 108, 132, 39205, 5272, 7233, 49044, 113, 2554, 2037, 1303, 110, 108, 170, 127, 154, 9539, 112, 1070, 9361, 1521, 253, 130, 110, 144, 252, 111, 3028, 109, 16121, 113, 110, 144, 252, 115, 1044, 646, 50581, 76004, 116, 110, 107, 110, 105, 191, 283, 2314, 1]]}"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "preprocess_function(raw_datasets['train'][:2])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zS-6iXTkIrJT"
   },
   "source": [
    "To apply this function on all the pairs of sentences in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "id": "DDtsaJeVIrJT"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "158e3eba7baf4338b49135bb836d4240",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/2 [00:00<?, ?ba/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5c20a00ed3e6438e804dc518ad10d03d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/1 [00:00<?, ?ba/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c816649cc6f14c42a1930ccea95b78b4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/1 [00:00<?, ?ba/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "voWiw8C7IrJV"
   },
   "source": [
    "Even better, the results are automatically cached by the 🤗 `Datasets` library to avoid spending time on this step the next time you run your notebook. The 🤗 `Datasets` library is normally smart enough to detect when the function you pass to map has changed (and thus requires to not use the cache data). For instance, it will properly detect if you change the task in the first cell and rerun the notebook. 🤗 `Datasets` warns you when it uses cached files, you can pass `load_from_cache_file=False` in the call to `map` to not use the cached files and force the preprocessing to be applied again.\n",
    "\n",
    "Note that we passed `batched=True` to encode the texts by batches together. This is to leverage the full benefit of the fast tokenizer we loaded earlier, which will use multi-threading to treat the texts in a batch concurrently."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "545PP3o8IrJV"
   },
   "source": [
    "## Fine-tuning the model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FBiW8UpKIrJW"
   },
   "source": [
    "Now that our data is ready, we can download the pretrained model and fine-tune it. Since our task is of the sequence-to-sequence kind, we use the `AutoModelForSeq2SeqLM` class. Like with the tokenizer, the `from_pretrained` method will download and cache the model for us."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "id": "TlqNaB8jIrJW"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4e63bfb7f0ab4fdca402afbc3e1f880f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/2.12G [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer\n",
    "\n",
    "model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CczA5lJlIrJX"
   },
   "source": [
    "Note that  we don't get a warning like in our classification example. This means we used all the weights of the pretrained model and there is no randomly initialized head in this case."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_N8urzhyIrJY"
   },
   "source": [
    "To instantiate a `Seq2SeqTrainer`, we will need to define three more things. The most important is the [`Seq2SeqTrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Seq2SeqTrainingArguments), which is a class that contains all the attributes to customize the training. It requires one folder name, which will be used to save the checkpoints of the model, and all other arguments are optional:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "id": "phpGhdw_ir69"
   },
   "outputs": [],
   "source": [
    "batch_size = 2\n",
    "model_name = model_checkpoint.split(\"/\")[-1]\n",
    "args = Seq2SeqTrainingArguments(\n",
    "    f\"{model_name}-finetuned-pubmed\",\n",
    "    evaluation_strategy = \"epoch\",\n",
    "    learning_rate=2e-5,\n",
    "    per_device_train_batch_size=batch_size,\n",
    "    per_device_eval_batch_size=batch_size,\n",
    "    weight_decay=0.01,\n",
    "    save_total_limit=3,\n",
    "    num_train_epochs=5,\n",
    "    predict_with_generate=True,\n",
    "    push_to_hub=True,\n",
    "    seed = 42,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "km3pGVdTIrJc"
   },
   "source": [
    "Here we set the evaluation to be done at the end of each epoch, tweak the learning rate, use the `batch_size` defined at the top of the cell and customize the weight decay. Since the `Seq2SeqTrainer` will save the model regularly and our dataset is quite large, we tell it to make three saves maximum. Lastly, we use the `predict_with_generate` option (to properly generate summaries) and activate mixed precision training (to go a bit faster).\n",
    "\n",
    "The last argument to setup everything so we can push the model to the [Hub](https://huggingface.co/models) regularly during training. Remove it if you didn't follow the installation steps at the top of the notebook. If you want to save your model locally in a name that is different than the name of the repository it will be pushed, or if you want to push your model under an organization and not your name space, use the `hub_model_id` argument to set the repo name (it needs to be the full name, including your namespace: for instance `\"sgugger/t5-finetuned-xsum\"` or `\"huggingface/t5-finetuned-xsum\"`).\n",
    "\n",
    "Then, we need a special kind of data collator, which will not only pad the inputs to the maximum length in the batch, but also the labels:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "id": "2QUAqk8Lir6-"
   },
   "outputs": [],
   "source": [
    "data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7sZOdRlRIrJd"
   },
   "source": [
    "The last thing to define for our `Seq2SeqTrainer` is how to compute the metrics from the predictions. We need to define a function for this, which will just use the `metric` we loaded earlier, and we have to do a bit of pre-processing to decode the predictions into texts:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "id": "UmvbnJ9JIrJd"
   },
   "outputs": [],
   "source": [
    "import nltk\n",
    "import numpy as np\n",
    "\n",
    "def compute_metrics(eval_pred):\n",
    "    predictions, labels = eval_pred\n",
    "    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)\n",
    "    # Replace -100 in the labels as we can't decode them.\n",
    "    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)\n",
    "    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)\n",
    "    \n",
    "    # Rouge expects a newline after each sentence\n",
    "    decoded_preds = [\"\\n\".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]\n",
    "    decoded_labels = [\"\\n\".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]\n",
    "    \n",
    "    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)\n",
    "    # Extract a few results\n",
    "    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}\n",
    "    \n",
    "    # Add mean generated length\n",
    "    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]\n",
    "    result[\"gen_len\"] = np.mean(prediction_lens)\n",
    "    \n",
    "    return {k: round(v, 4) for k, v in result.items()}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "rXuFTAzDIrJe"
   },
   "source": [
    "Then we just need to pass all of this along with our datasets to the `Seq2SeqTrainer`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "id": "imY1oC3SIrJf"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Cloning https://huggingface.co/Kevincp560/pegasus-cnn_dailymail-finetuned-pubmed into local empty directory.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    }
   ],
   "source": [
    "trainer = Seq2SeqTrainer(\n",
    "    model,\n",
    "    args,\n",
    "    train_dataset=tokenized_datasets[\"train\"],\n",
    "    eval_dataset=tokenized_datasets[\"validation\"],\n",
    "    data_collator=data_collator,\n",
    "    tokenizer=tokenizer,\n",
    "    compute_metrics=compute_metrics\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CdzABDVcIrJg"
   },
   "source": [
    "We can now finetune our model by just calling the `train` method:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "id": "uNx5pyRlIrJh"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "The following columns in the training set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning\n",
      "  warnings.warn(\n",
      "***** Running training *****\n",
      "  Num examples = 2000\n",
      "  Num Epochs = 5\n",
      "  Instantaneous batch size per device = 2\n",
      "  Total train batch size (w. parallel, distributed & accumulation) = 2\n",
      "  Gradient Accumulation steps = 1\n",
      "  Total optimization steps = 5000\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='5000' max='5000' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [5000/5000 1:30:25, Epoch 5/5]\n",
       "    </div>\n",
       "    <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       " <tr style=\"text-align: left;\">\n",
       "      <th>Epoch</th>\n",
       "      <th>Training Loss</th>\n",
       "      <th>Validation Loss</th>\n",
       "      <th>Rouge1</th>\n",
       "      <th>Rouge2</th>\n",
       "      <th>Rougel</th>\n",
       "      <th>Rougelsum</th>\n",
       "      <th>Gen Len</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>2.244900</td>\n",
       "      <td>1.894218</td>\n",
       "      <td>36.449400</td>\n",
       "      <td>14.994800</td>\n",
       "      <td>23.827900</td>\n",
       "      <td>33.308100</td>\n",
       "      <td>124.482000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>2.080300</td>\n",
       "      <td>1.844026</td>\n",
       "      <td>36.998000</td>\n",
       "      <td>15.499200</td>\n",
       "      <td>24.091000</td>\n",
       "      <td>33.661400</td>\n",
       "      <td>125.678000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>2.016600</td>\n",
       "      <td>1.817618</td>\n",
       "      <td>37.470300</td>\n",
       "      <td>16.035800</td>\n",
       "      <td>24.573500</td>\n",
       "      <td>34.178900</td>\n",
       "      <td>125.094000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1.991100</td>\n",
       "      <td>1.805519</td>\n",
       "      <td>37.133800</td>\n",
       "      <td>15.792100</td>\n",
       "      <td>24.141200</td>\n",
       "      <td>33.829300</td>\n",
       "      <td>125.874000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>1.941900</td>\n",
       "      <td>1.804964</td>\n",
       "      <td>37.256900</td>\n",
       "      <td>15.820500</td>\n",
       "      <td>24.196900</td>\n",
       "      <td>34.033100</td>\n",
       "      <td>125.892000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table><p>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Saving model checkpoint to pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-500\n",
      "Configuration saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-500/config.json\n",
      "Model weights saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-500/special_tokens_map.json\n",
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/special_tokens_map.json\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Saving model checkpoint to pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-1000\n",
      "Configuration saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-1000/config.json\n",
      "Model weights saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-1000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-1000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-1000/special_tokens_map.json\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "Saving model checkpoint to pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-1500\n",
      "Configuration saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-1500/config.json\n",
      "Model weights saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-1500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-1500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-1500/special_tokens_map.json\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/special_tokens_map.json\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Saving model checkpoint to pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-2000\n",
      "Configuration saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-2000/config.json\n",
      "Model weights saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-2000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-2000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-2000/special_tokens_map.json\n",
      "Deleting older checkpoint [pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-500] due to args.save_total_limit\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "Saving model checkpoint to pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-2500\n",
      "Configuration saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-2500/config.json\n",
      "Model weights saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-2500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-2500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-2500/special_tokens_map.json\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/special_tokens_map.json\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Deleting older checkpoint [pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-1000] due to args.save_total_limit\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Saving model checkpoint to pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-3000\n",
      "Configuration saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-3000/config.json\n",
      "Model weights saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-3000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-3000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-3000/special_tokens_map.json\n",
      "Deleting older checkpoint [pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-1500] due to args.save_total_limit\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "Saving model checkpoint to pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-3500\n",
      "Configuration saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-3500/config.json\n",
      "Model weights saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-3500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-3500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-3500/special_tokens_map.json\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/special_tokens_map.json\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Deleting older checkpoint [pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-2000] due to args.save_total_limit\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Saving model checkpoint to pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-4000\n",
      "Configuration saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-4000/config.json\n",
      "Model weights saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-4000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-4000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-4000/special_tokens_map.json\n",
      "Deleting older checkpoint [pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-2500] due to args.save_total_limit\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "Saving model checkpoint to pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-4500\n",
      "Configuration saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-4500/config.json\n",
      "Model weights saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-4500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-4500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-4500/special_tokens_map.json\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/special_tokens_map.json\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Deleting older checkpoint [pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-3000] due to args.save_total_limit\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Saving model checkpoint to pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-5000\n",
      "Configuration saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-5000/config.json\n",
      "Model weights saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-5000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-5000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-5000/special_tokens_map.json\n",
      "Deleting older checkpoint [pegasus-cnn_dailymail-finetuned-pubmed/checkpoint-3500] due to args.save_total_limit\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "\n",
      "\n",
      "Training completed. Do not forget to share your model on huggingface.co/models =)\n",
      "\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "TrainOutput(global_step=5000, training_loss=2.1037696166992186, metrics={'train_runtime': 5426.9617, 'train_samples_per_second': 1.843, 'train_steps_per_second': 0.921, 'total_flos': 2.885677635649536e+16, 'train_loss': 2.1037696166992186, 'epoch': 5.0})"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainer.train()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vHaPlSJyir6_"
   },
   "source": [
    "You can now upload the result of the training to the Hub, just execute this instruction:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "id": "jj7tm3Hvir6_"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Saving model checkpoint to pegasus-cnn_dailymail-finetuned-pubmed\n",
      "Configuration saved in pegasus-cnn_dailymail-finetuned-pubmed/config.json\n",
      "Model weights saved in pegasus-cnn_dailymail-finetuned-pubmed/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-cnn_dailymail-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-cnn_dailymail-finetuned-pubmed/special_tokens_map.json\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cb0c307844204c4dacac63de11292ffc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Upload file pytorch_model.bin:   0%|          | 32.0k/2.13G [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "To https://huggingface.co/Kevincp560/pegasus-cnn_dailymail-finetuned-pubmed\n",
      "   93d2f18..1566538  main -> main\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "To https://huggingface.co/Kevincp560/pegasus-cnn_dailymail-finetuned-pubmed\n",
      "   1566538..5bf91d0  main -> main\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'https://huggingface.co/Kevincp560/pegasus-cnn_dailymail-finetuned-pubmed/commit/156653812526e7ba6702b868242a095f05175822'"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainer.push_to_hub()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lMAoH478ir6_"
   },
   "source": [
    "You can now share this model with all your friends, family, favorite pets: they can all load it with the identifier `\"your-username/the-name-you-picked\"` so for instance:\n",
    "\n",
    "```python\n",
    "from transformers import AutoModelForSeq2SeqLM\n",
    "\n",
    "model = AutoModelForSeq2SeqLM.from_pretrained(\"sgugger/my-awesome-model\")\n",
    "```"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "machine_shape": "hm",
   "name": "pegasus-arxiv-pubmed-summary-final.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
