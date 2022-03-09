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
      "Collecting datasets\n",
      "  Downloading datasets-1.18.4-py3-none-any.whl (312 kB)\n",
      "\u001b[K     |████████████████████████████████| 312 kB 7.1 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting transformers\n",
      "  Downloading transformers-4.17.0-py3-none-any.whl (3.8 MB)\n",
      "\u001b[K     |████████████████████████████████| 3.8 MB 37.2 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting rouge-score\n",
      "  Downloading rouge_score-0.0.4-py2.py3-none-any.whl (22 kB)\n",
      "Collecting nltk\n",
      "  Downloading nltk-3.7-py3-none-any.whl (1.5 MB)\n",
      "\u001b[K     |████████████████████████████████| 1.5 MB 41.7 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: torch in ./miniconda3/envs/fastai/lib/python3.8/site-packages (1.9.1)\n",
      "Requirement already satisfied: ipywidgets in ./miniconda3/envs/fastai/lib/python3.8/site-packages (7.6.4)\n",
      "Requirement already satisfied: aiohttp in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (3.7.4.post0)\n",
      "Collecting dill\n",
      "  Downloading dill-0.3.4-py2.py3-none-any.whl (86 kB)\n",
      "\u001b[K     |████████████████████████████████| 86 kB 5.3 MB/s  eta 0:00:01\n",
      "\u001b[?25hCollecting pyarrow!=4.0.0,>=3.0.0\n",
      "  Downloading pyarrow-7.0.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (26.7 MB)\n",
      "\u001b[K     |████████████████████████████████| 26.7 MB 46.0 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting fsspec[http]>=2021.05.0\n",
      "  Downloading fsspec-2022.2.0-py3-none-any.whl (134 kB)\n",
      "\u001b[K     |████████████████████████████████| 134 kB 36.7 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting huggingface-hub<1.0.0,>=0.1.0\n",
      "  Downloading huggingface_hub-0.4.0-py3-none-any.whl (67 kB)\n",
      "\u001b[K     |████████████████████████████████| 67 kB 4.6 MB/s  eta 0:00:01\n",
      "\u001b[?25hCollecting responses<0.19\n",
      "  Downloading responses-0.18.0-py3-none-any.whl (38 kB)\n",
      "Requirement already satisfied: numpy>=1.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (1.20.3)\n",
      "Collecting multiprocess\n",
      "  Downloading multiprocess-0.70.12.2-py38-none-any.whl (128 kB)\n",
      "\u001b[K     |████████████████████████████████| 128 kB 39.3 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: requests>=2.19.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (2.26.0)\n",
      "Requirement already satisfied: tqdm>=4.62.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (4.62.2)\n",
      "Collecting xxhash\n",
      "  Downloading xxhash-3.0.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (212 kB)\n",
      "\u001b[K     |████████████████████████████████| 212 kB 41.5 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: packaging in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (21.0)\n",
      "Requirement already satisfied: pandas in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (1.3.2)\n",
      "Requirement already satisfied: pyyaml>=5.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (5.4.1)\n",
      "Collecting tokenizers!=0.11.3,>=0.11.1\n",
      "  Downloading tokenizers-0.11.6-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.5 MB)\n",
      "\u001b[K     |████████████████████████████████| 6.5 MB 34.6 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting filelock\n",
      "  Downloading filelock-3.6.0-py3-none-any.whl (10.0 kB)\n",
      "Collecting sacremoses\n",
      "  Downloading sacremoses-0.0.47-py2.py3-none-any.whl (895 kB)\n",
      "\u001b[K     |████████████████████████████████| 895 kB 43.6 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting regex!=2019.12.17\n",
      "  Downloading regex-2022.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (764 kB)\n",
      "\u001b[K     |████████████████████████████████| 764 kB 17.4 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: six>=1.14.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from rouge-score) (1.16.0)\n",
      "Collecting absl-py\n",
      "  Downloading absl_py-1.0.0-py3-none-any.whl (126 kB)\n",
      "\u001b[K     |████████████████████████████████| 126 kB 37.0 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: click in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nltk) (8.0.1)\n",
      "Requirement already satisfied: joblib in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nltk) (1.0.1)\n",
      "Requirement already satisfied: typing_extensions in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from torch) (3.10.0.2)\n",
      "Requirement already satisfied: nbformat>=4.2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (5.1.3)\n",
      "Requirement already satisfied: ipykernel>=4.5.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (6.2.0)\n",
      "Requirement already satisfied: widgetsnbextension~=3.5.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (3.5.1)\n",
      "Requirement already satisfied: jupyterlab-widgets>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (1.0.0)\n",
      "Requirement already satisfied: ipython-genutils~=0.2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (0.2.0)\n",
      "Requirement already satisfied: ipython>=4.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (7.27.0)\n",
      "Requirement already satisfied: traitlets>=4.3.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (5.1.0)\n",
      "Requirement already satisfied: debugpy<2.0,>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (1.4.1)\n",
      "Requirement already satisfied: matplotlib-inline<0.2.0,>=0.1.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (0.1.2)\n",
      "Requirement already satisfied: jupyter-client<8.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.1.12)\n",
      "Requirement already satisfied: tornado<7.0,>=4.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.1)\n",
      "Requirement already satisfied: decorator in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (5.0.9)\n",
      "Requirement already satisfied: setuptools>=18.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (58.0.4)\n",
      "Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (3.0.17)\n",
      "Requirement already satisfied: pygments in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (2.10.0)\n",
      "Requirement already satisfied: jedi>=0.16 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.18.0)\n",
      "Requirement already satisfied: pickleshare in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.7.5)\n",
      "Requirement already satisfied: backcall in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.2.0)\n",
      "Requirement already satisfied: pexpect>4.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (4.8.0)\n",
      "Requirement already satisfied: parso<0.9.0,>=0.8.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jedi>=0.16->ipython>=4.0.0->ipywidgets) (0.8.2)\n",
      "Requirement already satisfied: jupyter-core>=4.6.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jupyter-client<8.0->ipykernel>=4.5.1->ipywidgets) (4.7.1)\n",
      "Requirement already satisfied: python-dateutil>=2.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jupyter-client<8.0->ipykernel>=4.5.1->ipywidgets) (2.8.2)\n",
      "Requirement already satisfied: pyzmq>=13 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jupyter-client<8.0->ipykernel>=4.5.1->ipywidgets) (22.2.1)\n",
      "Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbformat>=4.2.0->ipywidgets) (3.2.0)\n",
      "Requirement already satisfied: attrs>=17.4.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (21.2.0)\n",
      "Requirement already satisfied: pyrsistent>=0.14.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (0.17.3)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from packaging->datasets) (2.4.7)\n",
      "Requirement already satisfied: ptyprocess>=0.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from pexpect>4.3->ipython>=4.0.0->ipywidgets) (0.7.0)\n",
      "Requirement already satisfied: wcwidth in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=4.0.0->ipywidgets) (0.2.5)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (1.26.6)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (2.0.4)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (2021.5.30)\n",
      "Requirement already satisfied: idna<4,>=2.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (3.2)\n",
      "Requirement already satisfied: notebook>=4.4.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from widgetsnbextension~=3.5.0->ipywidgets) (6.4.3)\n",
      "Requirement already satisfied: nbconvert in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (5.6.1)\n",
      "Requirement already satisfied: terminado>=0.8.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.9.4)\n",
      "Requirement already satisfied: argon2-cffi in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (20.1.0)\n",
      "Requirement already satisfied: prometheus-client in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.11.0)\n",
      "Requirement already satisfied: jinja2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (3.0.1)\n",
      "Requirement already satisfied: Send2Trash>=1.5.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.8.0)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (5.1.0)\n",
      "Requirement already satisfied: chardet<5.0,>=2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (4.0.0)\n",
      "Requirement already satisfied: async-timeout<4.0,>=3.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (3.0.1)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (1.5.1)\n",
      "Requirement already satisfied: cffi>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.14.0)\n",
      "Requirement already satisfied: pycparser in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from cffi>=1.0.0->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.20)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jinja2->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.0.1)\n",
      "Requirement already satisfied: pandocfilters>=1.4.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.4.3)\n",
      "Requirement already satisfied: defusedxml in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.7.1)\n",
      "Requirement already satisfied: testpath in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.5.0)\n",
      "Requirement already satisfied: bleach in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (4.0.0)\n",
      "Requirement already satisfied: mistune<2,>=0.8.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.8.4)\n",
      "Requirement already satisfied: entrypoints>=0.2.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.3)\n",
      "Requirement already satisfied: webencodings in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.5.1)\n",
      "Requirement already satisfied: pytz>=2017.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from pandas->datasets) (2021.1)\n",
      "Installing collected packages: regex, fsspec, filelock, dill, xxhash, tokenizers, sacremoses, responses, pyarrow, nltk, multiprocess, huggingface-hub, absl-py, transformers, rouge-score, datasets\n",
      "Successfully installed absl-py-1.0.0 datasets-1.18.4 dill-0.3.4 filelock-3.6.0 fsspec-2022.2.0 huggingface-hub-0.4.0 multiprocess-0.70.12.2 nltk-3.7 pyarrow-7.0.0 regex-2022.3.2 responses-0.18.0 rouge-score-0.0.4 sacremoses-0.0.47 tokenizers-0.11.6 transformers-4.17.0 xxhash-3.0.0\n"
     ]
    }
   ],
   "source": [
    "! pip install datasets transformers rouge-score nltk torch ipywidgets"
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
      "[nltk_data]   Unzipping tokenizers/punkt.zip.\n"
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
       "model_id": "7c880543892e47039795b477fd464e60",
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
      "The following NEW packages will be installed:\n",
      "  git-lfs\n",
      "0 upgraded, 1 newly installed, 0 to remove and 0 not upgraded.\n",
      "Need to get 3316 kB of archives.\n",
      "After this operation, 11.1 MB of additional disk space will be used.\n",
      "Get:1 http://fin1.clouds.archive.ubuntu.com/ubuntu focal/universe amd64 git-lfs amd64 2.9.2-1 [3316 kB]\n",
      "Fetched 3316 kB in 2s (2092 kB/s)  \u001b[0m33m\u001b[33m\u001b[33m\n",
      "\n",
      "\u001b7\u001b[0;23r\u001b8\u001b[1ASelecting previously unselected package git-lfs.\n",
      "(Reading database ... 143519 files and directories currently installed.)\n",
      "Preparing to unpack .../git-lfs_2.9.2-1_amd64.deb ...\n",
      "\u001b7\u001b[24;0f\u001b[42m\u001b[30mProgress: [  0%]\u001b[49m\u001b[39m [..........................................................] \u001b8\u001b7\u001b[24;0f\u001b[42m\u001b[30mProgress: [ 20%]\u001b[49m\u001b[39m [###########...............................................] \u001b8Unpacking git-lfs (2.9.2-1) ...\n",
      "\u001b7\u001b[24;0f\u001b[42m\u001b[30mProgress: [ 40%]\u001b[49m\u001b[39m [#######################...................................] \u001b8Setting up git-lfs (2.9.2-1) ...\n",
      "\u001b7\u001b[24;0f\u001b[42m\u001b[30mProgress: [ 60%]\u001b[49m\u001b[39m [##################################........................] \u001b8\u001b7\u001b[24;0f\u001b[42m\u001b[30mProgress: [ 80%]\u001b[49m\u001b[39m [##############################################............] \u001b8Processing triggers for man-db (2.9.1-1) ...\n",
      "\n",
      "\u001b7\u001b[0;24r\u001b8\u001b[1A\u001b[J"
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
    "model_checkpoint = \"google/bigbird-pegasus-large-arxiv\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4RRkXuteIrIh"
   },
   "source": [
    "This notebook is built to run  with any model checkpoint from the [Model Hub](https://huggingface.co/models) as long as that model has a sequence-to-sequence version in the Transformers library. Here we picked the [`google/bigbird-pegasus-large-arxiv`](https://huggingface.co/google/bigbird-pegasus-large-arxiv) checkpoint. "
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
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4a22183769d34cb2947a9c58ed9b2ed0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/4.88k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No config specified, defaulting to: pub_med_summarization_dataset/document\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading and preparing dataset pub_med_summarization_dataset/document to /home/user/.cache/huggingface/datasets/ccdv___pub_med_summarization_dataset/document/1.0.0/5792402f4d618f2f4e81ee177769870f365599daa729652338bac579552fec30...\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "87151408f6bb4a68ba2872cd0da805ce",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/779M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7b8da1475ad84fec8fef0e575a8c2b81",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/43.7M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d42af5cf51e34ebcafd9952c82a24921",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/43.8M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "0 examples [00:00, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "0 examples [00:00, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "0 examples [00:00, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset pub_med_summarization_dataset downloaded and prepared to /home/user/.cache/huggingface/datasets/ccdv___pub_med_summarization_dataset/document/1.0.0/5792402f4d618f2f4e81ee177769870f365599daa729652338bac579552fec30. Subsequent calls will reuse this data.\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1837fb9577fc47f9965cf49a0b26b6fb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/3 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "52e245151cda4affb0a1bbf948358f22",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/2.16k [00:00<?, ?B/s]"
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
       "      <td>a 65-year - old man with a medical history of stage iii nasopharyngeal cancer , end - stage renal disease treated with dialysis , hyperthyroidism , type 2 diabetes mellitus , hypertension , atrial fibrillation , and secondary hyperparathyroidism presented at the oral medicine clinic of the college of dentistry , university of florida for dental evaluation prior to the start of radiotherapy . intraoral radiographs were taken using a cs2000 intraoral x - ray system ( carestream dental llc , atlanta , ga , usa ) with # 2 soredex optime ( soredex , charlotte , nc , usa ) photo - stimulable phosphor ( psp ) sensors at 70 kvp , 7 ma , and a 0.142-second exposure time , using a standard bitewing technique . a panoramic image was obtained using an orthopantomograph  op100 d digital panoramic x - ray unit ( instrumentarium dental , tuusula , finland ) with exposure factors of 70 kvp with 12 ma for 17.6 seconds . the panoramic and bitewing radiographs revealed multiple tortuous vascular calcifications in the soft tissue of the neck and cheek bilaterally , with a dense \" rail track \" pattern of linear calcifications within the facial artery . based upon the radiographic presentation and the patient 's known medical history , multiple radiopaque entities were also noted in the soft tissue of the neck , more on the left than on the right , consistent with a diagnosis of carotid atherosclerosis ( figs . 1 and 2 ) . positron emission tomography / computed tomography ( pet / ct ) scans were performed using a philips gemini gxl 16 pet / ct system with 5 mm thickness . the acquired data were reviewed by a medical radiologist , and \" pipe - stem \" calcifications were observed in the bilateral , lingual , and facial arteries ( fig . calcifications may occur in several locations in the cardiovascular system , including the intima and media of vessels . intimal arterial calcification is associated with atherosclerosis , and vascular plaques form within the intima of the involved vessel.3 however , in mnckeberg arteriosclerosis , the calcific deposits are located entirely within the medial layer of the arterial wall and both the internal and external elastic membranes are spared.712 in a recent study , the prevalence of mnckeberg arteriosclerosis in the population was found to be 13.3% for males and 6.9% for females , and it is a well - recognized age - related phenomenon.1112 medial artery calcification can lead to vascular stiffness , resulting in increased vascular resistance , reduced compliance of the artery , and an inability to properly vasodilate in the setting of increased stress.13 medial calcinosis contributes to significant adverse cardiovascular outcomes in patients with chronic kidney disease and diabetes , where higher levels of medial artery calcification are a risk factor for amputation.14 the affected artery may not demonstrate evidence of a pulse . the exact pathogenic mechanism of medial calcinosis is not well understood . however , degenerative processes leading to the apoptosis or necrosis of medial smooth muscle cells and osteogenic processes leading to formation of bone - like structures are two distinct pathologic mechanisms that have been suggested for mnckeberg arteriosclerosis.15 meema et al.8 have suggested the possibility that two clinically and histologically different types of medial calcifications may exist . the first type is a benign , slowly progressive , essentially asymptomatic form with thin medial calcifications and little or no narrowing of the arterial lumen . in contrast , the second type is defined as a malignant , rapidly progressive form , in which massive and extensive medial calcifications may displace the internal elastica toward the lumen , causing luminal narrowing.78 mnckeberg initially described medial calcinosis as primarily affecting the arteries of the lower limbs , and occasionally affecting the peripheral arteries of the upper extremities . however , the process rarely affects the intraabdominal arteries , with the exception of the renal and splenic arteries.7 some reports in the literature have described mnckeberg arteriosclerosis . in 1977 , lachman et al.7 described the involvement of coronary , peripheral , and visceral arteries with mnckeberg calcification . a case of mnckeberg arteriosclerosis involving the aorta , pelvic , and lower limb arteries was reported by lanzer in 1998.16 the nonvascular involvement of soft tissue ( pharynx and larynx ) with mnckeberg sclerosis was reported by couri et al.6 in this report , we described a diabetic patient with end - stage renal disease on dialysis , who had advanced and previously undiagnosed mnckeberg medial calcinosis of the facial and lingual arteries . knowledge of the radiographic appearance of this calcification is clinically useful in developing a differential diagnosis . the proper interpretation of radiographic images presupposes a thorough knowledge of the anatomy , distribution , number , size , and shape of the calcifications . the calcified vessel appears as a parallel pair of thin , radiopaque lines that may have a straight course or a tortuous path , showing a pattern of blood vessels that looks like railroad tracks.511 carotid artery calcifications and phleboliths are calcifications that can be seen in the same location on a panoramic radiograph . carotid artery calcifications radiographically appear as curvilinear irregular parallel radiopacities in the soft tissues of the neck at or below the third and fourth cervical vertebrae , and inferior and lateral to the hyoid bone.1718 phleboliths can be seen on a panoramic radiograph as round or oval in shape with a homogeneously radiopaque center , giving phleboliths a \" target \" appearance.19 mnckeberg sclerosis is listed among the primary diseases of vessels that can be visualized on panoramic radiographs . to the best of our knowledge , our report describes the first known case of medial calcification in the facial artery on the panoramic radiograph of a diabetic patient with end - stage renal disease . soft tissue calcifications in the maxillofacial area are relatively common and can occur as the result of physiologic or pathologic mineralization , and generally correspond to radiographic findings in routine examinations , such as panoramic radiographs . a comprehensive review and thorough interpretation of all conventional and routine dental radiographs , especially beyond the region of interest , is necessary , and dental practitioners should be aware of the various calcified structures seen on panoramic radiographs , especially those associated with systemic diseases . a proper knowledge of radiographic features , however subtle they may be , assists the clinician in following up and further managing the patient , including appropriate referrals .</td>\n",
       "      <td>&lt;S&gt; mnckeberg sclerosis is a disease of unknown etiology , characterized by dystrophic calcification within the arterial tunica media of the lower extremities leading to reduced arterial compliance . &lt;/S&gt; &lt;S&gt; medial calcinosis does not obstruct the lumina of the arteries , and therefore does not lead to symptoms or signs of limb or organ ischemia . &lt;/S&gt; &lt;S&gt; mnckeberg sclerosis most commonly occurs in aged and diabetic individuals and in patients on dialysis . &lt;/S&gt; &lt;S&gt; mnckeberg arteriosclerosis is frequently observed in the visceral arteries , and it can occur in the head and neck region as well . &lt;/S&gt; &lt;S&gt; this report describes a remarkable case of mnckeberg arteriosclerosis in the head and neck region as detected on dental imaging studies . to the best of our knowledge , &lt;/S&gt; &lt;S&gt; this is the first case that has been reported in which this condition presented in the facial vasculature . &lt;/S&gt; &lt;S&gt; the aim of this report was to define the radiographic characteristics of mnckeberg arteriosclerosis in an effort to assist health care providers in diagnosing and managing this condition . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>professional mononuclear phagocytes , such as polymorphonuclear neutrophils ( pmn ) , monocytes , and macrophages , are considered as the first line of defence of the early host innate immune response [ 1 , 2 ] . their main function has been classically understood to kill invasive pathogens by a variety of potent intracellular microbicidal effector mechanisms [ 37 ] . after the first contact with pathogens , mononuclear phagocytes engulf and internalize them into their phagosomes . by the fusion with intracellular granules and the formation of phagolysosomes the pathogens may be killed intracellularly by a combination of non - oxidative and oxidative mechanisms [ 1 , 8 ] . actions of potent antimicrobial peptides , such as defensins , cathelicidins , cathepsins , pentraxin , and lactoferrin , are parts of non - oxidative killing mechanisms , while oxidative killing relies exclusively on the production of antimicrobial reactive oxygen species ( ros ) via the nadph oxidase ( nox ) complex . within blood circulating phagocytes , pmn are by far the most abundant cell population representing 5080% of the total white blood cells in different vertebrates . moreover , after being released from the bone marrow into the blood circulation , pmn are highly mobile and short - lived phagocytes , being densely packed with secretory granules [ 4 , 8 ] . pmn granules are categorized into three different types based on their contents : primary ( azurophilic ) , secondary ( specific ) , and tertiary ( gelatinase ) granules . the types of granules to be found in circulating pmn depend on their maturation stage . thus , pmn maturation starts with the formation of primary granules , followed by secondary and tertiary granules [ 4 , 9 , 10 ] . the content of primary granules includes myeloperoxidase ( mpo ) , neutrophil elastase ( ne ) , cathepsin g , proteinase 3 , defensins , and lysozyme ; secondary granules contain collagenase , gelatinase , cystatin , lysozyme , and lactoferrin ; and tertiary granules comprise gelatinase , lysozyme , and arginase amongst others . following granule maturation , pmn will possess all three types of granules displaying full killing capacity not only in the blood but also in tissues / organs and gut lumen . in addition , pmn act against pathogens by actively participating in complex inflammatory networks such as the release of a broad panel of proinflammatory chemokines , cytokines , and survival- and growth - factors which trigger both downstream proinflammatory effects and the transition into adaptive immune reactions . as such , several proinflammatory cytokines / chemokines were found enhanced in activated pmn in response to apicomplexan parasites , such as tnf- , il-1 , cc , and cxc chemokines ( e.g. , il-8 , ip-10 , gro- , rantes , and mip-1 ) [ 1115 ] . several of pmn - derived immunomodulatory molecules can augment the production of various chemokines and cytokines to further regulate phagocyte functions [ 16 , 17 ] . more importantly , by this means activated pmn recruit not only other innate immune cells but also t cells to the site of infection [ 1820 ] or even induce sterile inflammation [ 21 , 22 ] . beginning with the landmark study of brinkmann et al . , the paradigm of how pmn fight and kill pathogenic bacteria has profoundly been changed . the discovery of dna - based antimicrobial neutrophil extracellular traps ( nets ) changed the current knowledge on innate immune reactions not only on the level of the pathogen killing but also on the pathophysiology of metabolic , autoimmune , reproductive , and inflammatory diseases , as well as cancer progression [ 3237 ] . nets are released by activated pmn by a novel cell death process , called netosis , which can be stimulated by a variety of molecules and invasive pathogens . microorganisms such as bacteria [ 31 , 3941 ] , fungi [ 4244 ] , viruses [ 4549 ] , and parasites [ 5055 ] were identified as net inducers . also different molecules or cellular structures such as gm - csf / complement factor 5a [ 56 , 57 ] , activated platelets [ 40 , 58 ] , pma and zymosan [ 24 , 26 , 31 , 59 ] , singlet oxygen , lps [ 31 , 61 ] , and fc receptor   trigger netosis . in addition , il-8 as well - known chemoattractant for pmn was demonstrated as net inducer [ 31 , 62 ] . efficient netosis requires mature pmn and in most cases nox , mpo , ne , and peptidylarginine deiminase type iv ( pad4 ) activities [ 14 , 24 , 59 , 6365 ] . furthermore , the process of netosis obviously requires intracellularly signalling pathways of which raf - mek - erk kinases as well as p38 mapk are being the most frequently reported to be involved in this process [ 14 , 23 , 33 , 6669 ] . in addition , calcium release is needed for optimal net formation in different vertebrate species [ 14 , 23 , 7072 ] . upon stimulation of pmn , the nuclear envelope disintegrates permitting the mixture of chromatin with granular proteins / peptides . ne and mpo degrade histones ( h1 , h2a / h2b , h3 , and h4 ) and promote chromatin decondensation , mediated by pad4 via hypercitrullinating of specific histones to allow electrostatic coiling of the chromatin [ 64 , 73 , 74 ] . the total of the dna complexes being decorated with granular proteins / peptides and specific histones ( h1 , h2a / h2b , h3 , and h4 ) are finally extruded as nets to the extracellular environment by dying pmn . net formation is primarily a nox - dependent mechanism [ 14 , 24 , 59 , 75 , 76 ] . however , nox - independent netosis was also reported [ 29 , 60 , 67 , 68 , 77 ] . this mode of netosis is accompanied by a substantially lower level of erk activation and rather moderate level of akt activation , whereas activation of p38 is similar in both kinds of net formation [ 67 , 68 ] . as an example irrespectively of nox - dependency , pathogens may either be immobilised within sticky dna fibres [ 55 , 78 , 79 ] or be killed via the local high concentration of effector molecules [ 31 , 42 , 51 , 53 ] . meanwhile , other types of leukocytes of the innate immune system , such as macrophages [ 8083 ] , monocytes [ 26 , 28 ] , mast cells [ 84 , 85 ] , eosinophils [ 55 , 86 , 87 ] , and also basophils , have been reported to release net - like structures which are now collectively entitled as extracellular traps ( ets ) . described already many years ago that enucleated pmn may remain vital and are even capable of killing invasive microbes . more recent studies corroborated these findings proving that leukocytes do not necessarily die after et extrusion [ 56 , 68 , 86 ] . in this context , yousefi et al . [ 56 , 86 ] demonstrated that eosinophils and certain pmn subpopulations release ets of mitochondrial origin without dying . furthermore , yipp et al .   verified that pmn which had released nets were still viable and retained their capability to engulf bacteria via phagocytosis . however , it appears to be nonlethal for pmn and faster than nox - dependent net formation and to rely on a vesicular - based pathway releasing nuclear dna [ 33 , 68 ] . additionally , different molecular pathways will lead in a stimulus - dependent manner to the extrusion of different types of ets in vitro and in vivo . different morphological forms of ets were for the first time described in the human gout disease in vivo proving that monosodium urate crystals ( msu ) induced aggregated ( aggets ) , spread ( sprets ) , and diffused ( diffets ) et formation . as such , the parasitic nematode haemonchus contortus larvae triggered in ruminant pmn and eosinophils aggets , spreets , and diffets . while most net- and et - related studies focused on bacterial , viral , and fungal pathogens , little attention was paid to protozoan parasites . as such , the first ever published study on parasite - triggered netosis was published in 2008 by baker et al .   4 years after the discovery of this novel effector mechanism   and reported on plasmodium falciparum - triggered net formation . \\n parasites are mosquito - borne pathogens that cause malaria , a serious public health disease worldwide in the tropic and subtropics . globally , an estimated 3.3 billion people are at risk of being infected with malaria of whom approximately 1.2 billion are at high risk ( &gt; 1 in 1000 chance ) of developing malarial disease . the first report on p. falciparum - induced nets referred to p. falciparum - infected children and demonstrated in vivo net - entrapped trophozoite - infected erythrocytes in blood samples . moreover , baker and colleagues   provided first evidence on the involvement of parasite - triggered nets in the pathogenesis of malaria since the high levels of anti - dsdna antibodies were above the predictive levels for autoimmunity . interestingly , a recent study also indicates the capacity of p. falciparum to inhibit net formation   which may be of relevance in immunopathogenesis . thus , a mosquito - derived salivary protease inhibitor ( agaphelin ) induced by p. falciparum infection inhibited vertebrate elastase and net formation . whether this represents a true anti - net mechanism remains to be elucidated . parasites of the genus eimeria are worldwide of high veterinary and economic importance in livestock , especially in chicken , cattle and small ruminants [ 95100 ] . coccidiosis is a disease with high morbidity in animals of all ages , nonetheless inducing pathogenicity especially in young animals   and occasionally causing death of heavily infected animals [ 99 , 102 , 103 ] . several studies showed that pmn infiltrate intestinal mucosa in response to eimeria infections and are occasionally found in close contact to the parasitic stages in vivo [ 102 , 104107 ] . pmn have also been shown to directly interact with e. bovis stages and antigens in vitro , resulting in release of proinflammatory cytokines , chemokines , and inos . additionally , their phagocytic and oxidative burst activities were enhanced in response to eimeria stages in vitro and in vivo . first indications on eimeria spp . as potent net inducers came from behrendt and colleagues who reported on sporozoites to be entangled by an extracellular network of delicate dna fibres being extruded from pmn in vitro ( figure 1(a ) ) . using extracellular dna measurements and dnase treatments other studies confirmed typical characteristics of nets , such as the colocalization of ne , mpo , and histones in the dna backbone of eimeria - induced net - like structures . meanwhile , also other pathogenic ruminant eimeria species were shown to induce netosis , such as e. arloingi ( figures 2(a ) and 2(b ) ) [ 24 , 27 ] and e. ninakohlyakimovae ( prez , personal communication ) . importantly , muoz - caro and colleagues proved nets also to occur in vivo in eimeria - infected gut mucosa . the current data suggest that eimeria - induced netosis is a species- and stage - independent mechanism , since it was induced by sporozoites , merozoites i , or oocysts of different eimeria species [ 23 , 24 ] . given that pmn were described to act even in the intestinal lumen via different effector mechanisms [ 27 , 108 , 109 ] , it appears likely that interactions of luminal pmn with ingested eimeria oocysts or newly excysted sporozoites may occur [ 6 , 23 , 24 ] . in particular , net - related reactions against oocysts would have a high impact on the ongoing infection since they may hamper proper excystation of infective stages ( sporozoites ) and , in consequence , dampen the degree of infection at the earliest possible time point in the host . since e. arloingi sporozoites must egress from the oocyst circumplasm through the micropyle , nets covering this area of the oocyst will have a detrimental effect on proper excystation [ 6 , 24 ] . the same explanation seems feasible for e. bovis and e. ninakohlyakimovae , regardless of the fact that excystation occurs by rupture of the oocyst walls prior to sporozoites egress from sporocysts . although all eimeria species tested so far equally induced nets , significant differences in entrapment effectivity were reported amongst different host species , parasite species , and stages . thus , caprine nets immobilised a high proportion of e. arloingi sporozoites ( 72% ) , whilst in the bovine system considerably less parasite stages ( e. bovis sporozoites : 43% , b. besnoiti tachyzoites : 34% ) were found entrapped in net structures [ 23 , 59 ] . so far , it remains to be elucidated whether the varying effectivity of nets is based on the pmn origin ( goats are generally considered as strong immune responders ) or on the parasite species . the molecular basis of eimeria - induced netosis is not entirely understood , so far . enzyme activity measurements and inhibition studies revealed a key role of nox , ne , and mpo in eimeria - triggered net formation ( see table 1 ) which is in agreement to bacterial , fungal , and parasitic pathogens [ 14 , 25 , 59 , 65 , 75 , 110 ] . referring to signal cascades , analyses on the grade of phosphorylation revealed a key role of erk1/2 and p38 mapk in sporozoite - exposed bovine pmn . since respective inhibitor experiments led to decreased parasite - mediated net formation , muoz - caro et al .   this finding is in agreement with data on t. gondii - mediated net formation . referring to ca influx , further inhibition experiments proved e. bovis - mediated netosis as dependent on intracellular ca mobilization , since 2-abp ( inhibitor of store - operated ca entry )   and bapta - am ( binding intracellular ca ; muoz - caro , unpublished data ) but not egta ( inhibitor of ca influx from the extracellular compartment ; muoz - caro , unpublished data ) significantly blocked parasite - triggered netosis . so far , .   reported on enhanced cd11b surface expression on pmn following e. bovis sporozoite exposure . by antibody - mediated cd11b blockage leading to a significant reduction of parasite - triggered netosis bacteria and fungi netosis was reported as a lethal effector mechanism [ 31 , 42 ] . however , killing effects of nets were not observed in the case of eimeria spp . so far . given that eimeria spp . are obligate intracellular parasites , the main function of nets rather seems to be the extracellular immobilisation of infective stages hampering them from host cell invasion . accordingly , reduced host cell infections rates were reported for e. bovis and e. arloingi sporozoites when previously exposed to pmn [ 23 , 24 ] . the same feature was reported for monocyte - preexposed e. bovis sporozoites indicating that this leukocyte cell type also casts ets in response to this parasite stage and that etosis had an impact on parasite invasion . besides e. bovis , e. arloingi ( silva , unpublished data ) , and e. ninakohlyakimovae ( prez et al . , submitted manuscript ) were also shown to induce monocytes - derived ets . furthermore , e. ninakohlyakimovae - induced monocytes - etosis showed a rapid induction of ets release upon viable sporozoites , sporocysts , and oocysts encounters , corroborating a stage - independent process in monocyte - derived etosis . in addition , it was found that caprine monocyte - derived - etosis is nox - dependent . with the upregulation of the genes transcription encoding for il-12 and tnf- , relevant immunoregulatory cytokines with transition properties into the adaptive immunity   were also demonstrated in e. ninakohlyakimovae - exposed caprine monocytes ( prez et al , submitted manuscript ) . since the reduction in infection rates early after infection automatically results in decreased proliferation of the parasite , this indirect et - mediated effect should have a beneficial impact on the outcome of the disease . despite advantageous properties of ets , their ineffective clearance and/or poor regulation might also bear adverse pathological implications , leading to tissue damage in addition to enhanced local proinflammatory reactions [ 112 , 113 ] . toxoplasmosis is caused by the facultative heteroxenous apicomplexan polyxenous protozoan t. gondii representing one of the most common parasitic zoonoses worldwide . toxoplasma gondii is well known to affect almost all warm - blooded mammals including a wide range of domestic animals , wild mammals , marine mammals , marsupials , and humans [ 115 , 116 ] . in response to t. gondii infections , pmn are promptly recruited to the site of infection producing a variety of proinflammatory cytokines and chemokines [ 11 , 117 ] . in addition , pmn are capable of killing t. gondii tachyzoites via phagocytosis [ 118 , 119 ] . besides this effector mechanism , human , murine , bovine , and harbour seal ( phoca vitulina ) pmn additionally perform netosis in reaction to t. gondii tachyzoites ( figures 1(c ) and 1(d ) ) [ 25 , 26 ] . abi abdallah et al .   showed that netosis was triggered by tachyzoites in a parasite strain - independent fashion as an invasion / phagocytosis - independent process . interestingly , in the murine toxoplasmosis model , tachyzoites - induced nets were not the result of a random cell lysis , but of a controlled dna release process since lysozyme was still present in pmn after performing netosis [ 25 , 120 ] . in contrast to eimeria spp . , t. gondii - triggered netosis had modest toxoplasmacidal effects by killing up to 25% of the parasites . considering the obligate intracellular life style of t. gondii and its enormous proliferative capacity in mammalian host cells , parasite entrapment via nets might be of particular importance in vivo based on its interference with host cell invasion . consistently , harbour seal pmn - promoted nets significantly hampered host cell invasion of t. gondii tachyzoites in vitro . in vivo evidence of t. gondii - induced netosis was reported in a murine pulmonary infection model , revealing an increase of dsdna contents in the bronchoalveolar lavage fluids of t. gondii - infected mice . as equally reported for several other coccidian parasites [ 14 , 23 ] , t. gondii - induced nets were also proven to be nox- , ne- , mpo- , and ca- ( soce ) dependent and to be mediated by an erk 1/2-related signalling pathway in pmn ( see table 1 ) [ 25 , 26 ] . additionally , in earlier studies , not only the pivotal role of pmn but also the important role of monocytes in toxoplasmosis was clearly demonstrated [ 121123 ] ; however , their capacity to also induce ets in response to tachyzoite stages was just recently demonstrated . exposure of harbour seal - derived monocytes to viable t. gondii tachyzoites resulted in a significant induction of monocyte - ets and tachyzoites were firmly entrapped and immobilised within harbour seal monocyte - et structures , hampering parasite replication . bovine besnoitiosis caused by besnoitia besnoiti is an endemic disease in africa and asia [ 124126 ] and considered as emergent in europe . during the acute phase of cattle besnoitiosis , b. besnoiti tachyzoites mainly replicate in host endothelial cells of different organs [ 28 , 128 ] and , upon release , may be exposed to circulating leukocytes . besnoitia besnoiti tachyzoites were recently reported as effective inducers of pmn- and monocyte - derived ets ( figures 1(e ) , 1(g ) , and 1(h ) ) [ 28 , 59 ] . in the latter case , a high proportion of pmn was found to be involved in netosis , since up to 76% of encountered pmn were found to participate in netosis leading to the immobilisation of approximately one - third of the parasites . besnoitia besnoiti - triggered netosis furthermore proved as vitality - independent process that was even induced by soluble parasite molecules ( homogenates ) , though at lower levels . regarding pmn - derived effector molecules , nox , ne , and mpo proved as essential for efficient b. besnoiti - triggered netosis . thus , respective enzyme activities were encountered in tachyzoite - exposed pmn and chemical blockage of these enzymes via inhibitors blocked parasite - triggered netosis [ 28 , 59 ] . in contrast to tachyzoites of t. gondii , entrapped b. besnoiti tachyzoites were neither killed by nets nor ets since their host cell infectivity was entirely restored upon dnase i treatments [ 28 , 59 ] . given that b. besnoiti tachyzoites mainly proliferate within endothelial cells during the acute phase , these parasitic stages are released via cell lysis in close proximity to endothelium and are exposed to blood contents , such as leukocytes . several reports have shown that nets themselves interact with endothelium and may cause endothelial damage or dysfunction [ 129131 ] . since activated endothelial cells may produce a broad panel of immunomodulatory molecules with il-8 or p - selectin having been identified as potent net inducers [ 129 , 132 ] , interactions between infected endothelial cells , b. besnoiti tachyzoites , and nets are quite likely . recently reported on infection - induced upregulation of endothelial - derived il-8 and p - selectin gene transcription and furthermore presented indications on net formation occurring adjacent to infected endothelium after pmn adhesion assays being performed under physiological flow conditions as the ones present in small vessels . recent net - related investigations on the closely related cyst - forming apicomplexan protozoa neospora caninum have shown that bovine pmn exposed to viable tachyzoites also result in strong netosis ( figure 1(f ) ) . with regard to molecular mechanisms , n. caninum - triggered netosis seems to be p2y2- , nox- , soce- , mpo- , ne- , erk1/2- , p38 mapk- , and pad4-dependent ( villagra - blanco et al . , submitted manuscript ) . \\n cryptosporidium parvum is an euryxenous apicomplexan parasite with worldwide distribution and high zoonotic potential , mainly affecting young children , immunocompromised humans , and neonatal livestock . typically , cryptosporidiosis is a water- and food - borne enteric disease that causes diarrhoea , dehydration , weight losses , and abdominal pain and leads to significant economic losses in the livestock industry [ 133 , 134 ] . after ingestion , sporozoites are released from oocysts into the intestinal lumen and infect small intestine epithelial cells . recent studies reported on a significant contribution of pmn and macrophages to inflammatory responses in cryptosporidiosis in vivo [ 136 , 137 ] . muoz - caro and colleagues reported on nets being cast by both bovine and human pmn in response to c. parvum stages . parasite - triggered netosis proved stage - independent since it was induced by both sporozoites and oocysts ( figure 1(b ) ) . especially in the latter case parasite stages were occasionally entirely covered with net structures thereby most probably hampering proper sporozoite excystation . given that pmn were shown as active even within the intestinal lumen [ 108 , 109 , 138 , 139 ] , these reactions should have a significant impact on ongoing in vivo infection . in vitro infection experiments additionally showed the negative impact of nets on host cell invasion since infection rates were significantly reduced when using pmn - preexposed c. parvum stages . the fact that these reactions were entirely reversible via dnase i treatments rather argued against any cryptosporidicidal effects of nets . the colocalization of ne , histones , and mpo with dna in parasite - mediated extracellular fibres proved classical characteristics of nets and inhibitor experiments emphasized the key role of ne , nox , and mpo in efficient net formation . in agreement with findings on eimeria - induced netosis , inhibition experiments revealed c. parvum - triggered net formation as dependent on intracellular ca release and erk 1/2 and p38 mapk - mediated signalling pathways . interestingly , c. parvum sporozoite - exposed bovine pmn showed increased gene transcription of proinflammatory molecules , some of which were recently shown as potent net inducers ( e.g. , il-8 and tnf- ) [ 140 , 141 ] and may have potentiated net reactions . infections with leishmania spp . represent a major health problem and according to the who   10% of the human world population is at risk of infection , meaning that approximately 12 million people in 98 countries are infected , and 2 million new cases occur each year [ 142 , 143 ] . leishmaniasis is a vector - transmitted zoonosis caused by more than 25 different obligate intracellular protozoan leishmania species [ 142144 ] . particularly pmn have been implicated in the immunopathogenesis of leishmaniasis [ 145149 ] and recent studies examined the potential role of nets during the early phase of the disease of different leishmania species . .   showed for the first time that promastigotes of leishmania amazonensis , l. major , and l. chagasi were capable of triggering net formation . additionally , leishmania - triggered netosis seems not entirely stage - specific , since both promastigotes ( l. amazonensis , l. major , l. chagasi , l. donovani , l. mexicana , and l brasiliensis ) and amastigotes ( l. amazonensis , l. braziliensis ) promoted net formation in vitro and in vivo [ 51 , 147 , 150152 ] . more importantly , guimares - costa et al .   provided first indications on possible parasite - specific ligands being responsible for leishmania - mediated netosis . thus , leishmania - derived lipophosphoglycans ( lpg ) were suggested as the main trigger of net release since these molecules also induced nets in a purified form . the former authors showed that nets possessed detrimental effects on parasites as net - entrapped l. amazonensis promastigotes exhibited decreased viability . authors also demonstrated that the extracellular dna and histones found on nets were involved in the parasite inactivation / killing process . the leishmanicidal effects of histones were proven in promastigotes cocultures with purified h2a histones leading to the killing of parasites and by a significant reduction of leishmanicidal effects when cocultured in the presence of anti - histone antibodies . demonstrated that also the histone h2b could directly and efficiently kill promastigotes of l. amazonensis , l. major , l. braziliensis , and l. mexicana . in case of l. donovani , gabriel et al .   reported netosis as a ros - dependent process which was equally triggered in human and murine pmn ( see table 1 ) . however , leishmania - lipophosphoglycan- ( lpg- ) dependent net induction reported by guimares - costa et al .   was not observed with l. donovani . when using genetically modified l. donovani promastigotes gabriel et al nonetheless , in this infection system , lpg appeared to be involved in the resistance to nets - mediated killing , since the wild type of l. donovani maintained its viability in the presence of nets , whilst mutant parasites lacking lpg were efficiently killed by these extracellular structures . a more recent study revealed that leishmania parasites trigger not only the classical ros - dependent netosis as previously demonstrated but also a ros - independent form , named as early / rapid vital netosis . during this early / rapid leishmania - triggered netosis , in which net formation takes place after 515  min of activation without affecting pmn viability [ 29 , 68 ] , the parasites are also being efficiently entrapped . regarding net - related evasion strategies of trypanosomatidae parasites , leishmania spp . seem capable of evading net killing by firstly blocking the oxidative burst activity of pmn or even by resisting microbicidal activity of nets [ 145 , 150 ] . moreover , guimares - costa et al .   showed that l. infantum promastigotes express the enzyme 3-nucleotidase / nuclease which was previously described to be involved in parasite nutrition and infection and was proven to be part of the ability of promastigotes to escape net - mediated killing . a recent investigation has shown that a salivary component of the sand fly insect that transmits leishmaniasis may also play a role in the survival of leishmania in the definitive hosts , by modulating their innate immune system . a molecule named lundep from the salivary gland of lutzomyia longipalpis was recently described as an endonuclease with net - destroying properties in humans . in the presence of lundep , measured the ne release from nets as an indicator of net destruction , since ne is normally decorating nets backbone structures and found at low concentrations in culture supernatants , as previously demonstrated . lundep was responsible for the significant increase of ne concentration in the supernatants when compared to negative controls . in conclusion , these experiments showed degradation of dna scaffold of nets , destroying their functional integrity , and increasing promastigote survival and exacerbating l. major infection . approximately eight million people are affected by this tropical disease in the americas and an average of 12,000 deaths per year is known to occur due to american trypanosomiasis . it is well known that macrophages , eosinophils , monocytes , and pmn are implicated in the control of early infection [ 30 , 155 ] . .   demonstrated in vitro that t. cruzi is able to trigger nets in a dose- , time- , and ros - dependent manner . in agreement with reports on eimeria spp . and b. besnoiti [ 23 , 24 , 59 ] but in contrast to observations on t. gondii and leishmania spp . [ 25 , 51 ] , the viability of t. cruzi stages was not affected by nets , but netosis significantly impaired the parasite host cell infectivity . in fact , nets components as ne may affect t. cruzi infectivity , since this enzyme appears to be involved in increased trypanocidal activity and in the reduction of trypomastigote release by prestimulated infected macrophages [ 30 , 156 ] . additionally , the authors showed via antibody - mediated blockage that t. cruzi - triggered netosis is a tlr2- and tlr4-dependent process . moreover , the study showed that not only viable t. cruzi trypomastigote forms but also soluble antigens and killed t. cruzi parasites induced net release in human pmn . in vivo murine studies indicated the relevance of netosis for the outcome of trypanosomiasis since significantly decreased parasites numbers were found in the blood system of those animals which had previously been infected with nets - pretreated parasites . during the last years a vast amount of data on protozoan - mediated etosis was published strengthening the role of this effector mechanism in the defence of parasitic infections . several in vivo data have now proven the existence and importance of this early host innate effector mechanism . however , there is still a total lack of information on parasite - derived ligands triggering etosis . taking into account that in most cases et formation is considered as a species- and stage - independent process , rather ubiquitary occurring molecules may represent parasite - derived target molecules of ets . moreover , recent data revealed that other leukocytes such as monocytes , macrophages , basophils , mast cells , and eosinophils also perform etosis upon pathogen encounter . furthermore , et - related research mainly focused on the leukocytes aptitude to impact the parasites life cycle , but not on the propensity of parasitic stages to develop counter mechanisms for ets avoidance . while a bunch of data is available on bacterial nucleases or other counter mechanisms , taken together , we call for more parasite - related studies in the exciting field of etosis .</td>\n",
       "      <td>&lt;S&gt; professional mononuclear phagocytes such as polymorphonuclear neutrophils ( pmn ) , monocytes , and macrophages are considered as the first line of defence against invasive pathogens . &lt;/S&gt; &lt;S&gt; the formation of extracellular traps ( ets ) by activated mononuclear phagocytes is meanwhile well accepted as an effector mechanism of the early host innate immune response acting against microbial infections . &lt;/S&gt; &lt;S&gt; recent investigations showed evidence that etosis is a widely spread effector mechanism in vertebrates and invertebrates being utilized to entrap and kill bacteria , fungi , viruses , and protozoan parasites . &lt;/S&gt; &lt;S&gt; ets are released in response to intact protozoan parasites or to parasite - specific antigens in a controlled cell death process . &lt;/S&gt; &lt;S&gt; released ets consist of nuclear dna as backbone adorned with histones , antimicrobial peptides , and phagocyte - specific granular enzymes thereby producing a sticky extracellular matrix capable of entrapping and killing pathogens . &lt;/S&gt; &lt;S&gt; this review summarizes recent data on protozoa - induced etosis . &lt;/S&gt; &lt;S&gt; special attention will be given to molecular mechanisms of protozoa - induced etosis and on its consequences for the parasites successful reproduction and life cycle accomplishment . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>bat guano , an excrement of the cave - dwelling bats forms the basis of the ecology inside the cave by acting as a food source for detritivorous microbes . it contains high content of organic carbon , nitrogen , phosphate , and potassium , . bacteria present in bat guano were reported to be involved in nitrification process and were also known as potential chitinase producer . a clone library based study in bat guano samples has revealed the presence of group 1.1a and 1.1b crenarchaeota , an efficient ammonia oxidizer , . analyzing bat guano is also important since they often harbor various pathogens which can be thread for speleologists , and tourists . although the microbial communities in diverse cave ecosystems have been studied , little is known about the microbial communities of bat guano heaps ,   and there has been no studies using high throughput sequencing technology . meghalaya is known to possess the largest and most diverse karst caves in the world . pnahkyndeng cave located in ri - bhoi district of meghalaya , india is a home of various bats and offering an ideal environment for studying the bat guano microbiota without any anthropological influence . samples were collected on february 2014 from the bat guano of pnahkyndeng cave ( 255722.70n , 915543.10e ) , nongpoh , ri - bhoi district , india . ten composite guano samples were collected from different places of the cave floor and the soil community dna was extracted separately using the fast dna spin kit for soils ( mp biomedical , solon , oh , usa ) . the freshly extracted dna was purified twice using 0.5% low melting point agarose gel and mixed to prepare a composite sample . final dna concentrations were quantified by the using a microplate reader ( bmg labtech , jena , germany ) . the v4 region of the 16s rrna gene was amplified using f515/r806 primer combination ( 5-gtgccagcmgccgcggtaa-3 ; 5-ggactachvgggtwtctaat-3 ) . amplicon was extracted from 2% agarose gels and purified using the qia quick gel extraction kit ( qiagen , valencia , ca , usa ) according to the manufacturer 's instructions . quality filtering on raw sequences was performed according to base quality score distributions , average base content per read and gc distribution in the reads . singletons , the unique otu that did not cluster with other sequences , were removed as it might be a result of sequencing errors and can result in spurious otus . chimeras were also removed using uchime and pre - processed consensus v4 sequences were grouped into operational taxonomic units ( otus ) using the clustering program uclust at a similarity threshold of 0.97 , . all the pre - processed reads were used to identify the otus using qiime program for constructing a representative sequence for each otus . the representative sequence was finally aligned to the greengenes core set reference databases using pynast program , . , 403,529 reads were classified at the phylum , 282,350 at the order , 188,406 at the family and 2926 sequences were classified at the species levels . classified otus belonged to 18 different phyla dominated by chloroflexi , crenarchaeota , actinobacteria , bacteroidetes , proteobacteria , and planctomycetes ( fig .  1 ) . analysis of bacterial communities revealed the two most dominant bacteria 's  chloroflexi ( 29.97% ) and actinobacteria ( 22.55% ) , which are known to be a common inhabitant of cave microflora . other identified phyla include crenarchaeota ( 16.96% ) , planctomycetes ( 12.41% ) and proteobacteria ( 12.03% ) . chloroflexi was divided into 11 classes  thermomicrobia , planctomycetia , gitt - gs-136 , ktedonobacteria , anaerolineae , tk10 , tk17 , s085 , chloroflexi , ellin6529 , and gitt - gs-136 . the most dominant otu within this phyla was denovo 317 , classified under the class thermomicrobia ( 40.52% ) followed by denovo 710 under the thermomicrobia ( 7.61% ) , denovo 235 under the genus thermogemmatisporaceae ( 7.18% ) and denovo 3 under the genus gemmataceae ( 2.98% ) . the dominant otu within this phyla was denovo 74 under the genus mycobacterium ( 29.39% ) followed by denovo 993 ( 8.27% ) and denovo 372 ( 5.27% ) classified under the genus acidimicrobiales and actinomycetales , respectively . only five otus were classified up to the species level ( mycobacterium llatzerense and mycobacterium celatum ) . a third dominant phylum in this sample was identified as planctomycetes comprising of 91 otus and 46,063 reads . seventeen archeal otus were classified under the order nrp - j , methanomicrobiales and methanosarcinales . four of them were identified at the genus level ( methanosarcina , haloquadratum , methanosaeta and methanocorpusculum ) . the phylogenetic tree based on the genus level relationships is provided as supplementary fig .  1 . previous study on archaeal communities present in bat guano identified many ammonia oxidizing bacteria but it was limited with a few number of clones . our analysis provides in - depth and high throughput identification of the bacterial communities present in bat guano . in the present study , we identified 18 bacterial phyla and most of the bacterial genus identified was known to be involved in nitrogen cycling as seen in previous study . a significant portion of otus still remains unclassified which indicates the possibility for the presence of novel species in pnahkyndeng cave . further studies like whole metagenome sequencing or functional metagenomics can illustrate the detailed information of this bacterial community .</td>\n",
       "      <td>&lt;S&gt; v4 hypervariable region of 16s rdna was analyzed for identifying the bacterial communities present in bat guano from the unexplored cave  pnahkyndeng , meghalaya , northeast india . &lt;/S&gt; &lt;S&gt; metagenome comprised of 585,434 raw illumina sequences with a 59.59% g+c content . &lt;/S&gt; &lt;S&gt; a total of 416,490 preprocessed reads were clustered into 1282 otus ( operational taxonomical units ) comprising of 18 bacterial phyla . &lt;/S&gt; &lt;S&gt; the taxonomic profile showed that the guano bacterial community is dominated by chloroflexi , actinobacteria and crenarchaeota which account for 70.73% of all sequence reads and 43.83% of all otus . &lt;/S&gt; &lt;S&gt; metagenome sequence data are available at ncbi under the accession no . &lt;/S&gt; &lt;S&gt; srp051094 . &lt;/S&gt; &lt;S&gt; this study is the first to characterize bat guano bacterial community using next - generation sequencing approach . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>according to the international headache society ( ihs ) , classical trigeminal neuralgia ( tn ) is defined as  a unilateral disorder characterized by brief electric shock - like pains , abrupt in onset and termination , limited to the distribution of one or more divisions of the trigeminal nerve  . this is different from symptomatic tn , which is defined by ihs as  pain indistinguishable from classical tn but caused by a demonstrable structural lesion other than vascular compression  . patients often describe the pain as attacks or paroxysms , which may last for a few seconds to 2  min . interestingly between paroxysms there is a refractory period in which the pain can not be triggered and the patient is asymptomatic . the intensity of the pain is severe and the quality is usually described as electric shock - like , sharp , stabbing , or shooting . the pain might be triggered spontaneously or by light touch in a specific area or simply by eating or talking . tn is considered to be a rare disease with an annual incidence of 5.9/100,000 women and 3.4/100,000 men in the usa . the incidence increases with age and tends to be higher in women at all ages with a male to female ratio of 2:3 . the trigeminal nerve root entry zone has been found to be compressed by an aberrant loop of artery or vein , which ultimately leads to demyelination of the trigeminal nerve . furthermore , it has been demonstrated that tn is more common in patients with multiple sclerosis and an elevated relative risk has been associated with hypertension ( htn ) , particularly among women . it has been suggested that patients with tn have arterial tortuosity , which may lead to increased arterial pulse pressure waveforms secondary to vascular stiffness . however , very few studies have explored the relationship between htn and tn [ 2 , 4 ] . therefore , the objective of this study was to determine the prevalence of htn in patients diagnosed with classical tn and describe the characteristics of classical tn including age , gender and race among the patient population seen at the orofacial pain and oral medicine center at the usc school of dentistry ( usc ofp - om center ) in los angeles , california , usa between june 2003 and august 2007 . a retrospective chart review was conducted from the electronic medical record database ( soapware , fayetteville , ar ) at usc ofp - om center of over 3,000 patient records from june 2003 to august 2007 . the study was approved by the university of southern california university park institutional review board and ethics committee ( usc upirb # up-07 - 00416 ) and has therefore been performed in accordance with the ethical standards laid down in the 1964 declaration of helsinki . we identified all patients who were diagnosed with tn using the chart searcher function implemented in the soapware program with the appropriate search terminology . all patients were clinically diagnosed as having tn by the faculty , or residents under the supervision of faculty . a thorough history and head and neck exam was performed for every patient along with necessary radiographic investigations to rule out all potential dental and bony pathologies . brain mri with and without contrast was done for all patients prior to initiating treatment . only patients with classic tn ( idiopathic ) without any obvious pathology such as multiple sclerosis , plaques , tumors , and abnormalities of the skull base were considered in our study . inclusion criteria for tn included a history and clinical presentation that satisfied the international headache society criteria for classical tn . dental caries , periapical lesions , periodontal pockets with bone loss , cracked teeth , hyperocclusion , non - vital teeth and other bony pathologies were excluded with a thorough diagnostic workup . from this subset of tn patients , those currently taking anti - hypertensive medications for a minimum period of 6  months and with a diagnosis of htn ( established by the patient s physician ) were considered as having both tn and htn . for the control group we identified thrice the number of age- and gender - matched controls from the usc ofp - om center using the existing electronic medical record database ( soapware ) . inclusion criteria were gender - matched and age - matched controls ( matched within a 2-year range ) who may or may not have a diagnosis of htn . all control patients with htn were diagnosed by their physician and were on anti - hypertensive medication . chi - square test with yates correction and fisher s exact test were performed to calculate the p value . chi - square test with yates correction and fisher s exact test were performed to calculate the p value . the study population comprised of a total of 84 tn patients ( 54 female ; 30 male ) and age- and gender - matched controls ( n  =  252 ; 162 female ; 90 male ) between the ages of 33 and 93  years ( mean 65.3  years ) . the racial characteristics of our patient population were as follows : caucasian ( 102 ) ; hispanic ( 117 ) ; asian ( 33 ) ; black ( 38 ) ; american indian ( 3 ) ; others ( 19 ) and unknown ( 24 ) . thirty - one patients with tn reported having htn and were taking anti - hypertensive medication , out of which 13 of these patients were males and 18 were females . . the odds ratio of having htn in tn is 1.24 ( 95% confidence interval , 0.72 ) . the difference in prevalence of htn in tn cases versus controls was not found to be statistically significant using chi - square test with yates correction ( p  =  0.50 ) and fisher s exact test ( p  =  0.42 ) . hispanics had the highest prevalence of htn in both tn and controls followed by caucasians.fig .  1prevalence of htn in cases versus controls ; x axis tn cases and controls ; y axis prevalence of htn prevalence of htn in cases versus controls ; x axis tn cases and controls ; y axis prevalence of htn the risk factors that predispose an individual to develop tn include age and female gender . although , arterial hypertension has been reported as a risk factor for developing tn based on the theory of increased arterial tortuosity and pulse pressure , little or no epidemiologic data exist to validate this concept . arterial stiffness has been associated with the development of htn   and its association with tn has been investigated , but this relationship has not been established since studies failed to demonstrate that patients with tn have an increase in arterial stiffness . in a population - based study of tn patients conducted in rochester , minnesota , 25% ( 19 out 75 ) of the patients with tn were found to have htn with an odds ratio of 1.96 ( 95% confidence interval , 1.23 ) . in our study , we have found the prevalence of htn in tn to be 37% compared to the 32% seen in the control population . the reported rates of htn in the normal population have ranged from 28.7 to 29.3% [ 7 , 8 ] . the slight increase in the prevalence of htn in the control group might be attributed to the fact that our sample is a convenience sample . the 5% difference in the prevalence of htn between the two groups was not statistical significant . this might also be due to our sample size , which was too small to show a statistically significant difference . also , tn is a rare disease , with an overall prevalence of 0.10.2 per 1,000 and an incidence ranging from about 45/100,000/year , and therefore it is difficult to obtain a large tn patient sample . larger population - based studies might provide more evidence regarding the association between tn and htn . in our study , htn was not found to be a significant risk factor in patients with tn . the limitations of our study include the following : ( 1 ) retrospective chart review , ( 2 ) convenience sample , which will restrict the extrapolation of prevalence data to general population , and ( 3 ) lack of data regarding the duration of htn , which will be important as the vascular stiffness and nerve compression may worsen with prolonged hypertension . in general , retrospective studies are not the best way to study risk factors for diseases . therefore , further prospective studies , with an accurate clinical assessment of htn , are needed to conclusively clarify the possible relationship between htn and tn . to the best of our knowledge , our data show that hispanics had the highest prevalence of tn followed by caucasians in comparison to other races . since the los angeles area has a large population of hispanics this could explain the higher prevalence of tn noted in hispanics in this study . however , our study is comprised of a sample size with a very diverse racial composition , which prevents the racial differences from this study from being extrapolated to the general population of the united states . in conclusion , our results suggest that there is no correlation between htn and the development of tn . since , both tn and htn are seen in the elderly , it is possible that htn is simply a co - existing condition in patients with tn . this conclusion should be taken with caution as this is a retrospective study and further prospective studies , with an accurate clinical assessment of hypertension , are needed to conclusively clarify the possible relationship between htn and tn .</td>\n",
       "      <td>&lt;S&gt; it is unclear whether hypertension ( htn ) is a predisposing factor for the development of trigeminal neuralgia ( tn ) . &lt;/S&gt; &lt;S&gt; the purpose of this study was to determine the prevalence of htn in tn patients and controls at the usc orofacial pain and oral medicine center . &lt;/S&gt; &lt;S&gt; a retrospective chart review was conducted from a database of over 3,000 patient records from 2003 to 2007 . &lt;/S&gt; &lt;S&gt; we identified patients diagnosed with tn with or without htn . &lt;/S&gt; &lt;S&gt; a total of 84 patients ( 54 females ; 30 males ) between the ages of 33 and 93  years were diagnosed with tn ; 37% had tn with htn and 32% of controls had htn . &lt;/S&gt; &lt;S&gt; the increased prevalence of htn in the tn patients was not statistically significant ( p  =  0.50 ) . since &lt;/S&gt; &lt;S&gt; , both tn and htn are seen in the elderly , it is likely that htn is simply a co - existing condition in patients with tn and not a risk factor for its development . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>brown - squard syndrome ( bss ) , which occurs due to dysfunction of the spinothalamic tract , typically reflects the hemisection of the spinal cord at the cervical or thoracic level . the syndrome mainly occurs as a result of penetrating trauma , syringomyelia , hematomyelia , tumor , severe discs , or blunt trauma . among the multiple etiologies , the most common cause is penetrating trauma , such as a gunshot7,8 ) . therefore , most management guidelines focus on penetrating cervical injuries and/or vertebral artery ( va ) injury12 ) . non - missile penetrating spinal cord and va injuries are rare because of the bony structures that protect the spinal cord and va14 ) . thus , the treatment approach for wounds caused in non - missile penetrating spinal injuries such as a knife , a power drill bit , or even a pen could be different from common missile penetrating injuries5,6,13,17).to our knowledge , there are few reports in the literature of complete obstruction of the va due to penetration of a foreign body through the neural foramen into the spinal canal . herein , the authors report on va dissection and bss caused by penetration of an electric screw driver bit . a 25-year - old machine operator was involved in a violent episode and was stabbed in his right neck with an electric screw driver bit that was thrown by the opponent . on arrival at the emergency department , the electric screw driver bit was placed in the right lateral aspect of the neck at zone i ( fig . the tip of the electric screw driver bit was located at the center of the vertebral canal of c3 ( fig . he was given high - dose methylprednisolone ( bolus dose of 30mg / kg followed by 5.4mg / kg / hour for 23 hours ) according to the protocol for spinal cord injury . an immediate interventional angiography was undertaken without general anesthesia due to the nature of the emergency . the angiography revealed a total occlusion with dissection of the right va at the level of c3 . immediate coil embolization at both proximal and distal ends of the injury site was performed ( fig . an attempt at manual extraction of the electric screw driver bit failed with great resistance . after the patient was moved to the operating room , the electric screw driver bit was removed manually with muscle dissection under general anesthesia . venous blood spilled out and was controlled easily by application of several pieces of gelatin sponge . no postoperative complications such as wound dehiscence , cerebrospinal fluid ( csf ) leakage , or infection were observed . the neurological motor function of the right upper and lower extremities recovered to 3/5 and 4/5 , respectively , with persistent decreased sensory function after one year . fortunately , the patient experienced no neck swelling , auscultation of a neck bruit , or delayed ischemic complications . penetrating injury is the third most frequent cause of spinal cord injury in adults , surpassed only by traffic accidents and falls3,18 ) . stab wounds are associated with lesser surrounding tissue injury than gunshot wounds because the former delivers less energy than missile injuries9 ) . although vascular injury is the most common sequel of penetrating neck trauma , va injury is rare because it is well protected by the transverse foramen4,10 ) . therefore , penetrating injury of the va is mostly caused by gunshot wounds which deliver large kinetic energy , depending upon the bullet 's mass and speed12 ) . in this article , we report a rare case of va penetration by an electric screw driver bit with spinal cord insult , consequently presenting as bss . moreover , surgical exploration of the va can cause additional damage to the spine and surrounding tissues . therefore , it may be reasonable to embolize an occluded artery , because the unilateral ligation of the va rarely results in brainstem ischemia11,16 ) . there are a few reports regarding the treatment of traumatic va injury such as the arteriovenous fistulas and pseudoaneurysms2 ) . emergent surgical exploration is necessary for patients with hard signs of vascular injury , such as hemodynamic instability , hemorrhage exsanguinations , or expanding hematoma15 ) . patients that are hemodynamically stable and who are without respiratory compromise should undergo further diagnostic imaging evaluation15 ) . as presented in this case , endovascular techniques were a safe and effective method of treatment and were not associated with significant morbidity or mortality1 ) . airway management , intubation methods , and surgical positions can be points of debate between anesthesiologists and surgeons9 ) . if a lacerated va can be successfully obliterated , a penetrating electric screw driver bit may be extracted without general anesthesia . nevertheless , the authors recommend that surgeons should be prepared for conversion to open surgery and extraction should be performed with the support of a surgical team . we initially tried to extract the electric screw driver bit manually without general anesthesia in the intervention theater after va embolization . however , the electric screw driver bit was positioned firmly in the neural foramen , and the patient complained of severe pain when the electric screw driver bit was being pulled out . in addition , there was more important rationale that justified surgical exploration for extraction of the electric screw driver bit . on extraction of the electric screw driver bit , the authors describe a rare case of penetrating cervical injury caused by an electric screw driver bit with accompanying va penetration and bss .</td>\n",
       "      <td>&lt;S&gt; there are few reports in the literature of complete obstruction of the vertebral artery ( va ) due to an electric screw driver bit penetration through the neural foramen into the spinal canal with brown - squard syndrome ( bss ) . &lt;/S&gt; &lt;S&gt; a 25-year - old man was admitted to the emergency department with a penetrated neck injury by an electric screw driver bit after a struggle . &lt;/S&gt; &lt;S&gt; the patient presented the clinical features of bss . &lt;/S&gt; &lt;S&gt; computed tomography scan revealed that the electric screw driver bit penetrated through the right neural foramen at the level of c3 - 4 , and it caused an injury to the right half of the spinal cord . &lt;/S&gt; &lt;S&gt; emergent angiography revealed va dissection , which was managed by immediate coil embolization at both proximal and distal ends of the injury site . &lt;/S&gt; &lt;S&gt; after occlusion of the va , the electric screw driver bit was extracted under general anesthesia . &lt;/S&gt; &lt;S&gt; bleeding was minimal and controlled without difficulties . &lt;/S&gt; &lt;S&gt; no postoperative complications , such as wound dehiscence , csf leakage , or infection , were noted . &lt;/S&gt; &lt;S&gt; endovascular approaches for occlusion of vertebral artery lesions are safe and effective methods of treatment . &lt;/S&gt;</td>\n",
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
    "That vocabulary will be cached, so it's not downloaded again the next time we run the cell."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "eXNLu_-nIrJI"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a7fbe8aa1dba4666bd849d2efa0e7b50",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/1.17k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "faf8fa8086ae485a8a00605a1eab2d7c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/1.03k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fd3b3fbda7dc4526afc9d49fb7de8b38",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/1.83M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "715d50c0dc294ee6a05cf18998c1bd9e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/3.35M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "29ca36c18d5b4697b80e45778e0e3cbd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/775 [00:00<?, ?B/s]"
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
   "execution_count": 16,
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
     "execution_count": 16,
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
   "execution_count": 17,
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
     "execution_count": 17,
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
   "execution_count": 18,
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
   "execution_count": 19,
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
    "The max input length of `google/bigbird-pegasus-large-arxiv` is 4096, so `max_input_length = 4096`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "id": "vc0BSBLIIrJQ"
   },
   "outputs": [],
   "source": [
    "max_input_length = 4096\n",
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
   "execution_count": 21,
   "metadata": {
    "id": "-b70jh26IrJS"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'input_ids': [[126, 4403, 115, 154, 197, 4567, 113, 1044, 111, 218, 1111, 8895, 115, 878, 1020, 113, 15791, 110, 108, 704, 115, 1044, 12857, 16020, 111, 191, 490, 7755, 2495, 107, 740, 32680, 117, 3365, 130, 142, 14069, 22021, 476, 113, 58117, 143, 110, 55654, 110, 158, 143, 110, 55654, 110, 105, 665, 3957, 943, 110, 20815, 110, 158, 111, 218, 6860, 130, 114, 711, 113, 109, 5910, 1568, 110, 108, 11300, 110, 108, 2111, 5173, 110, 108, 16020, 110, 108, 132, 7755, 2495, 110, 107, 8823, 1683, 2298, 120, 5690, 111, 49159, 233, 2881, 562, 244, 7755, 2495, 110, 108, 704, 115, 693, 111, 3464, 15791, 110, 108, 218, 129, 12409, 141, 32680, 107, 6304, 32680, 432, 64142, 2775, 253, 130, 8466, 110, 108, 10353, 110, 108, 111, 35368, 1379, 28247, 110, 108, 111, 2297, 218, 133, 114, 2404, 1298, 124, 348, 113, 271, 143, 15593, 6045, 110, 158, 111, 637, 1932, 115, 1044, 122, 1695, 110, 107, 2297, 110, 108, 112, 927, 1312, 7233, 110, 108, 15593, 6045, 110, 108, 111, 32261, 115, 1044, 122, 1695, 110, 108, 126, 192, 129, 3048, 112, 248, 114, 10774, 1014, 115, 5987, 8149, 170, 217, 791, 118, 1695, 233, 1589, 32680, 143, 9073, 304, 110, 158, 111, 319, 5235, 603, 110, 107, 1458, 49334, 117, 142, 957, 230, 112, 2555, 29599, 110, 55654, 373, 114, 613, 908, 110, 108, 155, 109, 1298, 117, 110, 108, 6808, 110, 108, 4274, 111, 137, 1007, 1651, 9361, 3198, 111, 1562, 12439, 110, 107, 115, 24374, 2827, 6316, 115, 1044, 122, 9073, 304, 110, 108, 110, 8723, 34764, 20547, 34261, 233, 13568, 3073, 143, 110, 24397, 116, 110, 158, 1788, 1225, 3445, 115, 110, 55654, 476, 110, 108, 8408, 49334, 1096, 110, 108, 111, 2521, 15593, 6045, 107, 12061, 802, 110, 108, 7732, 26588, 113, 1044, 171, 146, 2847, 112, 253, 3073, 110, 107, 115, 663, 110, 108, 109, 207, 113, 110, 24397, 116, 432, 2791, 2991, 160, 3726, 9361, 7557, 107, 14323, 2000, 115, 500, 1683, 110, 108, 110, 24397, 116, 195, 374, 112, 22384, 1380, 5690, 166, 110, 108, 132, 166, 112, 10497, 8973, 115, 1044, 1843, 110, 55654, 476, 2455, 154, 197, 665, 3957, 943, 110, 20815, 110, 107, 219, 1683, 953, 1044, 122, 291, 1708, 15791, 110, 108, 253, 130, 4622, 110, 108, 9577, 110, 108, 693, 111, 3464, 110, 108, 73483, 110, 108, 111, 29516, 116, 107, 43381, 109, 1905, 113, 1407, 112, 110, 8723, 34764, 554, 1879, 10124, 16374, 115, 1044, 122, 1695, 117, 9417, 11589, 112, 109, 3819, 2889, 15642, 449, 110, 108, 115, 162, 109, 281, 872, 113, 110, 8723, 34764, 20547, 34261, 13957, 109, 1366, 113, 14778, 2889, 110, 108, 2409, 5630, 2889, 2062, 107, 3602, 4448, 2889, 15642, 110, 108, 115, 3945, 110, 108, 4403, 173, 2889, 1366, 117, 15771, 262, 2889, 2062, 127, 29599, 143, 13204, 110, 68545, 10124, 110, 108, 110, 105, 1061, 110, 4652, 943, 16342, 110, 206, 12908, 22700, 110, 108, 110, 105, 5658, 250, 5517, 178, 14135, 72118, 110, 108, 114, 34700, 11265, 1788, 141, 109, 6395, 110, 108, 117, 164, 233, 9400, 115, 4906, 17120, 1653, 330, 1695, 110, 107, 178, 14135, 72118, 41465, 2889, 2725, 482, 2201, 26545, 110, 108, 2297, 13621, 109, 9808, 113, 3600, 2889, 111, 110, 26889, 12126, 113, 9418, 2889, 110, 108, 964, 112, 142, 1562, 5099, 113, 2889, 233, 7162, 110, 8723, 34764, 20547, 34261, 107, 5700, 5357, 223, 24374, 6316, 8703, 109, 868, 113, 34357, 143, 53301, 110, 158, 2889, 115, 663, 112, 110, 24397, 116, 115, 109, 791, 113, 32680, 115, 1044, 122, 1695, 110, 107, 223, 113, 219, 1683, 2375, 2757, 115, 110, 24397, 1407, 110, 108, 166, 112, 35845, 1407, 110, 108, 3746, 115, 110, 24397, 5734, 110, 108, 111, 2757, 115, 15593, 6045, 5384, 143, 173, 5844, 110, 158, 115, 5089, 113, 109, 1852, 204, 110, 24397, 116, 1600, 110, 107, 109, 5221, 1280, 140, 1991, 113, 13757, 2889, 5384, 107, 6113, 7090, 156, 692, 374, 114, 43693, 3746, 115, 109, 344, 113, 1044, 7662, 69450, 107, 7090, 136, 4947, 692, 9068, 109, 14376, 111, 17890, 113, 53301, 2889, 11325, 19206, 115, 1044, 122, 1695, 170, 133, 32680, 111, 170, 127, 12857, 791, 122, 16020, 111, 191, 490, 7755, 2495, 347, 109, 207, 113, 110, 24397, 116, 110, 107, 1044, 915, 109, 692, 791, 118, 665, 899, 1734, 141, 114, 6220, 6479, 857, 233, 164, 908, 110, 107, 3352, 1044, 195, 134, 583, 1204, 231, 459, 110, 108, 160, 112, 388, 114, 2891, 113, 16020, 111, 191, 490, 7755, 2495, 373, 305, 396, 113, 7476, 110, 108, 111, 196, 114, 609, 3809, 326, 47945, 72673, 110, 108, 110, 55654, 1099, 113, 68335, 3957, 943, 110, 20815, 132, 478, 110, 108, 114, 271, 21570, 113, 154, 197, 1202, 899, 110, 108, 111, 142, 7257, 12595, 28007, 456, 637, 1932, 113, 11364, 110, 107, 1044, 195, 163, 656, 112, 133, 114, 13204, 110, 68545, 10124, 476, 113, 1061, 110, 4652, 943, 16342, 132, 902, 132, 1955, 10420, 22700, 143, 110, 144, 116, 2130, 110, 158, 1099, 113, 8667, 132, 902, 111, 112, 133, 915, 220, 110, 24397, 116, 132, 53301, 2889, 2495, 373, 677, 390, 111, 220, 4868, 2889, 2495, 143, 2684, 5111, 943, 242, 132, 154, 110, 158, 373, 624, 390, 269, 9280, 110, 107, 1044, 195, 12489, 118, 66980, 554, 8723, 307, 3882, 30012, 2288, 556, 124, 1458, 896, 110, 108, 16763, 54972, 110, 108, 110, 26889, 11300, 110, 108, 50787, 132, 5692, 3027, 3602, 15642, 110, 108, 9386, 13204, 110, 68545, 10124, 143, 10511, 110, 4652, 943, 16342, 110, 158, 132, 1955, 10420, 22700, 143, 110, 144, 116, 2130, 110, 158, 143, 20495, 110, 158, 1099, 110, 108, 5248, 132, 60193, 110, 108, 6395, 15624, 143, 2476, 280, 132, 902, 451, 124, 1146, 1695, 12189, 830, 20970, 4249, 110, 158, 110, 108, 20768, 15624, 143, 13204, 69842, 1099, 4586, 5111, 943, 110, 20815, 110, 158, 110, 108, 1371, 4917, 6279, 16673, 13448, 110, 108, 510, 132, 328, 689, 113, 178, 5196, 77785, 15219, 110, 108, 1229, 88392, 1133, 95716, 692, 4054, 110, 108, 110, 71911, 112, 53301, 2889, 110, 108, 1108, 1458, 2201, 49334, 373, 109, 289, 280, 899, 110, 108, 132, 189, 73965, 2575, 373, 677, 390, 269, 9280, 110, 107, 1044, 195, 146, 1608, 112, 248, 189, 5692, 110, 108, 6470, 110, 108, 132, 10942, 6683, 4510, 2684, 5111, 132, 154, 113, 2889, 446, 242, 132, 1061, 5111, 5692, 2895, 446, 242, 333, 109, 692, 132, 857, 233, 164, 908, 110, 107, 1458, 49334, 116, 195, 6413, 134, 109, 1708, 5786, 110, 116, 8866, 175, 110, 55654, 1099, 8408, 112, 608, 3957, 943, 110, 20815, 132, 478, 110, 108, 111, 253, 1044, 195, 1341, 791, 10114, 110, 107, 1158, 2976, 4898, 140, 735, 141, 149, 1044, 269, 692, 4054, 110, 108, 111, 109, 6985, 111, 2518, 2010, 195, 2444, 141, 109, 8644, 933, 1042, 113, 4138, 5124, 50187, 386, 1695, 1104, 110, 107, 109, 692, 140, 3047, 115, 5301, 122, 109, 12920, 113, 178, 1191, 40131, 457, 111, 234, 2827, 846, 130, 4400, 115, 109, 214, 929, 113, 2043, 3158, 120, 40453, 109, 1471, 113, 883, 4370, 111, 109, 7600, 113, 2827, 11610, 110, 107, 1044, 915, 2416, 5111, 28910, 11026, 65093, 74900, 24672, 115, 1061, 16342, 1644, 37938, 111, 15764, 204, 109, 422, 113, 305, 1269, 3309, 118, 114, 916, 113, 665, 899, 110, 107, 109, 211, 5734, 140, 634, 333, 109, 211, 5602, 558, 143, 384, 390, 135, 109, 24772, 113, 16020, 132, 7755, 2495, 110, 158, 110, 107, 110, 144, 116, 2130, 140, 10129, 110, 108, 130, 6985, 20291, 28145, 2889, 2495, 173, 110, 144, 116, 2130, 1099, 195, 902, 197, 4567, 110, 107, 134, 109, 211, 5602, 558, 143, 396, 305, 110, 206, 13757, 110, 158, 110, 108, 114, 1458, 2630, 140, 3686, 118, 6214, 8610, 269, 109, 692, 791, 140, 547, 110, 107, 1044, 3243, 3309, 5602, 4397, 118, 791, 111, 2843, 110, 206, 111, 2491, 118, 857, 233, 164, 4397, 134, 396, 1265, 162, 953, 114, 573, 1312, 4712, 110, 107, 573, 1458, 2664, 111, 110, 144, 116, 2130, 195, 479, 290, 296, 899, 110, 108, 111, 435, 280, 899, 244, 289, 791, 143, 396, 1265, 110, 158, 110, 107, 573, 6214, 2843, 143, 110, 55654, 110, 108, 13204, 110, 68545, 10124, 110, 108, 920, 8207, 5888, 63589, 2664, 110, 108, 1955, 10420, 110, 108, 110, 144, 116, 2130, 110, 108, 13204, 2889, 110, 108, 916, 2889, 7864, 1865, 110, 108, 1108, 2201, 22500, 110, 108, 695, 1458, 2201, 2664, 122, 13945, 110, 108, 53342, 2664, 110, 108, 111, 13204, 38604, 8472, 110, 158, 195, 479, 134, 396, 305, 111, 134, 396, 1265, 143, 370, 113, 692, 110, 158, 110, 107, 9361, 702, 195, 9068, 134, 276, 5602, 558, 430, 692, 4052, 132, 9644, 110, 108, 111, 333, 109, 677, 390, 244, 109, 289, 692, 791, 110, 107, 110, 55654, 804, 602, 195, 1848, 130, 1021, 110, 108, 8812, 110, 108, 111, 499, 224, 149, 665, 899, 110, 107, 3850, 317, 495, 113, 110, 55654, 476, 195, 266, 317, 109, 13757, 110, 55654, 111, 110, 55654, 1099, 115, 109, 645, 899, 110, 108, 303, 110, 144, 233, 804, 110, 107, 114, 7286, 30033, 113, 891, 110, 105, 38217, 140, 263, 115, 109, 1382, 110, 107, 149, 11019, 195, 2303, 303, 10446, 116, 824, 48612, 143, 10446, 116, 12189, 26805, 110, 108, 439, 415, 110, 108, 38080, 110, 108, 20399, 110, 158, 110, 107, 1044, 915, 109, 692, 791, 118, 665, 899, 1734, 141, 114, 6220, 6479, 857, 233, 164, 908, 110, 107, 3352, 1044, 195, 134, 583, 1204, 231, 459, 110, 108, 160, 112, 388, 114, 2891, 113, 16020, 111, 191, 490, 7755, 2495, 373, 305, 396, 113, 7476, 110, 108, 111, 196, 114, 609, 3809, 326, 47945, 72673, 110, 108, 110, 55654, 1099, 113, 68335, 3957, 943, 110, 20815, 132, 478, 110, 108, 114, 271, 21570, 113, 154, 197, 1202, 899, 110, 108, 111, 142, 7257, 12595, 28007, 456, 637, 1932, 113, 11364, 110, 107, 1044, 195, 163, 656, 112, 133, 114, 13204, 110, 68545, 10124, 476, 113, 1061, 110, 4652, 943, 16342, 132, 902, 132, 1955, 10420, 22700, 143, 110, 144, 116, 2130, 110, 158, 1099, 113, 8667, 132, 902, 111, 112, 133, 915, 220, 110, 24397, 116, 132, 53301, 2889, 2495, 373, 677, 390, 111, 220, 4868, 2889, 2495, 143, 2684, 5111, 943, 242, 132, 154, 110, 158, 373, 624, 390, 269, 9280, 110, 107, 1044, 195, 12489, 118, 66980, 554, 8723, 307, 3882, 30012, 2288, 556, 124, 1458, 896, 110, 108, 16763, 54972, 110, 108, 110, 26889, 11300, 110, 108, 50787, 132, 5692, 3027, 3602, 15642, 110, 108, 9386, 13204, 110, 68545, 10124, 143, 10511, 110, 4652, 943, 16342, 110, 158, 132, 1955, 10420, 22700, 143, 110, 144, 116, 2130, 110, 158, 143, 20495, 110, 158, 1099, 110, 108, 5248, 132, 60193, 110, 108, 6395, 15624, 143, 2476, 280, 132, 902, 451, 124, 1146, 1695, 12189, 830, 20970, 4249, 110, 158, 110, 108, 20768, 15624, 143, 13204, 69842, 1099, 4586, 5111, 943, 110, 20815, 110, 158, 110, 108, 1371, 4917, 6279, 16673, 13448, 110, 108, 510, 132, 328, 689, 113, 178, 5196, 77785, 15219, 110, 108, 1229, 88392, 1133, 95716, 692, 4054, 110, 108, 110, 71911, 112, 53301, 2889, 110, 108, 1108, 1458, 2201, 49334, 373, 109, 289, 280, 899, 110, 108, 132, 189, 73965, 2575, 373, 677, 390, 269, 9280, 110, 107, 1044, 195, 146, 1608, 112, 248, 189, 5692, 110, 108, 6470, 110, 108, 132, 10942, 6683, 4510, 2684, 5111, 132, 154, 113, 2889, 446, 242, 132, 1061, 5111, 5692, 2895, 446, 242, 333, 109, 692, 132, 857, 233, 164, 908, 110, 107, 1458, 49334, 116, 195, 6413, 134, 109, 1708, 5786, 110, 116, 8866, 175, 110, 55654, 1099, 8408, 112, 608, 3957, 943, 110, 20815, 132, 478, 110, 108, 111, 253, 1044, 195, 1341, 791, 10114, 110, 107, 1158, 2976, 4898, 140, 735, 141, 149, 1044, 269, 692, 4054, 110, 108, 111, 109, 6985, 111, 2518, 2010, 195, 2444, 141, 109, 8644, 933, 1042, 113, 4138, 5124, 50187, 386, 1695, 1104, 110, 107, 109, 692, 140, 3047, 115, 5301, 122, 109, 12920, 113, 178, 1191, 40131, 457, 111, 234, 2827, 846, 130, 4400, 115, 109, 214, 929, 113, 2043, 3158, 120, 40453, 109, 1471, 113, 883, 4370, 111, 109, 7600, 113, 2827, 11610, 110, 107, 1044, 915, 2416, 5111, 28910, 11026, 65093, 74900, 24672, 115, 1061, 16342, 1644, 37938, 111, 15764, 204, 109, 422, 113, 305, 1269, 3309, 118, 114, 916, 113, 665, 899, 110, 107, 109, 211, 5734, 140, 634, 333, 109, 211, 5602, 558, 143, 384, 390, 135, 109, 24772, 113, 16020, 132, 7755, 2495, 110, 158, 110, 107, 110, 144, 116, 2130, 140, 10129, 110, 108, 130, 6985, 20291, 28145, 2889, 2495, 173, 110, 144, 116, 2130, 1099, 195, 902, 197, 4567, 110, 107, 134, 109, 211, 5602, 558, 143, 396, 305, 110, 206, 13757, 110, 158, 110, 108, 114, 1458, 2630, 140, 3686, 118, 6214, 8610, 269, 109, 692, 791, 140, 547, 110, 107, 1044, 3243, 3309, 5602, 4397, 118, 791, 111, 2843, 110, 206, 111, 2491, 118, 857, 233, 164, 4397, 134, 396, 1265, 162, 953, 114, 573, 1312, 4712, 110, 107, 573, 1458, 2664, 111, 110, 144, 116, 2130, 195, 479, 290, 296, 899, 110, 108, 111, 435, 280, 899, 244, 289, 791, 143, 396, 1265, 110, 158, 110, 107, 573, 6214, 2843, 143, 110, 55654, 110, 108, 13204, 110, 68545, 10124, 110, 108, 920, 8207, 5888, 63589, 2664, 110, 108, 1955, 10420, 110, 108, 110, 144, 116, 2130, 110, 108, 13204, 2889, 110, 108, 916, 2889, 7864, 1865, 110, 108, 1108, 2201, 22500, 110, 108, 695, 1458, 2201, 2664, 122, 13945, 110, 108, 53342, 2664, 110, 108, 111, 13204, 38604, 8472, 110, 158, 195, 479, 134, 396, 305, 111, 134, 396, 1265, 143, 370, 113, 692, 110, 158, 110, 107, 9361, 702, 195, 9068, 134, 276, 5602, 558, 430, 692, 4052, 132, 9644, 110, 108, 111, 333, 109, 677, 390, 244, 109, 289, 692, 791, 110, 107, 110, 55654, 804, 602, 195, 1848, 130, 1021, 110, 108, 8812, 110, 108, 111, 499, 224, 149, 665, 899, 110, 107, 3850, 317, 495, 113, 110, 55654, 476, 195, 266, 317, 109, 13757, 110, 55654, 111, 110, 55654, 1099, 115, 109, 645, 899, 110, 108, 303, 110, 144, 233, 804, 110, 107, 114, 7286, 30033, 113, 891, 110, 105, 38217, 140, 263, 115, 109, 1382, 110, 107, 149, 11019, 195, 2303, 303, 10446, 116, 824, 48612, 143, 10446, 116, 12189, 26805, 110, 108, 439, 415, 110, 108, 38080, 110, 108, 20399, 110, 158, 110, 107, 4248, 233, 668, 1044, 143, 1689, 652, 111, 608, 1024, 110, 158, 195, 3352, 110, 108, 57804, 110, 108, 111, 953, 115, 109, 692, 110, 206, 153, 1021, 779, 143, 971, 28593, 110, 108, 1126, 110, 116, 252, 110, 1100, 110, 158, 140, 8282, 231, 143, 305, 28321, 231, 110, 158, 110, 107, 16020, 6005, 992, 112, 109, 1708, 1695, 111, 953, 142, 42559, 51502, 326, 110, 108, 15034, 110, 108, 1035, 40387, 110, 108, 40870, 71038, 22004, 110, 108, 281, 233, 5734, 175, 3106, 2073, 22004, 110, 108, 39051, 67027, 4191, 110, 108, 39051, 30012, 4191, 110, 108, 110, 8884, 64794, 110, 108, 111, 536, 110, 107, 223, 113, 109, 953, 1044, 196, 153, 16020, 791, 130, 453, 121, 132, 776, 233, 540, 2495, 110, 107, 1044, 4456, 110, 108, 330, 779, 110, 108, 1708, 10497, 110, 108, 111, 1371, 79917, 791, 127, 32807, 115, 826, 305, 110, 107, 156, 1532, 2342, 333, 109, 692, 135, 169, 10497, 143, 244, 396, 280, 110, 158, 110, 108, 111, 668, 1044, 31322, 135, 109, 692, 262, 113, 14304, 143, 339, 244, 396, 296, 110, 108, 111, 228, 244, 396, 384, 110, 158, 110, 107, 42402, 143, 624, 95108, 110, 158, 1044, 1413, 114, 2119, 113, 339, 3332, 110, 108, 738, 143, 530, 72576, 110, 158, 1413, 2899, 3332, 110, 108, 111, 1265, 143, 371, 95108, 110, 158, 1413, 149, 7778, 2771, 3309, 3332, 110, 107, 130, 684, 115, 826, 280, 110, 108, 109, 1021, 110, 55654, 476, 113, 109, 1182, 1044, 134, 13757, 140, 56492, 3957, 943, 110, 20815, 143, 8812, 110, 108, 110, 36696, 3957, 943, 110, 20815, 110, 206, 499, 110, 108, 56103, 3957, 943, 110, 20815, 7764, 2507, 3957, 943, 110, 20815, 110, 158, 110, 107, 118, 109, 738, 1044, 170, 1413, 134, 583, 2899, 3332, 110, 108, 109, 1021, 411, 115, 153, 110, 55654, 476, 140, 17908, 3957, 943, 110, 20815, 143, 8812, 110, 108, 12474, 3957, 943, 110, 20815, 110, 206, 499, 110, 108, 26418, 3957, 943, 110, 20815, 112, 19173, 3957, 943, 110, 20815, 110, 158, 110, 107, 118, 109, 1265, 1044, 170, 1413, 109, 664, 791, 908, 143, 665, 899, 110, 158, 110, 108, 109, 1021, 110, 55654, 476, 411, 140, 16537, 3957, 943, 110, 20815, 143, 8812, 110, 108, 12622, 3957, 943, 110, 20815, 110, 206, 499, 110, 108, 18524, 3957, 943, 110, 20815, 112, 27407, 3957, 943, 110, 20815, 110, 206, 891, 3092, 26652, 11161, 110, 158, 110, 107, 1965, 143, 384, 69468, 110, 158, 113, 109, 1925, 1044, 170, 1413, 134, 583, 339, 2889, 60760, 196, 114, 154, 197, 305, 3957, 943, 110, 20815, 815, 115, 153, 110, 55654, 476, 110, 107, 58117, 476, 852, 118, 109, 1265, 1044, 170, 1413, 7778, 2889, 60760, 127, 1673, 115, 1868, 305, 110, 107, 220, 53301, 2889, 233, 985, 9361, 702, 195, 1668, 790, 1044, 333, 109, 692, 132, 109, 857, 233, 164, 908, 110, 107, 110, 144, 116, 2130, 140, 10129, 333, 109, 692, 908, 110, 108, 111, 220, 1044, 196, 110, 144, 116, 2130, 1099, 815, 112, 154, 197, 4567, 110, 107, 109, 1330, 110, 68545, 10124, 476, 790, 1044, 170, 1413, 134, 583, 2899, 53301, 2889, 3332, 140, 2628, 21178, 110, 4652, 943, 16342, 110, 206, 109, 1021, 476, 134, 109, 370, 113, 692, 908, 118, 109, 664, 456, 140, 110, 41949, 110, 4652, 943, 16342, 110, 107, 668, 143, 280, 72576, 110, 158, 1044, 915, 1458, 49334, 116, 111, 195, 1341, 791, 10114, 143, 339, 244, 396, 296, 110, 108, 7662, 69450, 134, 110, 55654, 1099, 113, 56103, 3957, 943, 110, 20815, 110, 108, 46131, 3957, 943, 110, 20815, 110, 108, 111, 34423, 3957, 943, 110, 20815, 110, 206, 156, 244, 396, 384, 110, 108, 7662, 69450, 134, 142, 110, 55654, 476, 113, 43362, 3957, 943, 110, 20815, 110, 206, 111, 156, 244, 396, 950, 110, 108, 7662, 69450, 134, 142, 110, 55654, 476, 113, 35700, 3957, 943, 110, 20815, 110, 158, 110, 107, 580, 110, 55654, 1099, 127, 1589, 122, 20764, 15593, 6045, 111, 2570, 8408, 1380, 5690, 107, 522, 1147, 791, 113, 32680, 148, 25839, 995, 118, 1044, 110, 108, 432, 29755, 5110, 46195, 2757, 110, 107, 1670, 109, 868, 113, 110, 24397, 116, 117, 210, 233, 1614, 115, 6108, 9073, 304, 110, 108, 461, 2084, 195, 938, 2244, 160, 109, 2404, 1298, 113, 110, 24397, 116, 124, 5690, 115, 181, 1044, 122, 1695, 107, 43381, 2084, 160, 109, 887, 118, 89194, 55002, 13486, 208, 115, 1044, 122, 1695, 122, 902, 110, 55654, 1099, 170, 127, 2886, 110, 24397, 195, 163, 5145, 115, 223, 6316, 107, 8101, 108, 8791, 115, 663, 110, 108, 109, 433, 89266, 261, 1521, 113, 1458, 448, 49334, 116, 120, 218, 133, 12028, 112, 17574, 71899, 8973, 195, 5145, 269, 107, 6304, 110, 108, 1182, 115, 150, 4947, 692, 110, 108, 145, 3174, 109, 17890, 113, 303, 2889, 27654, 1600, 112, 2029, 32680, 115, 1044, 122, 1695, 170, 127, 12857, 16020, 347, 109, 207, 113, 110, 24397, 116, 132, 1458, 49334, 110, 108, 162, 256, 129, 114, 3538, 2049, 110, 108, 704, 118, 1044, 122, 75456, 15791, 110, 107, 4868, 2889, 117, 1294, 112, 15079, 111, 3074, 7509, 110, 108, 155, 580, 1532, 22118, 110, 108, 2111, 1646, 1114, 12126, 110, 108, 111, 2111, 11118, 262, 113, 114, 827, 499, 113, 23908, 110, 26889, 9361, 1521, 2516, 203, 1380, 5800, 107, 9965, 32680, 113, 4906, 1568, 218, 2902, 115, 1044, 122, 1695, 111, 117, 1589, 122, 142, 815, 115, 178, 14135, 72118, 1099, 110, 108, 162, 14719, 4868, 2889, 12126, 111, 4499, 25030, 2889, 207, 110, 108, 110, 88016, 189, 433, 1298, 113, 1209, 14485, 113, 4868, 2889, 107, 4262, 53301, 2889, 2495, 2838, 7997, 1407, 112, 860, 83713, 386, 110, 49984, 173, 1711, 122, 4868, 2889, 132, 220, 2889, 115, 142, 30678, 1044, 122, 1695, 170, 127, 2886, 16020, 107, 6113, 7090, 4868, 2889, 6683, 122, 110, 24397, 116, 2375, 220, 1225, 1280, 204, 110, 24397, 116, 1600, 115, 6108, 9073, 304, 107, 7090, 11660, 28910, 11026, 76168, 19282, 111, 2889, 74900, 1699, 112, 133, 154, 11123, 1008, 5771, 204, 2889, 110, 23233, 56057, 110, 107, 114, 423, 6667, 1008, 3850, 2498, 3004, 112, 403, 1651, 142, 304, 43947, 14500, 16339, 7557, 108, 9613, 162, 117, 3542, 115, 150, 692, 110, 108, 115, 162, 220, 1044, 1184, 7557, 111, 220, 1044, 31322, 135, 109, 692, 262, 113, 9361, 1521, 110, 107, 634, 120, 109, 1021, 110, 55654, 815, 303, 110, 24397, 116, 122, 53301, 2889, 115, 156, 423, 3922, 2498, 140, 14163, 3957, 943, 110, 20815, 108, 7090, 109, 602, 3686, 115, 150, 692, 127, 21619, 1225, 110, 107, 219, 4469, 246, 129, 701, 3542, 111, 340, 9068, 115, 1599, 1683, 110, 108, 115, 162, 574, 253, 130, 109, 5520, 6002, 113, 53301, 2889, 2495, 122, 2132, 112, 16020, 111, 109, 5520, 916, 5734, 113, 53301, 2889, 246, 129, 3035, 110, 107, 109, 207, 113, 53301, 2889, 11325, 19206, 140, 938, 5216, 141, 114, 456, 115, 42344, 120, 4525, 109, 207, 113, 28910, 11026, 16262, 20279, 15067, 144, 17017, 112, 2555, 110, 24397, 111, 1458, 49334, 116, 130, 114, 791, 118, 9073, 304, 110, 107, 2889, 233, 31835, 1044, 2839, 122, 28910, 11026, 16262, 20279, 15067, 144, 17017, 1600, 143, 3178, 3092, 46440, 110, 158, 196, 114, 8812, 113, 13602, 3957, 943, 110, 20815, 815, 115, 58117, 1099, 1711, 122, 274, 2886, 853, 791, 122, 110, 24397, 116, 143, 3178, 3092, 7052, 110, 206, 8812, 110, 108, 13264, 3957, 943, 110, 20815, 110, 158, 110, 107, 150, 692, 110, 108, 802, 110, 108, 117, 21318, 115, 303, 2889, 2495, 115, 114, 609, 233, 2889, 233, 15642, 449, 107, 9169, 2889, 19591, 244, 53301, 2889, 2495, 110, 108, 122, 866, 2084, 160, 109, 887, 113, 1690, 4367, 15791, 111, 4917, 110, 108, 382, 129, 2244, 110, 107, 109, 1330, 13204, 110, 68545, 10124, 476, 115, 109, 799, 692, 115, 1044, 170, 1413, 134, 583, 950, 899, 113, 53301, 2889, 2495, 140, 2628, 21178, 110, 4652, 943, 16342, 110, 107, 205, 113, 109, 4413, 6790, 1695, 111, 7899, 115, 2889, 233, 42802, 1044, 472, 135, 1044, 122, 178, 5196, 77785, 15219, 132, 1044, 170, 127, 12857, 82560, 110, 107, 1299, 1558, 731, 142, 815, 115, 91198, 28193, 209, 115, 1044, 122, 178, 5196, 77785, 15219, 244, 157, 1070, 68615, 107, 10340, 10808, 335, 2518, 109, 4097, 317, 53301, 2889, 2495, 111, 902, 4917, 872, 127, 5404, 111, 146, 210, 233, 2394, 107, 4311, 115, 617, 110, 108, 32680, 1110, 117, 114, 887, 2634, 118, 7899, 115, 1044, 2886, 82560, 107, 10822, 114, 63762, 1382, 113, 8144, 317, 2889, 111, 12439, 115, 154, 197, 371, 16322, 1044, 2886, 82560, 1668, 220, 1562, 1323, 872, 135, 13204, 110, 68545, 10124, 1099, 130, 281, 130, 20987, 110, 4652, 943, 16342, 107, 4311, 109, 2186, 519, 113, 2495, 115, 1044, 122, 1695, 117, 113, 11537, 2991, 110, 108, 162, 256, 129, 142, 853, 1280, 113, 53301, 2889, 204, 109, 207, 113, 110, 24397, 116, 115, 253, 1044, 110, 107, 112, 701, 845, 223, 113, 109, 574, 2244, 110, 108, 150, 320, 117, 1062, 114, 2660, 2498, 118, 53301, 2889, 115, 1044, 122, 1695, 170, 133, 32680, 112, 4480, 109, 602, 3552, 115, 136, 4947, 2498, 110, 107, 115, 663, 110, 108, 145, 138, 129, 383, 190, 58572, 113, 1407, 112, 53301, 2889, 110, 108, 253, 130, 13204, 178, 14135, 72118, 476, 110, 107, 53301, 2889, 2495, 1600, 117, 963, 111, 218, 129, 957, 115, 3024, 110, 55654, 1099, 115, 1044, 122, 1695, 170, 127, 12857, 1371, 79917, 2495, 110, 107, 701, 24374, 6316, 127, 690, 112, 845, 223, 113, 109, 574, 2244, 115, 150, 4947, 692, 110, 107, 1], [15962, 39237, 35368, 144, 16865, 143, 110, 144, 252, 110, 158, 110, 108, 114, 57070, 477, 1298, 244, 895, 3411, 112, 76004, 116, 110, 108, 117, 10592, 141, 391, 132, 956, 110, 108, 8123, 110, 108, 42245, 26357, 113, 114, 3526, 132, 3526, 456, 110, 108, 122, 23495, 5573, 110, 108, 1813, 2283, 110, 108, 162, 218, 2384, 109, 18825, 110, 108, 11850, 110, 108, 3464, 110, 108, 132, 749, 110, 107, 110, 144, 252, 148, 174, 1673, 112, 1070, 115, 160, 15418, 113, 1044, 170, 133, 196, 300, 233, 1286, 3411, 112, 76004, 116, 110, 107, 110, 107, 109, 580, 887, 113, 110, 144, 252, 118, 50581, 76004, 116, 117, 666, 112, 711, 135, 153, 5404, 22685, 118, 34641, 22293, 110, 107, 1711, 122, 2953, 110, 108, 50581, 76004, 3073, 133, 114, 1626, 22685, 118, 33668, 6604, 24371, 522, 304, 197, 34641, 3138, 522, 22293, 110, 108, 122, 114, 580, 39147, 112, 20598, 110, 144, 252, 110, 107, 790, 136, 83931, 117, 666, 112, 133, 38492, 918, 134, 49419, 83531, 2288, 204, 19226, 20019, 60690, 9550, 34641, 20890, 2288, 12252, 111, 117, 110, 108, 1923, 110, 108, 1589, 122, 114, 221, 580, 15111, 113, 911, 9662, 40299, 14644, 16569, 143, 110, 42843, 110, 158, 110, 107, 21905, 110, 108, 114, 25754, 1382, 113, 3922, 1546, 12948, 6316, 3498, 120, 83931, 163, 7997, 66998, 2775, 113, 15962, 39237, 5573, 110, 107, 145, 731, 114, 437, 113, 9559, 1019, 233, 459, 3719, 110, 108, 10857, 112, 1074, 32021, 755, 110, 108, 8115, 164, 112, 280, 971, 110, 108, 1848, 122, 3726, 46532, 35368, 31897, 518, 22178, 3464, 5573, 1126, 1868, 305, 110, 1100, 110, 107, 3794, 689, 6031, 3264, 178, 140, 646, 110, 10885, 36924, 19464, 280, 5111, 77651, 118, 280, 590, 111, 237, 83931, 371, 5111, 118, 372, 384, 590, 110, 107, 1082, 113, 3464, 35368, 144, 16865, 113, 1532, 134, 1925, 231, 110, 108, 109, 1532, 1848, 122, 7635, 7630, 59854, 110, 108, 509, 40026, 124, 360, 2887, 111, 8231, 328, 549, 110, 108, 850, 429, 135, 238, 110, 108, 15114, 6902, 5746, 110, 108, 23632, 1759, 110, 108, 1756, 22041, 118, 280, 390, 110, 206, 162, 140, 73851, 244, 9817, 110, 107, 992, 112, 109, 1499, 110, 108, 156, 1151, 382, 133, 8926, 943, 266, 546, 113, 342, 589, 111, 244, 120, 1532, 3135, 313, 165, 113, 480, 110, 108, 111, 3597, 607, 2137, 2775, 110, 107, 136, 140, 15186, 130, 45949, 122, 446, 4464, 7457, 15866, 675, 110, 108, 111, 178, 140, 2839, 122, 110, 10885, 36924, 19464, 280, 5111, 943, 242, 118, 280, 590, 111, 237, 122, 83931, 371, 5111, 943, 242, 118, 384, 590, 110, 107, 115, 289, 228, 857, 233, 164, 116, 1532, 368, 146, 799, 1847, 110, 108, 111, 1499, 1668, 4694, 3464, 5573, 110, 108, 162, 195, 784, 130, 114, 297, 113, 169, 1380, 43022, 6671, 111, 146, 784, 3415, 110, 108, 6522, 24215, 3464, 5573, 1668, 195, 2987, 130, 297, 113, 22214, 2764, 743, 50497, 181, 6555, 115, 2959, 110, 107, 130, 3464, 35368, 144, 16865, 1562, 110, 108, 109, 1532, 196, 114, 3726, 5907, 130, 1532, 196, 112, 376, 169, 1233, 893, 169, 693, 118, 109, 337, 110, 107, 109, 1815, 192, 5148, 173, 109, 1532, 140, 7538, 308, 111, 140, 12001, 333, 1756, 110, 107, 178, 254, 3135, 646, 425, 640, 112, 3726, 3464, 5573, 395, 19728, 111, 33384, 1011, 110, 107, 169, 2755, 111, 616, 12112, 17193, 195, 1644, 110, 107, 333, 19496, 231, 113, 779, 1532, 140, 115, 44216, 16778, 111, 26364, 76629, 110, 107, 176, 11243, 195, 8115, 122, 2080, 1034, 116, 1393, 110, 108, 111, 1532, 140, 163, 1406, 112, 399, 110, 108, 155, 640, 112, 115, 65167, 111, 69433, 110, 108, 178, 368, 146, 1566, 280, 971, 244, 339, 5086, 110, 107, 178, 518, 109, 17176, 110, 107, 122, 1077, 2735, 7233, 111, 271, 766, 110, 108, 178, 947, 130, 142, 63134, 75762, 115, 109, 3153, 3067, 130, 114, 1360, 561, 110, 107, 110, 108, 178, 140, 374, 112, 129, 509, 204, 2611, 110, 108, 16373, 110, 108, 39801, 110, 108, 111, 613, 26597, 110, 107, 1254, 110, 108, 109, 1532, 196, 13167, 525, 6799, 110, 206, 118, 162, 169, 594, 266, 546, 113, 342, 110, 108, 111, 38543, 342, 110, 107, 124, 2287, 1932, 2843, 110, 108, 3337, 9051, 110, 108, 16331, 1434, 2749, 110, 108, 13401, 79150, 14506, 110, 108, 2617, 1579, 4712, 110, 108, 22085, 1026, 233, 20610, 110, 108, 7214, 20003, 7705, 2037, 195, 1644, 110, 107, 244, 4985, 12263, 81170, 110, 108, 110, 62801, 1034, 116, 1568, 111, 176, 4367, 2791, 113, 35368, 144, 16865, 195, 8258, 165, 110, 107, 109, 1532, 140, 2839, 122, 57745, 305, 5111, 916, 17722, 1907, 143, 110, 144, 252, 116, 110, 158, 110, 108, 35616, 15002, 43933, 1182, 5111, 110, 144, 252, 116, 110, 108, 9100, 55492, 457, 30209, 4915, 7760, 280, 5111, 23349, 6006, 143, 110, 28794, 110, 158, 110, 107, 244, 280, 590, 110, 108, 186, 140, 181, 2757, 113, 279, 7732, 110, 107, 13160, 33115, 22872, 377, 5111, 140, 717, 110, 206, 1562, 164, 112, 599, 5111, 110, 108, 9100, 55492, 415, 30209, 4915, 7760, 2785, 112, 280, 5111, 110, 107, 122, 332, 2757, 244, 384, 590, 113, 791, 118, 35368, 144, 16865, 110, 108, 7189, 53935, 32101, 1877, 439, 20301, 32101, 143, 1061, 1877, 1182, 110, 158, 140, 717, 141, 12263, 81170, 111, 1562, 164, 112, 4806, 110, 144, 252, 116, 111, 13160, 33115, 22872, 28902, 110, 107, 244, 665, 590, 113, 791, 110, 108, 1532, 148, 2521, 279, 8895, 122, 35616, 15002, 43933, 4781, 5111, 110, 108, 7189, 53935, 32101, 1877, 439, 20301, 32101, 143, 1061, 1877, 1182, 110, 158, 233, 4806, 110, 28794, 110, 108, 111, 57745, 305, 5111, 110, 28794, 110, 107, 1678, 437, 1574, 1668, 110, 144, 252, 1690, 122, 281, 233, 5734, 50581, 76004, 116, 253, 130, 83931, 599, 5111, 132, 114, 4278, 68441, 551, 54040, 738, 5111, 122, 895, 5400, 113, 3411, 113, 279, 665, 4262, 590, 115, 1614, 20140, 4222, 172, 28603, 132, 189, 176, 49451, 4222, 110, 107, 110, 42843, 115, 956, 111, 15962, 39237, 35368, 44121, 13856, 110, 108, 115, 970, 110, 108, 133, 174, 9730, 4525, 115, 28603, 110, 107, 254, 577, 114, 344, 113, 1683, 2298, 120, 23349, 1044, 306, 902, 1372, 113, 110, 42843, 143, 1669, 40380, 2675, 110, 108, 35368, 144, 16865, 110, 108, 15675, 4775, 2396, 110, 158, 111, 110, 144, 252, 1711, 112, 1044, 122, 114, 5220, 113, 28603, 110, 108, 473, 373, 109, 110, 28794, 1948, 148, 174, 1250, 110, 107, 109, 887, 117, 374, 112, 129, 296, 112, 371, 488, 902, 115, 6979, 1044, 1711, 112, 758, 1044, 110, 107, 115, 663, 112, 779, 110, 108, 109, 887, 117, 1072, 27489, 112, 110, 151, 2912, 4336, 110, 108, 1036, 111, 916, 5734, 113, 109, 76004, 2108, 110, 108, 2210, 113, 4301, 6006, 110, 108, 109, 207, 113, 1857, 88917, 116, 122, 12263, 36533, 19473, 110, 108, 1331, 1312, 10648, 143, 13401, 93308, 2495, 110, 158, 110, 108, 109, 2210, 113, 176, 1312, 4222, 253, 130, 5040, 132, 142, 2554, 6006, 110, 108, 3629, 779, 113, 3411, 110, 108, 111, 109, 2210, 113, 911, 9662, 40299, 14644, 2775, 616, 115, 791, 110, 107, 136, 1532, 1034, 116, 3726, 35368, 31897, 3464, 5573, 1184, 373, 613, 908, 113, 530, 590, 113, 3411, 112, 50581, 76004, 116, 110, 10885, 36924, 19464, 280, 5111, 111, 237, 83931, 371, 5111, 209, 110, 108, 162, 137, 1007, 4513, 911, 9662, 40299, 14644, 477, 1521, 110, 107, 115, 136, 437, 110, 108, 887, 1958, 118, 1690, 1651, 27217, 110, 144, 252, 195, 12263, 36533, 8207, 3411, 110, 108, 39205, 5272, 7233, 110, 108, 2900, 4944, 2764, 110, 108, 18222, 95959, 110, 108, 111, 9101, 616, 8933, 113, 477, 1521, 110, 107, 136, 437, 4461, 11558, 113, 27806, 31328, 580, 5734, 453, 2233, 76004, 116, 115, 1532, 122, 8945, 92875, 1152, 28975, 111, 39205, 5272, 7233, 122, 15129, 613, 4894, 6632, 6337, 110, 206, 864, 86049, 130, 45949, 110, 206, 964, 112, 253, 3726, 9361, 1521, 262, 1044, 122, 2554, 2037, 1303, 127, 154, 9539, 112, 1070, 9361, 1521, 172, 110, 144, 252, 110, 107, 2297, 110, 108, 69094, 207, 113, 76004, 116, 110, 108, 122, 2067, 111, 5030, 8610, 117, 356, 110, 108, 111, 39979, 32453, 252, 2764, 132, 40910, 5573, 355, 129, 8703, 2679, 111, 784, 3415, 110, 107, 109, 3802, 29183, 120, 157, 133, 3686, 149, 1530, 1532, 4898, 1878, 110, 107, 115, 109, 515, 109, 1532, 741, 116, 110, 158, 148, 943, 133, 634, 169, 943, 215, 943, 153, 4898, 118, 169, 943, 215, 943, 153, 1055, 111, 176, 2827, 257, 112, 129, 1668, 115, 109, 4819, 110, 107, 109, 1044, 630, 120, 153, 1964, 111, 38506, 138, 146, 129, 1299, 111, 640, 1645, 138, 129, 266, 112, 23717, 153, 3149, 110, 108, 155, 23304, 137, 146, 129, 3891, 110, 107, 109, 3802, 29183, 120, 157, 133, 3686, 149, 1530, 1532, 4898, 1878, 110, 107, 115, 109, 515, 109, 1532, 741, 116, 110, 158, 148, 943, 133, 634, 169, 943, 215, 943, 153, 4898, 118, 169, 943, 215, 943, 153, 1055, 111, 176, 2827, 257, 112, 129, 1668, 115, 109, 4819, 110, 107, 109, 1044, 630, 120, 153, 1964, 111, 38506, 138, 146, 129, 1299, 111, 640, 1645, 138, 129, 266, 112, 23717, 153, 3149, 110, 108, 155, 23304, 137, 146, 129, 3891, 110, 107, 1]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'labels': [[110, 105, 283, 2314, 1688, 1321, 23722, 115, 1044, 122, 1695, 170, 127, 12857, 1371, 2495, 117, 3732, 8674, 111, 218, 27811, 348, 113, 271, 115, 219, 1044, 110, 107, 109, 1298, 113, 1458, 49334, 117, 432, 4274, 111, 218, 129, 1589, 122, 1651, 9361, 702, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 110, 8723, 34764, 20547, 34261, 233, 13568, 3073, 127, 146, 957, 115, 7732, 26588, 113, 1044, 111, 218, 133, 114, 2404, 1298, 124, 1380, 5690, 107, 54867, 116, 497, 4676, 109, 14376, 111, 17890, 113, 34357, 2889, 2495, 115, 1044, 122, 1695, 170, 133, 609, 233, 2889, 233, 15642, 32680, 111, 170, 127, 12857, 791, 122, 16020, 347, 109, 207, 113, 110, 8723, 34764, 20547, 34261, 233, 13568, 3073, 107, 36005, 116, 30551, 1044, 122, 1907, 15791, 111, 609, 233, 2889, 233, 15642, 32680, 195, 953, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 28910, 11026, 74900, 134, 114, 5734, 113, 2416, 5111, 140, 634, 115, 613, 34357, 60760, 3309, 118, 114, 916, 113, 665, 899, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 58117, 476, 140, 5844, 134, 13757, 110, 108, 290, 296, 899, 110, 108, 111, 280, 899, 244, 109, 289, 2889, 20718, 143, 396, 1265, 110, 158, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 9361, 702, 985, 112, 34357, 2889, 195, 6667, 445, 1668, 107, 56131, 1313, 1182, 1044, 953, 110, 108, 1925, 143, 624, 95108, 110, 158, 1413, 134, 583, 339, 2889, 60760, 111, 1], [110, 105, 283, 2314, 15962, 39237, 35368, 144, 16865, 143, 110, 144, 252, 110, 158, 117, 114, 1651, 477, 1298, 113, 76004, 6098, 110, 108, 154, 122, 2953, 76004, 116, 110, 108, 120, 117, 3744, 38483, 115, 2790, 1044, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 1683, 403, 120, 7340, 50581, 76004, 116, 133, 114, 1074, 887, 113, 110, 144, 252, 110, 107, 130, 114, 711, 110, 108, 223, 17869, 218, 133, 1184, 114, 4797, 1083, 113, 750, 173, 31328, 219, 6098, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 145, 731, 114, 437, 113, 9559, 1019, 233, 459, 3719, 122, 8945, 92875, 1152, 28975, 111, 39205, 5272, 7233, 110, 108, 170, 1184, 3726, 110, 144, 252, 244, 580, 5734, 613, 5400, 3411, 112, 50581, 76004, 110, 10885, 36924, 19464, 111, 237, 83931, 110, 107, 110, 105, 191, 283, 2314, 110, 105, 283, 2314, 109, 1000, 113, 136, 800, 117, 112, 6034, 109, 3679, 112, 129, 69094, 111, 13740, 269, 303, 5048, 580, 5734, 453, 2233, 76004, 116, 115, 1532, 122, 220, 1962, 49451, 556, 110, 108, 8945, 92875, 1152, 28975, 110, 108, 132, 39205, 5272, 7233, 49044, 113, 2554, 2037, 1303, 110, 108, 170, 127, 154, 9539, 112, 1070, 9361, 1521, 253, 130, 110, 144, 252, 111, 3028, 109, 16121, 113, 110, 144, 252, 115, 1044, 646, 50581, 76004, 116, 110, 107, 110, 105, 191, 283, 2314, 1]]}"
      ]
     },
     "execution_count": 21,
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
   "execution_count": 22,
   "metadata": {
    "id": "DDtsaJeVIrJT"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b79c6867bf4947ad961d1d2853139559",
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
       "model_id": "bd1ddf7a2bfe488a8dada48501261536",
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
       "model_id": "7c993afb5ffd4c1e9de9a29ad58000e3",
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
   "execution_count": 23,
   "metadata": {
    "id": "TlqNaB8jIrJW"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "523bf03f03ab4e0fb01452114bb3e6d1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/2.15G [00:00<?, ?B/s]"
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
   "execution_count": 24,
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
    "    fp16 = True,\n",
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
   "execution_count": 25,
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
   "execution_count": 26,
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
   "execution_count": 27,
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
      "Cloning https://huggingface.co/Kevincp560/bigbird-pegasus-large-arxiv-finetuned-pubmed into local empty directory.\n"
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
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using amp half precision backend\n"
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
   "execution_count": 28,
   "metadata": {
    "id": "uNx5pyRlIrJh"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "The following columns in the training set  don't have a corresponding argument in `BigBirdPegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `BigBirdPegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning\n",
      "  warnings.warn(\n",
      "***** Running training *****\n",
      "  Num examples = 2000\n",
      "  Num Epochs = 5\n",
      "  Instantaneous batch size per device = 2\n",
      "  Total train batch size (w. parallel, distributed & accumulation) = 4\n",
      "  Gradient Accumulation steps = 1\n",
      "  Total optimization steps = 2500\n",
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.\n",
      "To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  /opt/conda/conda-bld/pytorch_1631630839582/work/aten/src/ATen/native/BinaryOps.cpp:467.)\n",
      "  return torch.floor_divide(self, other)\n",
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.\n",
      "  warnings.warn('Was asked to gather along dimension 0, but all '\n",
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/transformers/trainer.py:1443: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.\n",
      "  nn.utils.clip_grad_norm_(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='2500' max='2500' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [2500/2500 2:12:46, Epoch 5/5]\n",
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
       "      <td>2.594000</td>\n",
       "      <td>1.987911</td>\n",
       "      <td>33.636400</td>\n",
       "      <td>13.507400</td>\n",
       "      <td>21.428600</td>\n",
       "      <td>29.715800</td>\n",
       "      <td>189.014000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>1.914600</td>\n",
       "      <td>1.649374</td>\n",
       "      <td>44.005600</td>\n",
       "      <td>19.006900</td>\n",
       "      <td>27.514200</td>\n",
       "      <td>40.049200</td>\n",
       "      <td>210.528000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>1.737800</td>\n",
       "      <td>1.621283</td>\n",
       "      <td>44.707100</td>\n",
       "      <td>19.355900</td>\n",
       "      <td>27.680600</td>\n",
       "      <td>40.612400</td>\n",
       "      <td>213.596000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1.692000</td>\n",
       "      <td>1.608123</td>\n",
       "      <td>45.150500</td>\n",
       "      <td>19.735500</td>\n",
       "      <td>28.060000</td>\n",
       "      <td>41.010800</td>\n",
       "      <td>213.674000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>1.665600</td>\n",
       "      <td>1.604914</td>\n",
       "      <td>45.480700</td>\n",
       "      <td>20.019900</td>\n",
       "      <td>28.362100</td>\n",
       "      <td>41.461800</td>\n",
       "      <td>219.144000</td>\n",
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
      "Input ids are automatically padded from 2635 to 2688 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2635 to 2688 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2699 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2699 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4075 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4075 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3241 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3241 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3346 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3346 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3104 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3104 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3620 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3620 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3587 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3587 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2745 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2745 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4015 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4015 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1690 to 1728 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1690 to 1728 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3020 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3020 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3690 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3690 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4038 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4038 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3809 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3809 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4016 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4016 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3468 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3468 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3979 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3979 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3539 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3539 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4067 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4067 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3915 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3915 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3425 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3425 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3962 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3962 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3750 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3750 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3752 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3752 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3812 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3812 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3877 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3877 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2968 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2968 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2768 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2768 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3211 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3211 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3825 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3825 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3849 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3849 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3164 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3164 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2991 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2991 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3775 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3775 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4026 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4026 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2571 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2571 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3257 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3257 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3999 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3999 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3695 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3695 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3790 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3790 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2585 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2585 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3859 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3859 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4065 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4065 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3408 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3408 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2968 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2968 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4050 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4050 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3716 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3716 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1886 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1886 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4080 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4080 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3547 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3547 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2241 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2241 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2933 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2933 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3118 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3118 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2408 to 2432 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2408 to 2432 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3331 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3331 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2919 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2919 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3929 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3929 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2531 to 2560 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2531 to 2560 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2352 to 2368 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2352 to 2368 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3818 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3818 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3983 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3983 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2413 to 2432 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2413 to 2432 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3370 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3370 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2617 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2617 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3794 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3794 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2695 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2695 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3549 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3549 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3677 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3677 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3208 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3208 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3465 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3465 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3022 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3022 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3736 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3736 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4091 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4091 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3948 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3948 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2191 to 2240 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2191 to 2240 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3741 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3741 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3247 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3247 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3458 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3458 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3329 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3329 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3724 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3724 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2634 to 2688 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2634 to 2688 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3397 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3397 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Saving model checkpoint to bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-500\n",
      "Configuration saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-500/config.json\n",
      "Model weights saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-500/pytorch_model.bin\n",
      "tokenizer config file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-500/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-500/special_tokens_map.json\n",
      "tokenizer config file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/special_tokens_map.json\n"
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
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "The following columns in the evaluation set  don't have a corresponding argument in `BigBirdPegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `BigBirdPegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 4\n"
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
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.\n",
      "  warnings.warn('Was asked to gather along dimension 0, but all '\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3570 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3570 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2671 to 2688 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2671 to 2688 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4012 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4012 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3134 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3134 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3721 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3721 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3489 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3489 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3341 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3341 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4090 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4090 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3750 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3750 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3930 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3930 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1576 to 1600 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1576 to 1600 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3303 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3303 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3935 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3935 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3262 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3262 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1849 to 1856 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1849 to 1856 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3708 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3708 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3563 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3563 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3051 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3051 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4080 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4080 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3053 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3053 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3234 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3234 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2359 to 2368 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2359 to 2368 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3807 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3807 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3964 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3964 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2732 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2732 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2989 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2989 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4087 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4087 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3455 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3455 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3471 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3471 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4065 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4065 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3587 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3587 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3785 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3785 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2251 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2251 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3886 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3886 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3812 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3812 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3979 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3979 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3921 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3921 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3518 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3518 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3692 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3692 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3259 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3259 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3052 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3052 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3999 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3999 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3468 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3468 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3171 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3171 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3211 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3211 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3170 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3170 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2698 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2698 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3919 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3919 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3485 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3485 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3296 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3296 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2947 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2947 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3047 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3047 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4068 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4068 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1996 to 2048 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1996 to 2048 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3978 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3978 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3812 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3812 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3896 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3896 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3312 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3312 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2672 to 2688 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2672 to 2688 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2478 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2478 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4028 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4028 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3250 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3250 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3041 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3041 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3118 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3118 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3071 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3071 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2477 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2477 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2869 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2869 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3241 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3241 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3861 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3861 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3367 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3367 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4050 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4050 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3465 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3465 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3969 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3969 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3775 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3775 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4067 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4067 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3229 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3229 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3642 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3642 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2887 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2887 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3936 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3936 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4095 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4095 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3863 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3863 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Saving model checkpoint to bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-1000\n",
      "Configuration saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-1000/config.json\n",
      "Model weights saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-1000/pytorch_model.bin\n",
      "tokenizer config file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-1000/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-1000/special_tokens_map.json\n"
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
      "tokenizer config file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/special_tokens_map.json\n"
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
      "The following columns in the evaluation set  don't have a corresponding argument in `BigBirdPegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `BigBirdPegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 4\n"
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
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.\n",
      "  warnings.warn('Was asked to gather along dimension 0, but all '\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2506 to 2560 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2506 to 2560 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2595 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2595 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4060 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4060 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3236 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3236 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3547 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3547 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3603 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3603 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3075 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3075 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2478 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2478 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3930 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3930 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3964 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3964 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3309 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3309 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3526 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3526 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3155 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3155 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1619 to 1664 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1619 to 1664 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3058 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3058 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2424 to 2432 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2424 to 2432 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2216 to 2240 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2216 to 2240 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3549 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3549 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3155 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3155 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2382 to 2432 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2382 to 2432 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3794 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3794 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3383 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3383 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3496 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3496 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3541 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3541 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3173 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3173 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3471 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3471 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3547 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3547 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3259 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3259 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2820 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2820 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2989 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2989 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3716 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3716 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4015 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4015 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3447 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3447 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3266 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3266 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3710 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3710 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3762 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3762 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2490 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2490 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4067 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4067 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3801 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3801 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4087 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4087 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3298 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3298 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3049 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3049 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3257 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3257 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2707 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2707 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3715 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3715 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4068 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4068 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1377 to 1408 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1377 to 1408 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4041 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4041 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2635 to 2688 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2635 to 2688 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3290 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3290 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3732 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3732 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3849 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3849 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2233 to 2240 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2233 to 2240 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2839 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2839 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3183 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3183 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3865 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3865 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2733 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2733 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3666 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3666 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3799 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3799 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1726 to 1728 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1726 to 1728 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3750 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3750 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3492 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3492 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3768 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3768 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3995 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3995 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2866 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2866 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3341 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3341 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3768 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3768 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3790 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3790 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3812 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3812 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3341 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3341 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3449 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3449 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4091 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4091 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3724 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3724 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3854 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3854 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2195 to 2240 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2195 to 2240 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3357 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3357 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3190 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3190 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2310 to 2368 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2310 to 2368 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2818 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2818 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3922 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3922 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2787 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2787 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3886 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3886 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3979 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3979 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3444 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3444 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3233 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3233 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2356 to 2368 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2356 to 2368 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3598 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3598 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3742 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3742 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2249 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2249 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3545 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3545 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Saving model checkpoint to bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-1500\n",
      "Configuration saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-1500/config.json\n",
      "Model weights saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-1500/pytorch_model.bin\n",
      "tokenizer config file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-1500/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-1500/special_tokens_map.json\n"
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
      "tokenizer config file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/special_tokens_map.json\n"
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
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "The following columns in the evaluation set  don't have a corresponding argument in `BigBirdPegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `BigBirdPegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 4\n"
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
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.\n",
      "  warnings.warn('Was asked to gather along dimension 0, but all '\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3465 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3465 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3969 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3969 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3827 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3827 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3290 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3290 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3967 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3967 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3710 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3710 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3514 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3514 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2698 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2698 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4015 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4015 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3850 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3850 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3335 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3335 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3836 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3836 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3481 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3481 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3979 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3979 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4012 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4012 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3821 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3821 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3348 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3348 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3458 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3458 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3620 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3620 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2370 to 2432 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2370 to 2432 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2708 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2708 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3257 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3257 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3962 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3962 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2997 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2997 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3424 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3424 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2565 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2565 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2584 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2584 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3589 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3589 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3496 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3496 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3878 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3878 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3852 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3852 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3039 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3039 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3983 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3983 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3785 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3785 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3724 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3724 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3474 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3474 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2280 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2280 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3417 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3417 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2947 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2947 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2933 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2933 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3863 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3863 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3535 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3535 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2082 to 2112 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2082 to 2112 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3104 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3104 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1146 to 1152 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1146 to 1152 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4028 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4028 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2511 to 2560 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2511 to 2560 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3514 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3514 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1491 to 1536 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1491 to 1536 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3725 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3725 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3698 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3698 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3902 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3902 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2618 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2618 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3485 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3485 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2524 to 2560 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2524 to 2560 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2188 to 2240 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2188 to 2240 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3818 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3818 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4068 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4068 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3346 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3346 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3624 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3624 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3031 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3031 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3930 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3930 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3155 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3155 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3294 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3294 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3455 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3455 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1491 to 1536 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1491 to 1536 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3993 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3993 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3300 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3300 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3935 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3935 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4067 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4067 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3183 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3183 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3512 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3512 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3936 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3936 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3475 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3475 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3128 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3128 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4067 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4067 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3571 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3571 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3312 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3312 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3262 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3262 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3178 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3178 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3309 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3309 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2145 to 2176 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2145 to 2176 to be a multiple of `config.block_size`: 64\n",
      "Saving model checkpoint to bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-2000\n",
      "Configuration saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-2000/config.json\n",
      "Model weights saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-2000/pytorch_model.bin\n",
      "tokenizer config file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-2000/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-2000/special_tokens_map.json\n"
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
      "tokenizer config file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/special_tokens_map.json\n"
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
      "Deleting older checkpoint [bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-500] due to args.save_total_limit\n"
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
      "The following columns in the evaluation set  don't have a corresponding argument in `BigBirdPegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `BigBirdPegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 4\n",
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.\n",
      "  warnings.warn('Was asked to gather along dimension 0, but all '\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2830 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2830 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4070 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4070 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2265 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2265 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3226 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3226 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3053 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3053 to 3072 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4050 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4050 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3872 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3872 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3818 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3818 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3827 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3827 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2808 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2808 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2921 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2921 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3902 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3902 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3624 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3624 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3900 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3900 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2052 to 2112 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2052 to 2112 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4015 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4015 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2290 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2290 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3006 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3006 to 3008 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1880 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1880 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3801 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3801 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3546 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3546 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3341 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3341 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3993 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3993 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3449 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3449 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3785 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3785 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3103 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3103 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3471 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3471 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2506 to 2560 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2506 to 2560 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4018 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4018 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3873 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3873 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3467 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3467 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3643 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3643 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2607 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2607 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3979 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3979 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2477 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2477 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2403 to 2432 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2403 to 2432 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3867 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3867 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2870 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2870 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3509 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3509 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2183 to 2240 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2183 to 2240 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3739 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3739 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4065 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4065 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3093 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3093 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3194 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3194 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2453 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2453 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3229 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3229 to 3264 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3347 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3347 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4026 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4026 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3367 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3367 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3716 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3716 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3474 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3474 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3697 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3697 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2580 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2580 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3877 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3877 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3589 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3589 to 3648 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4041 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4041 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4095 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4095 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4038 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4038 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3153 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3153 to 3200 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2869 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2869 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4060 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4060 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2857 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2857 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3850 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3850 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3886 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3886 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2852 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2852 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3967 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3967 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3724 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3724 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3652 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3652 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3364 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3364 to 3392 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3653 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3653 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3323 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3323 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3896 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3896 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4067 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4067 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3725 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3725 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3843 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3843 to 3904 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3547 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3547 to 3584 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3104 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3104 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3113 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3113 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3688 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3688 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3812 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3812 to 3840 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2524 to 2560 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2524 to 2560 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4068 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4068 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3964 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3964 to 3968 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3468 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3468 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Saving model checkpoint to bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-2500\n",
      "Configuration saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-2500/config.json\n",
      "Model weights saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-2500/pytorch_model.bin\n",
      "tokenizer config file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-2500/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-2500/special_tokens_map.json\n"
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
      "tokenizer config file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/special_tokens_map.json\n"
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
      "Deleting older checkpoint [bigbird-pegasus-large-arxiv-finetuned-pubmed/checkpoint-1000] due to args.save_total_limit\n"
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
      "The following columns in the evaluation set  don't have a corresponding argument in `BigBirdPegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `BigBirdPegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 4\n",
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.\n",
      "  warnings.warn('Was asked to gather along dimension 0, but all '\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4066 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3439 to 3456 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2581 to 2624 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2270 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4049 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3771 to 3776 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2436 to 2496 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3109 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2936 to 2944 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3101 to 3136 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2788 to 2816 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2723 to 2752 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2838 to 2880 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4014 to 4032 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3287 to 3328 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3515 to 3520 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 4064 to 4096 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 2271 to 2304 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 1877 to 1920 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
      "Input ids are automatically padded from 3664 to 3712 to be a multiple of `config.block_size`: 64\n",
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
       "TrainOutput(global_step=2500, training_loss=1.920787060546875, metrics={'train_runtime': 7972.5528, 'train_samples_per_second': 1.254, 'train_steps_per_second': 0.314, 'total_flos': 1.1196507790727578e+17, 'train_loss': 1.920787060546875, 'epoch': 5.0})"
      ]
     },
     "execution_count": 28,
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
   "execution_count": 29,
   "metadata": {
    "id": "jj7tm3Hvir6_"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Saving model checkpoint to bigbird-pegasus-large-arxiv-finetuned-pubmed\n",
      "Configuration saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/config.json\n",
      "Model weights saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/pytorch_model.bin\n",
      "tokenizer config file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-arxiv-finetuned-pubmed/special_tokens_map.json\n"
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
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "To https://huggingface.co/Kevincp560/bigbird-pegasus-large-arxiv-finetuned-pubmed\n",
      "   5e899ba..6a50e1d  main -> main\n",
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
  "accelerator": "TPU",
  "colab": {
   "collapsed_sections": [],
   "machine_shape": "hm",
   "name": "bigbird-pegasus-arxiv-pubmed-summary-final.ipynb",
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
