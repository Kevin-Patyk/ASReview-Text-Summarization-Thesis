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
      "\u001b[K     |████████████████████████████████| 312 kB 7.8 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting transformers\n",
      "  Downloading transformers-4.17.0-py3-none-any.whl (3.8 MB)\n",
      "\u001b[K     |████████████████████████████████| 3.8 MB 36.6 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting rouge-score\n",
      "  Downloading rouge_score-0.0.4-py2.py3-none-any.whl (22 kB)\n",
      "Collecting nltk\n",
      "  Downloading nltk-3.7-py3-none-any.whl (1.5 MB)\n",
      "\u001b[K     |████████████████████████████████| 1.5 MB 17.7 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: ipywidgets in ./miniconda3/envs/fastai/lib/python3.8/site-packages (7.6.4)\n",
      "Collecting responses<0.19\n",
      "  Downloading responses-0.18.0-py3-none-any.whl (38 kB)\n",
      "Collecting huggingface-hub<1.0.0,>=0.1.0\n",
      "  Downloading huggingface_hub-0.4.0-py3-none-any.whl (67 kB)\n",
      "\u001b[K     |████████████████████████████████| 67 kB 4.5 MB/s  eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: pandas in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (1.3.2)\n",
      "Requirement already satisfied: numpy>=1.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (1.20.3)\n",
      "Collecting pyarrow!=4.0.0,>=3.0.0\n",
      "  Downloading pyarrow-7.0.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (26.7 MB)\n",
      "\u001b[K     |████████████████████████████████| 26.7 MB 18.4 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting fsspec[http]>=2021.05.0\n",
      "  Downloading fsspec-2022.2.0-py3-none-any.whl (134 kB)\n",
      "\u001b[K     |████████████████████████████████| 134 kB 24.7 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: tqdm>=4.62.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (4.62.2)\n",
      "Collecting multiprocess\n",
      "  Downloading multiprocess-0.70.12.2-py38-none-any.whl (128 kB)\n",
      "\u001b[K     |████████████████████████████████| 128 kB 35.0 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: packaging in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (21.0)\n",
      "Collecting dill\n",
      "  Downloading dill-0.3.4-py2.py3-none-any.whl (86 kB)\n",
      "\u001b[K     |████████████████████████████████| 86 kB 4.9 MB/s  eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: aiohttp in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (3.7.4.post0)\n",
      "Collecting xxhash\n",
      "  Downloading xxhash-3.0.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (212 kB)\n",
      "\u001b[K     |████████████████████████████████| 212 kB 50.5 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: requests>=2.19.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (2.26.0)\n",
      "Collecting regex!=2019.12.17\n",
      "  Downloading regex-2022.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (764 kB)\n",
      "\u001b[K     |████████████████████████████████| 764 kB 37.6 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting filelock\n",
      "  Downloading filelock-3.6.0-py3-none-any.whl (10.0 kB)\n",
      "Collecting sacremoses\n",
      "  Downloading sacremoses-0.0.47-py2.py3-none-any.whl (895 kB)\n",
      "\u001b[K     |████████████████████████████████| 895 kB 40.6 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting tokenizers!=0.11.3,>=0.11.1\n",
      "  Downloading tokenizers-0.11.6-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.5 MB)\n",
      "\u001b[K     |████████████████████████████████| 6.5 MB 18.0 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: pyyaml>=5.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (5.4.1)\n",
      "Requirement already satisfied: six>=1.14.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from rouge-score) (1.16.0)\n",
      "Collecting absl-py\n",
      "  Downloading absl_py-1.0.0-py3-none-any.whl (126 kB)\n",
      "\u001b[K     |████████████████████████████████| 126 kB 32.0 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: joblib in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nltk) (1.0.1)\n",
      "Requirement already satisfied: click in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nltk) (8.0.1)\n",
      "Requirement already satisfied: traitlets>=4.3.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (5.1.0)\n",
      "Requirement already satisfied: jupyterlab-widgets>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (1.0.0)\n",
      "Requirement already satisfied: ipykernel>=4.5.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (6.2.0)\n",
      "Requirement already satisfied: nbformat>=4.2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (5.1.3)\n",
      "Requirement already satisfied: ipython-genutils~=0.2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (0.2.0)\n",
      "Requirement already satisfied: widgetsnbextension~=3.5.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (3.5.1)\n",
      "Requirement already satisfied: ipython>=4.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (7.27.0)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (3.10.0.2)\n",
      "Requirement already satisfied: matplotlib-inline<0.2.0,>=0.1.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (0.1.2)\n",
      "Requirement already satisfied: jupyter-client<8.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.1.12)\n",
      "Requirement already satisfied: tornado<7.0,>=4.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.1)\n",
      "Requirement already satisfied: debugpy<2.0,>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (1.4.1)\n",
      "Requirement already satisfied: setuptools>=18.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (58.0.4)\n",
      "Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (3.0.17)\n",
      "Requirement already satisfied: jedi>=0.16 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.18.0)\n",
      "Requirement already satisfied: backcall in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.2.0)\n",
      "Requirement already satisfied: pygments in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (2.10.0)\n",
      "Requirement already satisfied: pexpect>4.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (4.8.0)\n",
      "Requirement already satisfied: pickleshare in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.7.5)\n",
      "Requirement already satisfied: decorator in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (5.0.9)\n",
      "Requirement already satisfied: parso<0.9.0,>=0.8.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jedi>=0.16->ipython>=4.0.0->ipywidgets) (0.8.2)\n",
      "Requirement already satisfied: python-dateutil>=2.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jupyter-client<8.0->ipykernel>=4.5.1->ipywidgets) (2.8.2)\n",
      "Requirement already satisfied: jupyter-core>=4.6.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jupyter-client<8.0->ipykernel>=4.5.1->ipywidgets) (4.7.1)\n",
      "Requirement already satisfied: pyzmq>=13 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jupyter-client<8.0->ipykernel>=4.5.1->ipywidgets) (22.2.1)\n",
      "Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbformat>=4.2.0->ipywidgets) (3.2.0)\n",
      "Requirement already satisfied: attrs>=17.4.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (21.2.0)\n",
      "Requirement already satisfied: pyrsistent>=0.14.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (0.17.3)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from packaging->datasets) (2.4.7)\n",
      "Requirement already satisfied: ptyprocess>=0.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from pexpect>4.3->ipython>=4.0.0->ipywidgets) (0.7.0)\n",
      "Requirement already satisfied: wcwidth in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=4.0.0->ipywidgets) (0.2.5)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (2021.5.30)\n",
      "Requirement already satisfied: idna<4,>=2.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (3.2)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (1.26.6)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (2.0.4)\n",
      "Requirement already satisfied: notebook>=4.4.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from widgetsnbextension~=3.5.0->ipywidgets) (6.4.3)\n",
      "Requirement already satisfied: argon2-cffi in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (20.1.0)\n",
      "Requirement already satisfied: jinja2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (3.0.1)\n",
      "Requirement already satisfied: prometheus-client in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.11.0)\n",
      "Requirement already satisfied: Send2Trash>=1.5.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.8.0)\n",
      "Requirement already satisfied: nbconvert in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (5.6.1)\n",
      "Requirement already satisfied: terminado>=0.8.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.9.4)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (5.1.0)\n",
      "Requirement already satisfied: chardet<5.0,>=2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (4.0.0)\n",
      "Requirement already satisfied: async-timeout<4.0,>=3.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (3.0.1)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (1.5.1)\n",
      "Requirement already satisfied: cffi>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.14.0)\n",
      "Requirement already satisfied: pycparser in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from cffi>=1.0.0->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.20)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jinja2->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.0.1)\n",
      "Requirement already satisfied: bleach in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (4.0.0)\n",
      "Requirement already satisfied: mistune<2,>=0.8.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.8.4)\n",
      "Requirement already satisfied: testpath in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.5.0)\n",
      "Requirement already satisfied: defusedxml in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.7.1)\n",
      "Requirement already satisfied: entrypoints>=0.2.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.3)\n",
      "Requirement already satisfied: pandocfilters>=1.4.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.4.3)\n",
      "Requirement already satisfied: webencodings in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.5.1)\n",
      "Requirement already satisfied: pytz>=2017.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from pandas->datasets) (2021.1)\n",
      "Installing collected packages: regex, fsspec, filelock, dill, xxhash, tokenizers, sacremoses, responses, pyarrow, nltk, multiprocess, huggingface-hub, absl-py, transformers, rouge-score, datasets\n",
      "Successfully installed absl-py-1.0.0 datasets-1.18.4 dill-0.3.4 filelock-3.6.0 fsspec-2022.2.0 huggingface-hub-0.4.0 multiprocess-0.70.12.2 nltk-3.7 pyarrow-7.0.0 regex-2022.3.2 responses-0.18.0 rouge-score-0.0.4 sacremoses-0.0.47 tokenizers-0.11.6 transformers-4.17.0 xxhash-3.0.0\n"
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
       "model_id": "46d1c673799648609a7e1b5c72cd31b0",
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
      "Fetched 3316 kB in 1s (2576 kB/s)[0mm\u001b[33m\u001b[33m\n",
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
    "model_checkpoint = \"google/pegasus-arxiv\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4RRkXuteIrIh"
   },
   "source": [
    "This notebook is built to run  with any model checkpoint from the [Model Hub](https://huggingface.co/models) as long as that model has a sequence-to-sequence version in the Transformers library. Here we picked the [`google/pegasus-arxiv`](https://huggingface.co/google/pegasus-arxiv) checkpoint. "
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
       "model_id": "423f7fdbb3094c1c9796570a19ace502",
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
       "model_id": "5e266163db2348c29665e083f8ba1adb",
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
       "model_id": "a68330c1b40c44a6bc437d1c603d24a1",
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
       "model_id": "cd30d3126fd44d1c825b8a2f28cb5e32",
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
       "model_id": "a6eac09f8f91422b85aecd20fa02331e",
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
       "model_id": "075d93c66f914a208a3adc719aab4979",
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
       "      <td>the experimental animals , feeding and management : the experimental \\n procedures complied with the guide for the care and use of agricultural animals of obihiro \\n university . the experiment was carried out at the field center of animal science and agriculture , \\n obihiro university of agriculture and veterinary medicine . fifty multiparous holstein cows , \\n in which parity was from 1st to 5th at dry period , were used in this study and had calved \\n between september 2011 and august 2012 . parity and body condition score ( bcs ) of the \\n experimental cows at initiation of the study were 2.4  0.2 and 3.41  0.04 , respectively . \\n the study was performed from 3 weeks before the expected parturition to 100 days pp . cows \\n close to the dry period , about 1 month before the expected calving date , were moved to a \\n paddock and fed a limited total mixed ration [ dry matter ( dm ) basis : 127 g of crude protein \\n ( cp)/kg and 6.6 mj of net energy for lactation ( nel)/kg ] consisting of grass silage ( 3.5 kg , \\n dm basis : 165 g of cp / kg and 5.5 mj of nel / kg ) , maize silage ( 5.1 kg , dm basis : 86 g of \\n cp / kg and 6.0 mj of nel / kg ) , concentrate for dry cows ( 2.0 kg , dm basis : 170 g of cp / kg and \\n 6.8 mj of nel / kg ) and grass hay ( dm basis : 125 g of cp / kg and 5.7 mj of nel / kg ) ad \\n libitum until parturition . after parturition , cows were housed in a free - stall \\n barn and fed a lactation diet , which was a mixed ration ( dm basis : 155 g of cp / kg and 6.2 mj \\n of nel / kg ) consisting of grass ( 6.5 kg , dm basis : 165 g of cp / kg and 5.5 mj of nel / kg ) , \\n maize silage ( 12.5 kg , dm basis : 84 g of cp / kg and 6.4 mj of nel / kg ) and concentrate for \\n dairy cows ( 8.0 kg , dm basis : 180 g of cp / kg and 7.1 mj of nel / kg ) ad \\n libitum . in addition , the diets were supplemented with minerals , and the dairy \\n cow concentrate was prepared according to each cow s specific requirements for milk \\n production . grass hay ( dm basis : 104 g of cp / kg and 5.2 mj of nel / kg ) and water were \\n available ad libitum . cows were milked twice daily between 05:00 and 06:30 \\n hr and between 17:00 and 18:30 hr . the experimental itt and sampling : the itt was performed 3 weeks before \\n the expected calving date . the cows were weighed the day before the initiation of the itt , \\n and bw was used to determine the doses of insulin for the itt . immediately before the itt , an extension catheter was inserted into the \\n right or left jugular vein . the itt was performed by intravenously administering 0.05 iu / kg \\n bw of insulin ( novolin r 100 iu / ml ; novo nordisk pharma , tokyo , japan ) , \\n followed by administration of 5 ml heparinized saline ( 100 \\n iu / ml ) . blood samples were \\n collected via the jugular vein at 0 ( before insulin injection ) , 30 , 45 and 60 min relative \\n to the administration of insulin via caudal venipuncture to measure glucose and insulin . bcs was assessed twice a week from 3 weeks before the expected parturition to 3 weeks after \\n calving by the same operator by using a 1 to 5 scale with 0.25 intervals , where 1=thin and \\n 5=very fat . blood samples were obtained by \\n caudal venipuncture twice a week from 3 weeks before the expected parturition to 3 weeks \\n after calving . blood samples were collected via the jugular vein from the calves immediately \\n after birth . nonheparinized and silicone - coated 9-ml tubes ( venoject , \\n autosep , gel + clot . act . vp - as109k ; terumo corporation , tokyo , japan ) were used for \\n biochemical analysis , and sterile 10-ml tubes containing 200 \\n l of stabilizer solution ( 0.3 m edta and 1% acetyl salicylic acid , ph \\n 7.4 ) were used for hormonal analysis . serum was obtained by centrifuging the blood samples \\n for 15 min at 38c in an incubator . all the tubes were centrifuged at 2,000  \\n g for 20 min at 4c , and plasma samples were maintained at 30c until \\n analysis . in addition , milk samples were collected twice a week after milking until the \\n onset of luteal activity . the milk samples were centrifuged at 1,500  g \\n for 15 min at 4c , and the skim milk samples were stored at 30c until analysis for \\n progesterone concentration . daily milk yield was recorded until 100 days pp . peripartum \\n diseases , such as milk fever , hypocalcemia , ketosis , ruminal acidosis , displaced abomasum , \\n lameness , retained placenta , endometritis and mastitis , were recorded when that has been \\n diagnosed from 3 weeks prepartum to 3 weeks postpartum by veterinarian in the experimental \\n farm . the experimental measurement of hormones and metabolites : plasma and skim \\n milk progesterone concentrations were determined using enzyme immunoassay ( eia ) after \\n extraction with diethyl ether , as described previously ; the extraction efficiency was 90% . the standard curve ranged from 0.05 to 50 \\n ng / ml , and the 50% effective dose ( ed50 ) of \\n the assay was 0.66 ng / ml . the mean intra - assay and \\n inter - assay coefficients of variation ( cvs ) were 6.0% and 9.2% , respectively . the total \\n plasma insulin - like growth factor 1 ( igf-1 ) concentration was determined using eia by using \\n the biotin  streptavidin amplification technique \\n after protein extraction by using acid ethanol ( 87.5% ethanol and 12.5% 2 n hydrochloric \\n acid ) to obtain igf-1 free from binding proteins . intra- and inter - assay cvs were 5.9% and 6.1% , \\n respectively , and the ed50 of this assay system was 7.2 \\n ng / ml . the plasma gh concentrations were determined \\n using eia as described previously ; the standard \\n curve ranged from 0.78 to 100 ng / ml , and the \\n ed50 was 21 ng / ml . intra- and inter - assay cvs \\n were 3.1% and 8.2% , respectively . the plasma insulin concentrations were determined using an \\n enzyme - linked immunosorbent assay ( elisa ) kit ( bovine insulin elisa 10 - 1201 - 01 ; mercodia , \\n uppsala , sweden ) . the serum concentrations of glucose , non - esterified fatty acids ( nefa ) , -hydroxybutyrate \\n ( bhba ) , total protein ( tp ) , albumin ( alb ) , blood urea nitrogen ( bun ) and total cholesterol \\n ( t - cho ) and the activities of aspartate aminotransferase ( ast ) were measured using a \\n clinical chemistry automated analyzer ( tba120fr ; toshiba medical systems co. , ltd . , tochigi , \\n japan ) . the experimental identification of the onset of luteal activity : when the \\n progesterone concentration in the plasma or skim milk had increased to more than 1 \\n ng / ml , the cows were considered to show luteal activity \\n . the experimental statistical analysis : sixteen cows were excluded from \\n data analysis , because of the following reasons : a pregnancy period of more than 287 or less \\n than 273 days ( n=6 ) , severe mastitis ( n=3 ) , twin calving ( n=1 ) , blood collection loss at itt \\n ( n=3 ) and mistakes in insulin injection at itt ( n=3 ) . cows were divided into two groups \\n based on the time required for glucose to reach the minimum levels after insulin injection . \\n cows with a minimum glucose at 60 min after insulin injection were considered to have lower \\n insulin sensitivity and/or lower glucose metabolism compared to cows with a minimum glucose \\n level by 45 min after insulin injection . therefore , cows with a minimum glucose level at 60 \\n min after insulin injection were defined as the insulin resistant group ( ir group ) , whereas \\n those with a minimum glucose level by 45 min after insulin injection were defined as the \\n non - insulin resistant ( nir group ) in this study . before data analysis , bcs , plasma igf-1 , gh \\n and insulin concentrations , and serum metabolite concentrations were averaged weekly . the \\n period of 06 days after calving was considered as the parturient week ( 0 week pp ) , and the \\n kolmogorov  smirnov test ( sas enterprise guide version 4.3 ; sas institute inc . , cary , nc , \\n u.s.a . ) was used for statistical testing of normality . in addition , the data were analyzed \\n separately for the prepartum and pp periods . stat view ( stat view 5.0 software ; abacus \\n concepts inc . , berkeley , ca , u.s.a . ) was used for data analysis by using the repeated \\n measures of the anova procedure , including time ( week ) , group ( nir or ir ) and their \\n interaction in the model as fixed effects . diagnosis of peripartum diseases and sex of calves in the nir and ir groups were analyzed \\n using the chi - square test , and other data , including results for calves between nir and ir , \\n were analyzed using the student s t - test or wilcoxon s signed rank test \\n ( sas enterprise guide version 4.3 ; sas institute inc . ) . results are presented as mean  \\n standard error of the mean ( sem ) ; differences with p&lt;0.05 were \\n considered significant . in 28 of the 34 experimental cows , the time required for glucose to reach the minimum level \\n was 45 min after insulin injection with one exception ( 30 min ; n=1 , 45 min ; n=27 , nir \\n group ) . the remaining experimental cows ( n=6 ) required 60 min after insulin injection to \\n attain the minimum glucose levels ( ir group ) . serum glucose concentrations at 60 min after \\n insulin injection were higher in the nir group than in the ir group , although glucose levels \\n at the other time points did not differ between the nir and ir groups ( fig . 1.the change in serum glucose concentration after insulin injection at insulin \\n tolerance test ( itt ) in the nir ( n=28 ) and ir ( n=6 ) groups . p&lt;0.05 ) . the change in serum glucose concentration after insulin injection at insulin \\n tolerance test ( itt ) in the nir ( n=28 ) and ir ( n=6 ) groups . table 1table 1.parity , calving difficulty , sex of calves , peripartum disease , luteal activity \\n onset and milk yield in the nir and ir groupsnir groupir groupp - value(n=28)(n=6)parity at the onset of experiment2.4  0.32.2  0.70.460calving difficulty1.1  0.11.0  0.00.700sex of calves ( male / female)14/143/31.000diagnosis of peripartum disease9/28 ( 32%)1/6 ( 17%)0.645days to the onset of luteal activity ( days)38.3  3.820.3  3.60.039average of daily milk yield between days 7 and 100 pp ( kg)41.4  0.935.9  2.00.013total milk yield from days 7 to 100 pp ( kg)3,888.1  81.23,375.5  185.90.013values are the mean  sem . a ) nir group ; cows with a minimum glucose level by 45 min \\n after insulin injection . ir group ; cows with a minimum glucose level at 60 min after \\n insulin injection . b ) 1 , unassisted birth ( natural , without human assistance ) ; 2 , easy \\n calving with human assistance ; 3 , difficult calving with a few humans ; 4 , dystocia \\n ( requiring considerably more force than normal ) ; and 5 , surgical treatment or death of \\n cow . c ) milk fever , hypocalcemia , ketosis , ruminal acidosis , displaced abomasum , \\n lameness , retained placenta , endometritis and mastitis from 3 weeks prepartum to 3 \\n weeks postpartum . shows the parity , calving difficulty , sex of calves , peripartum disease \\n diagnosis , luteal activity onset and milk yield until 100 days pp in the nir and ir groups . \\n days until the onset of luteal activity in the ir group were fewer than those in the nir \\n group ( p&lt;0.05 ) . in addition , the average ( p&lt;0.05 ) \\n and total ( p&lt;0.05 ) milk yields until 100 days pp were lower in the ir \\n group than in the nir group . peripartum diseases were diagnosed as mastitis ( n=6 ) , \\n hypocalcemia ( n=1 ) and milk fever ( n=2 ) in nir group , and as mastitis ( n=1 ) in ir group , and \\n there was no significant difference in the number of cows with the peripartum diseases \\n between nir and ir groups . no significant difference was noted in other factors between the \\n nir and ir groups . a ) nir group ; cows with a minimum glucose level by 45 min \\n after insulin injection . ir group ; cows with a minimum glucose level at 60 min after \\n insulin injection . b ) 1 , unassisted birth ( natural , without human assistance ) ; 2 , easy \\n calving with human assistance ; 3 , difficult calving with a few humans ; 4 , dystocia \\n ( requiring considerably more force than normal ) ; and 5 , surgical treatment or death of \\n cow . c ) milk fever , hypocalcemia , ketosis , ruminal acidosis , displaced abomasum , \\n lameness , retained placenta , endometritis and mastitis from 3 weeks prepartum to 3 \\n weeks postpartum . 2.serum metabolite concentrations , activities of enzymes and plasma metabolic hormones , \\n and bcs during the experimental period [ mean  sem : solid , nir ( n=28 ) ; open , ir ( n=6 ) \\n groups ] . * indicates differences of p&lt;0.05 , and  indicates \\n differences of p&lt;0.1 between the nir and ir groups . shows the circulating serum metabolite concentrations , enzyme levels , plasma \\n metabolic concentrations and bcs during the experimental period . during the prepartum \\n period , bcs ( p&lt;0.05 ) , and serum bun concentrations \\n ( p&lt;0.05 ) were lower , whereas serum glucose ( p=0.05 ) and \\n alb concentrations ( p=0.10 ) tended to be lower in the ir group than in the \\n nir group . during the pp period , cows of the nir group had higher serum nefa \\n ( p&lt;0.05 ) and bhba ( p=0.09 ) concentrations than those \\n in the ir group . in addition , treatment and time effects were observed \\n ( p&lt;0.05 ) for bcs during the pp period : bcs at 0 \\n ( p=0.08 ) and 1 ( p&lt;0.05 ) week pp were lower in the ir \\n group than in the nir group . no significant differences were noted in the other factors \\n between the nir and ir groups in each period . serum metabolite concentrations , activities of enzymes and plasma metabolic hormones , \\n and bcs during the experimental period [ mean  sem : solid , nir ( n=28 ) ; open , ir ( n=6 ) \\n groups ] . * indicates differences of p&lt;0.05 , and  indicates \\n differences of p&lt;0.1 between the nir and ir groups . bw , plasma metabolic hormone levels and serum glucose concentrations at birth in the calves \\n of cows of the nir and ir groups are shown in table \\n 2table 2.bw and plasma metabolic hormones and serum glucose concentrations at birth in the \\n calves of the nir and ir groupscalves of nircalves of irp - value(n=28)(n=6)bw at the birth ( kg)47.2  0.942.1  1.70.020plasma gh concentration \\n ( ng / ml)13.6  1.315.2  4.80.653plasma igf-1 concentration \\n ( ng / ml)121.5  6.369.8  5.60.001plasma insulin concentration \\n ( ng / ml)0.3  0.00.7  0.20.061serum glucose concentration \\n ( mg / dl)77.4  5.272.1  14.80.684values are the mean  sem . a ) nir group ; cows with a minimum glucose level by 45 min \\n after insulin injection . ir group ; cows with a minimum glucose level at 60 min after \\n insulin injection .. bw at birth in the calves of the ir group was lower than that in the calves \\n of the nir group ( p&lt;0.05 ) . furthermore , the calves of the ir group \\n showed lower plasma igf-1 concentration ( p&lt;0.001 ) and higher plasma \\n insulin concentration ( p=0.06 ) . no significant differences were noted in \\n the plasma gh and serum glucose levels at birth between the calves of the nir and ir \\n groups . a ) nir group ; cows with a minimum glucose level by 45 min \\n after insulin injection . ir group ; cows with a minimum glucose level at 60 min after \\n insulin injection . in this study , the six cows that reached the minimum glucose levels at 60 min after insulin \\n injection were considered to be ir ; the reason for ir was thought to be the slow recovery of \\n glucose after insulin injection , which is consistent with the findings of a previous study \\n by lee et al . . in general , bcs \\n and blood glucose and bun concentrations are known to be associated with energy status and \\n feed intake [ 7 , 8 , 38 ] . during the prepartum period , ir \\n cows showed lower energy status and feed intake owing to the lower bcs and glucose and bun \\n concentrations . although ir by itt was confirmed at 3 weeks before calving , a difference in \\n energy status between ir and nir cows was noted . in particular , bcs can not be evaluated on \\n the basis of the change in energy status of the real - time feed intake ; therefore , in this study , ir cows might have become insulin \\n resistant during an earlier time . malnutrition causes imbalance in glucose homeostasis , and \\n the decrease of insulin in circulation induces the reduction of feed intake in dairy cows \\n . therefore , the feed intake reduction in ir \\n cows might have inhibited the volatile fatty acid production in the rumen and thus \\n suppressed gluconeogenesis in the liver . lower \\n energy status , such as lower bcs , before calving of ir group in this study might be caused \\n by long - term malnutrition from previous lactation . however , in this study , the reasons for \\n the lower energy status in the ir cows were not clear ; thus , further studies are warranted \\n to confirm the onset of insulin resistance in pregnant dairy cows . bcs at 0 and 1 week pp were lower in the ir group than in the nir group . nir cows showed \\n higher serum nefa and bhba concentrations than those in ir cows , although the levels of \\n metabolic hormones did not differ between the ir and nir cows . furthermore , the average and \\n total milk yield until 100 days pp were lower in the ir group than in the nir group . higher \\n nefa and bhba indicate greater mobilization of adipose tissue and failure of lipid \\n metabolism in the liver [ 14 , 15 ] . however , cows with lower bcs had sustained reduced plasma nefa and \\n bhba concentrations after calving compared to cows with higher bcs [ 28 , 33 ] . cows with lower bcs \\n produce milk by protein mobilization , because of the limited body fat ; thus , fat - corrected \\n milk yield in those cows was lower than moderate and fat cows . conversely , it was indicated that higher bcs cows have ability to \\n mobilize fat to maintain energetic homeostasis after feed restriction . additionally , roche et al .   have concluded that bcs at calving had positive effect on milk yield , \\n and optimal bcs at calving was 3.5 in the 5-point scale . in the present study , greater bcs \\n and better gluconeogenesis in nir group might produce greater milk yield compared with ir \\n group , although the differences of them between nir and ir groups were not so greater . days \\n to the onset of luteal activity in the ir group were fewer than in the nir group . in dairy \\n cows , lowered energy status during the peripartum period is known to delay the first \\n ovulation after parturition [ 2 , 19 ] . butler and smith   showed \\n that a negative energy balance was directly related to the pp interval to the first \\n ovulation and that the differences in the energy balance were reflected in the milk yield . \\n in addition , cows with a delayed first ovulation showed higher nefa and bhba concentrations \\n after parturition [ 21 , 31 , 39 ] . therefore , in this study , \\n higher nefa and bhba concentrations of nir cows during the pp period might have delayed the \\n onset of luteal activity , and the lowered milk yield of ir cows might induce earlier \\n resumption of ovarian activity . the maternal endocrine and metabolic milieu transferred through the placenta during late \\n pregnancy affects the environment of the fetus [ 17 , \\n 24 , 30 ] . in \\n humans , ir of the mother is associated with low birth weight of the infant ; in cattle , maternal malnutrition during gestation is \\n related to the lowered development of both the placenta and the fetus [ 24 , 30 ] . further , in ewes , \\n restricted maternal feeding during gestation was related to lower bw and plasma igf-1 , \\n insulin and glucose concentrations in the fetus , although maternal igf-1 concentrations were \\n not affected . in the present study , calves of \\n the ir cows showed lowered bw at birth and a lower plasma igf-1 concentration , supporting \\n the findings of previous studies . in addition , they showed higher insulin levels than those \\n of nir cows , despite the similar glucose levels . in the late gestation , fetal growth is \\n mainly regulated by igf-1 , and the dominant regulator of igf-1 production in the fetus is \\n fetal glucose and insulin . thus , the differences \\n in blood metabolic hormones and glucose concentrations between the calves of the nir and ir \\n groups might be attributed to the fetal nutritional condition that was affected by maternal \\n endocrine and metabolic milieu . in humans , lower bw at birth is known to be associated with \\n a wide range of adverse outcomes later in life , including diabetes ; further , obese children with low birth weight have higher blood \\n insulin to glucose concentration and show higher insulin resistance as revealed by the \\n homeostasis model assessment compared with obese children with normal birth weight . therefore , calves of ir cows might develop insulin \\n resistance in the future . in conclusion , the findings of the present study suggest that ir at 3 weeks before \\n parturition in dairy cows is related to the pp metabolic status , milk production and \\n resumption of ovarian activity along with growth , as well as the metabolic status of their \\n calves . therefore , ir evaluated on the basis of the recovery of glucose after an injection \\n of a small dose of insulin during the dry period might be an indication of the pp \\n performance of pregnant dairy cows , as well as the growth , fertility and milk production of \\n their calves . in addition , the reason for ir in the present study was thought to be the slow \\n recovery of glucose after insulin injection as well as the previous study . therefore , the enhancement of the gluconeogenesis in \\n the liver by energy supplementation , such as glycerol , or hepatic stimulant , such as amino \\n acids , should be confirmed in order to improve the ir .</td>\n",
       "      <td>&lt;S&gt; this study aimed to investigate the effects of insulin resistance ( ir ) during the \\n close - up dry period on the metabolic status and performance of dairy cows as well as to \\n determine the effects on body weight ( bw ) and metabolic status of their calves . an insulin \\n tolerance test ( itt ) was conducted by administering 0.05 iu / kg bw of insulin to 34 \\n multiparous holstein cows at 3 weeks prepartum . &lt;/S&gt; &lt;S&gt; blood samples were collected at 0 , 30 , 45 \\n and 60 min after insulin injection , and cows were divided into two groups based on the \\n time required for glucose to reach the minimum levels [ non - ir ( nir ) , 45 min ( n=28 ) ; and \\n ir , 60 min ( n=6 ) ] . &lt;/S&gt; &lt;S&gt; blood or milk sampling and body condition score ( bcs ) estimation were \\n performed twice weekly during the experimental period . &lt;/S&gt; &lt;S&gt; blood samples from calves were \\n collected immediately after birth . &lt;/S&gt; &lt;S&gt; cows with ir showed lower bcs \\n ( p&lt;0.05 ) and serum urea nitrogen ( p&lt;0.05 ) and \\n glucose concentration ( p=0.05 ) before calving , and lower serum \\n non - esterified fatty acid concentration ( p&lt;0.05 ) and milk yield \\n ( p&lt;0.05 ) and earlier resumption of luteal activity \\n ( p&lt;0.05 ) after calving ; their calves showed lower bw \\n ( p&lt;0.05 ) and plasma insulin - like growth factor - i concentration \\n ( p&lt;0.001 ) and higher plasma insulin concentration \\n ( p&lt;0.05 ) . in conclusion , ir at 3 weeks prepartum in dairy cows is \\n related to postpartum metabolic status and performance along with growth and metabolic \\n status of their calves . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>direct coronary stent implantation is an elegant technique for coronary artery revascularization.1 however , calcified coronary lesions , often seen in older patients suffering from diabetes mellitus , renal failure and hypertension , are challenging to deal with , as they require optimal lesion preparation prior stenting for avoiding stent underexpansion which is related to in - stent restenosis , target lesion revascularization and subsequent stent thrombosis.2 several strategies and technologies have been developed to address the problem of heavily calcified coronary lesions . these include simple dilatation using standard non - compliant balloon , cutting balloon and plaque modification using rotational atherectomy . we report on the management of an underexpanded bare - metal stent in a patient with heavily calcified lesion not amenable to high - pressure balloon - dilatation . a 72-year old man suffering from progressive angina over the past 8 weeks presented to our chest pain unit . he had previously documented insulin - dependent diabetes , alimentary obesity , hyperlipidemia and arterial hypertension . an ambulatory performed myocardial perfusion scintigraphy revealed a reduced tracer - uptake in the apex , left posterior and antero - lateral wall during physical examination ( 100 watt - cycling ) . coronary angiography , which was performed via right radial access using a 5f sheath , revealed a 50% stenosis of the left anterior descending artery ( lad ) and ramus posterolateralis sinistra ( rpls ) while the right coronary artery ( rca ) had a critical 90% stenosis ( fig . due to the patients symptoms and the angiographic findings we decided to perform a percutaneous coronary intervention ( pci ) . the patient received 600 mg clopidogrel , 500 mg aspirin and 5000u heparin followed by primary pci and direct stenting of a bare - metal stent ( bms ) ( coroflex blue 3.5 mm/8 mm , b. braun , melsungen , germany ) with 18 atm for 30 sec ( fig . post - pci angiography revealed a 75% stenosis in the mid - portion of the stent ( fig . 1c ) . a subsequent dilatation with a semi - compliant balloon ( pantera 3,5/10 mm with [ biotronik , berlin , germany ] 18 atm over 30 sec ) , a non - compliant balloon ( quantum 3,5/8 mm [ boston scientific , natick , usa ] with 20 atm over 30 sec ) and a cutting - balloon ( 3,0/10 mm [ boston scientific , natick , usa ] with 18 atm over 30 sec ) could not expand the stent further ; pointing out the heavily calcified nature of this lesion . due to the fact that an underexpanded stent is a predictor for worse clinical outcome we decided on rotablation . additionally we introduced a 5f sheath in right femoral vein and inserted a transient pacemaker lead . after passing across the stenosis with the 0.009 rotawire we ablade the heavily calcified stenosis as well as the stent struts ( stentablation ) ( fig . all ablations were performed with a 1.75 mm burr with at least 150,000 rpm and ablation times &lt; 30 sec without a decrease in rotational speed of &gt; 5,000 rpm . the procedure was free of complications and we continued with dilatation with a non - compliant balloon ( quantum 3,5/8 mm with 20 atm over 30 sec ) and a cutting - balloon ( 3,0/10 mm with 16 atm over 30 sec ) . with complete expansion of the balloons the procedure was continued with implantation of a drug - eluting stent ( taxus libert 4.0/12 mm [ boston scientific , natick , usa ] with 16 atm over 30 sec ) ( rotastenting ) ( fig . finally , there was timi 3 without evidence of dissection or residual stenosis ( fig . 2c ) . following uneventful hospital stay without evidence of myocardial necrosis the patient was discharged after 3 days on 100 mg aspirin , 75 mg clopidogrel , 5 mg bisoprolol , 5 mg of ramipril and 40 mg simvastatin with a recommendation for dual antiplatelet therapy of 1 year without any change in his extra - cardiovascular medications . a routine coronary angiography performed 6 months after index - pci revealed a good result with a mild ( 25% ) restenosis ( fig . 2d ) . direct stenting is the implanation of stents in coronary lesions without predilatation.1 from animal restenosis models , direct stenting without the need for predilatation appears to reduce vessel trauma , in particular as a result of less endothelial denudation , resulting in less neointimal hyperplasia subsequently.1 pci of calcified and complex lesions has been associated with lower success rates , an increased frequency of acute complications , and higher restenosis rates than pci of simple lesions.2 as seen in our case , delivering the stent may be difficult and stent expansion may be inadequate in heavily calcified lesions , resulting in smaller acute gain compared to non - calcified lesions.2 it is widely accepted that achieving postprocedural residual stenosis is a major determinant of restenosis during follow - up and optimal stent expansion is a crucial factor in minimizing the risk of stent thrombosis pointed out by the fact that only 22% of patients that experienced subacute stent thrombosis have an acceptable pci result as assessed by ivus.2,3 a variety of strategies and technologies have been developed to address the problem of an underexpanded stent . the postit trial revealed that in case of using only the stent delivery balloon over 70% of patients did not achieve optimal stent deployment.4 use of non - compliant balloon to achieve full distension in resitant lesions is a reasonable first - step . however , focal points of resistance within a lesion result in non - uniform balloon expansion and characteristic  dog - boning  with overexpansion in the more compliant segments . in this non - uniform expansion may cause vessel dissection and rupture acutely as well as restenosis due to deep - wall injury in the follow - up . cutting - balloon , designed to score the vessel longitudinally rather than causing uncontrolled plaque disruption , have been used successfully in the treatment of undilatable lesions.5 in our case , none of these techniques were successful in reducing the underexpansion , demonstrating the nature of the heavily calcification , which was not assumed on initial fluoroscopy . thus , despite the existence of limited data,6 we decided to rotablade the remaining calcification and the underexpanded stent struts to avoid aforementioned complications . high - speed rotational atherectomy preferentially cuts hard plaque , increases plaque compliance and thereby renders the lesion more amenable to balloon dilatation.7 the rotablator is able to ablate inelastic tissue selectively while maintaining the integrity of elastic tissue due to the principle of differential cutting . these particles are small enough to pass through the coronary microcirculation and ultimately undergo phagocytosis in the liver , spleen , and lung.7 the procedure performed in our case was uneventful with no dissection , slow - flow , heamodynamic compromise or myocardial necrosis . we had applied a transient pacemaker via the right femoral vein to overcome possible conduction disturbances when handling in the right coronary artery . several observational studies have confirmed that rotational atherectomy prior to stent deployment in severely calcified lesions does facilitate stent delivery and expansion , but incidence of restenosis remains unsatisfactory ( 23% ) when bms are used.8 there is limited information about rotational atherectomy followed by des implantation , but initial results seem promising.9 a comparison of bms ( n = 84 ) and des ( n = 213 ) after rotablation with cardiac death and recurrent myocardial infarction being defined as primary endpoint and binary restenosis as secondary endpoint revealed lower rates for primary endpoint in des group ( 2.3% versus 7.1% ; p = 0.04 ) during a follow - up of 1300 days.10 despite our procedural success and good midterm result , there are no data on long - term follow - up after stentablation and rotastening . thus , it should be emphasized that a better lesion preparation is needed to avoid stent underexpansion in undilatable lesions .</td>\n",
       "      <td>&lt;S&gt; calcified coronary lesions are challenging to deal with , as they require optimal lesion preparation . &lt;/S&gt; &lt;S&gt; direct stenting in this scenario is associated with risk of stent - underexpansion , which is related to in - stent restenosis , target lesion revascularization and stent - thrombosis . &lt;/S&gt; &lt;S&gt; we report on the interventional management of an underexpanded bare - metal stent not amenable to high - pressure balloon dilation and cutting - balloon . &lt;/S&gt; &lt;S&gt; by using rotablation we could abrade the underexpanded stent struts and the calcification with subsequent implantation of a drug - eluting stent . &lt;/S&gt; &lt;S&gt; follow - up of 6 months revealed good results without evidence of significant restenosis . &lt;/S&gt; &lt;S&gt; our clinical experience and case reports in the literature suggest that this strategy might be an option for underexpanded stents not amenable to conventional techniques . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>total joint arthroplasty , in particular tha , has had a revolutionary role in improving the quality of life . the use of bone cement for the implant stability versus biologic fixation with bone ingrowth in cementless tha has been a controversial issue for years . while immediate cement fixation in very old people or in those with poor bone stock might provide a quicker return to daily activity , cementless implants have gained more popularity over the years . the relative superiority of cementless acetabular component over cemented ones is nowadays a well - accepted fact . the femoral stem , however , has very good long - term reports both in cemented and cementless forms . efforts to decrease the rate of loosening have included the use of newer materials , improvement in the design of the implant ; and modification of operative techniques . after the midterm follow - up results of cementless tha , the long - term results with impressive survival rates , are being reported more and more . the purpose of this study is to evaluate the efficacy and prosthesis survival in an iranian society , with its unique cultural lifestyle and social differences from western societies . the cases of cementless total hip arthroplasty performed in nemazee hospital by a single surgeon from may 1997 to june 2007 were included in a retrospective outcome study . from the total of 63 hips in 52 consecutive patients , 3 patients had died at the time of the last follow up due to problems unrelated to the operation and 7 patients could not be reached . the information from medical records of all the cases , including radiographs , was collected and the patients were called in for an interview , physical examination , and radiographic assessment . the patients filled the general - health assessment form , short form 36 ( sf-36 ) ; the arthritis specific functional instrument womac ( western ontario and mcmaster universities osteoarthritis index ) , and patient reference disability questionnaire mactar ( mcmaster toronto arthritis ) . harris hip rating scale was also filled for all the hips , ( where 90 - 100 points would be excellent , 80 - 89 good , 70 - 79 fair , and below 70 is assumed a poor result ) . the radiographic assessment was by measurement of the cup and stem alignment in the immediate post - operative anteroposterior and frog - lateral views . the same views in the final follow - ups were specifically evaluated for any possible change in cup orientation , loosening in cup or stem ( based on gruen classification for zones of the femoral stem and martell et.al . osseous integration of the acetabular and femoral components were assessed using the proposed criteria of moore et al . considering the five criteria of absence of radiolucent lines , superolateral and inferomedial buttressing , medial stress - shielding , and radial trabeculae for acetabular shell and stem bony ingrowth according to engh et al . osseo - integration in stem was evaluated by the presence of spot welds , cortical hypertrophy ; and absence of radiolucent lines , and pedestal formation . harris - galante ii ( hg ii ) porous - coated prosthesis ( hgp , zimmer , warsaw , indiana ) was implanted in 15 and versys - trilogy ( v - t ) system ( zimmer , warsaw , indiana ) in the remaining 37 hips . the straight stem is tivanium ti-6 al-4v alloy with a proximal pure titanium fiber metal mesh coating ; with a collar and a modula.r morse tapered neck , which is available in three lengths . there are porous pads , made of commercially pure titanium wire , proximally on the anterior and posterior surfaces and a small medial pad immediately distal to the collar . the shell is a partial hemisphere made of titanium alloy with fiber metal mesh coating with variable number of holes for screw fixation . versys femoral stem ( zimmer , warsaw , indiana ) is a collarless proximally and circumferentially coated prosthesis for cementless use . trilogy shells are coated with commercially pure titanium fiber metal , which is clinically proven to enhance fixation through bone ingrowths . infection prophylaxis was with cephalosporin and gentamicin at the time of surgery and 48 hours post - surgery . thromboembolic prophylaxis was mostly by warfarin for 6 weeks with the intended inr ( international normalized ratio ) of 1.7 to 2 . early mobilization and first post - surgery day ambulation and crutch walking for 6 weeks were the uniform care received by all the patients . the prosthesis survivorship limit was defined as implant life span and revision - ready state with progressive symptoms . definite  when subsidence , varus , or valgus orientation change was observed in femoral component and angle change , or migration of two millimeter or more seen in two views for the acetabular component . the 42 patients ( 52 hips ) included fourteen males ( 33.3% ) and twenty - eight females ( 66.7% ) , with the mean age of 48.83 years ( 13 years , range 22 - 75 ) at surgery . harris - galante ii prosthesis was used in 15 cases and versys - trilogy prosthesis in 37 hips . the average duration of follow - up was 65 months ( range 26 - 136 ) . the hg ii group of prostheses had a longer follow - up of 105 months ( range 52 - 136 ) . this figure was 49 months for versys - trilogy group ( range 26 - 78 ) . the overall mean follow - up was 65 months ( 32 , range 26 - 136 ) . the overall arthroplasty survival ( i.e. , well - functioning prosthesis with no clinical or radiographic evidence of wear , loosening , infection , etc . ) , which would suggest the need for revision was 65 months . therefore , 43 hips in 34 patients were in good and functional status by the time of last follow - up . post - operatively , hips had a mean flexion arc of 114 degrees and 9 degrees of flexion contracture . the overall hhs with a mean of 85 ( 15 , range 24 - 100 ) was excellent in 65.9% , good in 27.3% , fair in 4.5% and poor in 2.3% of cases . the womac score had a mean of 22.7 ( 13 , range 3 - 62 ) , with 3 being the best and 62 the worst case scenario . the pain subscore of womac was 2.87 , joint stiffness 2.21 and functional subscore 17.62 . the items in the function , which were of most concern to the patients were , in a descending order : inability in stair climbing ; sitting or getting up from the floor or from flat - top toilets ; picking up objects from the floor ; and putting on or taking off socks . sf 36 measurement had a total mean score of 61.33 ( range 18 - 95 ) . out of the 8 items in sf 36 , the patient expectation questionnaire of mactar had the following findings : pain relief was achieved in 41 cases ( 97.6% ) , improvement in walking in 39 ( 92.8% ) , and improved ability in performing daily living activities in 37 ( 88% ) . the correlation of the above scoring system in this group of patients was evaluated . a close correlation between harris hip score and total womac score pain  and  function  items , but not as much with  stiffness  item in womac ( p values 0.002 , 0.001 and 0.45 , respectively ) . sf 36 and harris hip score were closely correlated and value of r was 0.67 . sf 36 was more closely correlated with  pain  and  function  subscores of womac ( r=-0.77 and 0.78 , respectively ) . there was no infection , and no thromboembolic event in any of the 52 hips . in the last follow - up assessments , 44 hips ( 84.6% ) were functional and well fixed ; 8 cases had undergone revision and one patient is suspected of the early stage of loosening and is being followed .  pedestal  was seen in 3 ( 7% ) stems , and 1 - 2 millimeter non - progressive radiolucent lines in 20% of femurs and 4.5% of acetabular components . heterotopic ossification as a late complication was found in 35 hips ( 67.3% ) , 29 ( 82.7% ) of which were brookers i and ii , 5 brookers iii and one brookers iv . since there were two groups of prostheses from the same company used in this study , they also were separately evaluated : among the 15 cases of harris - galante ii prosthesis , with average follow - up of 105 months ( range 52 - 136 ) , 8 cases had developed problems , all of which had been already revised . all of the revised cases had problems in the acetabulum with cup wear , loosening , and polyethylene fracture and two of them had a simultaneous femoral loosening and osteolysis secondary to polyethylene wear debris . the etiology of hip disease in these 8 revisions , included five acetabular dysplasia , one avascular necrosis and systemic lupus erythematosus , one multiple epiphyseal dysplasia , and one primary osteoarthritis . in reviewing the original radiographs , no initial radiographic malposition was present and the stems were in normal orientation and mean shell inclination angle was 47 degrees ( range 40 - 57 degrees ) , which was not statistically different from the versys - trilogy group ( p=0.51 ) . the primary etiology of hip disease , in terms of distribution in these two groups , was different ( table 1 ) . comparison of etiology of hip replacement in two groups broken tines of fiber metal - coated acetabular shells were seen in 5 patients , all in the failed acetabular components ( figure 1 ) . broken tine of the cup the harris hip score , womac and sf36 in the 15 hg ii cases were significantly poorer than the 37 cases with versys - trilogy prosthesis : harris hip score of 66 versus 92 , womac 30 versus 20 ; and sf36 of 49 versus 66 ( p value 0.009 ) . the versys - trilogy prostheses are all surviving in a mean follow - up of 49 months ( range 26 - 78 ) with no radiographic or clinical evidence of loosening or wear . the five early complications , mentioned above , were all in this group of prostheses . this is a small group of cases with a midterm follow - up on porous - coated hip arthroplasty in a society with unique social habits and customs . the charnley prosthesis reported by ranawat had 90% survival of the femoral component , while harris had about 80% survival with revision mainly on the acetabular side . porous - coated implants were used with the idea of removing the so - called  weak link  in the hip replacement from 1971 . this has survived as a very good hip arthroplasty option for young active individuals with good bone stock . the results with non - circumferential proximal porous - coating prosthesis like porous - coated anatomic , pca ( howmedica , rutherford , new jersey ) and harris - galante i ( zimmer , warsaw , indiana ) were not satisfactory : failures of 43% and only 57% survival in 8 years . kim recently reported that pca prosthesis ( howmedica ) had 21% revision in 20 years for the acetabular component and 9% for femoral component . after generally satisfactory short and midterm results of the second generation of cementless implants ( with proximal circumferential porous - coating ) , clohisy and harris reported a 96% 10-year survival rate for acetabular component and archibeck et al . in a study of 92 patients with the same follow - up had a 96.4% survival rate for acetabular and 100% for femoral components . most studies have evaluated the functional results with hhs with 83 - 95 points on average . reported 100% ten years survival in a hydroxyapatite - coated , proximally and circumferentially coated prosthesis . engh et al . , using an extensively coated prosthesis reported on 5-year , 10-year , and 15-year follow - ups . the acetabular component in most reports is the one with more problems and the responsible section for loosening ( kim , engh , archibeck ) . the number of holes for temporary screw fixation has also been a point of concern , as more holes might provide better access for migration of polyethylene debris behind the shell and into the femoral canal . the locking of the polyethylene cup into the metal shell is variable in different designs of prostheses . poor locking mechanism can cause micromotion between the liner and the shell , causing more wear and subsequent dislodgment of the liner . the survival rate in the present series with 96.2% for the femoral component and 84.6% for the acetabular component in 5.5 years is not a very promising result . the high revision rate of 15.4% was primarily in the hg ii components and all were related to the acetabular side with wear , breakage , and dislodgment of polyethylene liner . louwerse et al . in 1999 reported 26 cases of liner failure , 13 of which belonged to hg cups . curry et al . in a 10-year follow - up reported 271 cases of hg ii prosthesis in 2008 . our hg ii group of arthroplasty in 9 years average follow - up had 46.7% overall prosthesis survival . the femoral stems were revised in only two cases that had severe bone lyses secondary to the acetabular liner problem , and the remaining 50 ( 96.2% ) stems are stable and functioning well . although the revised cups , except one , did not have primary osteoarthritis or inflammatory arthritis , the numbers are too few to draw any conclusion as to whether the primary etiology could have had any bearing on the high rate of liner problem in hg ii cups . the appearance of broken tines was visible on radiographs , one to two years before the hips became symptomatic . broken tines are probably early warning signs of instability . excessive motion will cause wear of the liner and material debris will initiate retro acetabular and proximal femoral osteolysis . this would eventually lead to failure . at the same time , the trilogy cups ( zimmer , warsaw , indiana ) with versys circumferentially coated stems have 100% survival in 4 years average follow - up . the locking mechanism in the trilogy is split - ring mechanism , which has been used in several other designs with a good track record . the harris hip score in the v - t group and surviving hg ii ( not revised ) were excellent or good in 93.2% and good in , but in the total group including the revised hips were 85 . the adjusted general health measures ( sf36 ) and disease specific outcome measures ( womac ) and patients expectations have been previously studied for knee arthroplasty in this region , but not for hip arthroplasty . the hg ii group had , understandably , a significant drop in their womac and sf36 scores due to the inclusion of the 15.6% revision . expectations of the patients , that were mainly relief of pain and ability to walk comfortably , were fulfilled in nearly all the patients ( 97.6% and 92.8% , respectively ) . some preoperative problems relatively unique to our culture , flattop toilet and cross - legged sitting on the floor , were not the expectations of the patients and seem to be modified after surgery by the patients . the radiographic evaluation in the present paper showed good positioning of cups and stems in accordance with established standards . the radiolucent lines and pedestal formation in those few cases were not indicative of loosening . there were only 4 cases ( 9.3% ) of thigh pain in this series that had no correlation with the size of the femoral stem . the incidence of thigh pain , which is related to the stability of the prosthesis , is reported between 0 and 28% in different articles . in spite of the literature report of 0.28 - 4% infection and 2.2 - 14.7% thromboembolic events the ulnar nerve injuries were in the contralateral upper limbs from arm malpositioning during anesthesia . the tibial and peroneal nerve injuries from the traction effect of lengthening observed in this report is a recognized problem , and has been reported in the literature with an incidence of 0.3 - 3.7% in hip arthroplasty , usually associated with lengthening of over 1.7 centimeters . the main limitations of the present study are its retrospective nature and small numbers of cases , however , the merits are that it is a single surgeon s experience with uniform technique and post - operative care and being a unique study in iran with the special cultural and daily living habits . the generally satisfactory results of hip arthroplasty as demonstrated by harris hip scores and functional assessments with womac , sf 36 and mactar are shown in iranian society in spite of some cultural and social differences . the outcome of cementless tha is satisfactory and comparable with the literature based on the results of function and survival of this small comparative group . the use of hgii acetabular component should be abandoned , because of the poor locking mechanism of the shell with the liner .</td>\n",
       "      <td>&lt;S&gt; background : cementless hip prosthesis was designed to provide biologic fixation , without the use of cement . &lt;/S&gt; &lt;S&gt; the second generation components have shown more reliable bone ingrowths and survival rates . &lt;/S&gt; &lt;S&gt; we are reporting a midterm result of two designs of cementless prosthesis in a unique culture with different social habits and expectations.methods:52 primary cementless total hip arthroplasty in 42 patients with the mean age of 48.8 years were retrospectively studied . &lt;/S&gt; &lt;S&gt; two groups of prosthesis had been implanted : harris - galante ii ( hgii ) in 15 and versys - trilogy ( v - t ) in 37 hips , both from zimmer company . &lt;/S&gt; &lt;S&gt; the patients were assessed clinically , radiographically and with harris hip score , sf36 , womac , and mactar questionnaires , with 65 months ( 26 - 136 ) mean follow-up.results:all the v - t prostheses had survived well . &lt;/S&gt; &lt;S&gt; eight of hg ii were revised by the last follow - up in 19 - 102 months . &lt;/S&gt; &lt;S&gt; all had undergone acetabular revision and 2 combined with femoral revision . broken tines of hgii cups &lt;/S&gt; &lt;S&gt; were seen in 4 radiographs . &lt;/S&gt; &lt;S&gt; the 65 months overall survival was 96.2% for femoral and 84.6% for acetabular components . &lt;/S&gt; &lt;S&gt; 90% had good or excellent harris hip scores . &lt;/S&gt; &lt;S&gt; the functional scores were poorer in the hg ii group . &lt;/S&gt; &lt;S&gt; pain relief and improved walking were the two main patients expectations fulfilled in 97.6% and 92.8% , respectively.conclusions:the outcome of cementless total hip arthroplasty ( tha ) is satisfactory and comparable with the literature based on the results of function and survival of this small comparative group . the use of hgii acetabular component should be abandoned . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>out of the 10,807 patients enrolled for the original survey , access survival data were available for 7058 ( 65% ) patients . these patients resided in portugal , the united kingdom , ireland , italy , turkey , romania , slovenia , poland , and spain . the mean age was 63.515.0 years , 38.5% were female , 27.1% were diabetic , 90.6% had a native fistula , and 9.4% had a graft . median dialysis vintage was 43.2 months ( minimum : 0.1 months ; maximum : 419.6 months ) . access location was lower arm for 51.2% of patients . during the follow - up , prevalent needle sizes were 15 and 16  g for 63.7% and 32.2% of the patients , respectively ( 14  g : 2.7% 17  g : 1.4% ) . in spain , 98% of patients were treated with 15-g needles , and in romania 75% of patients were treated with 16-g needles . cannulation technique was area for 65.8% , rope - ladder for 28.2% , and buttonhole for 6% of patients , with some country preferences clearly visible : area technique was applied in as much as 77% of patients in romania , and rope - ladder was more common in poland than in the total study population ( 44% ) . the direction of arterial puncture was antegrade for 57.3% of patients ; this was the preference for 99% of patients in poland . the bevel orientation was upward for 70.2% of the patients , peaking in poland with 95% . the practice of needle rotation after insertion was practiced for 42% of patients , with a much higher percentage in italy ( 82% ) . the prevalent combination between arterial needle puncturing and bevel direction was antegrade with bevel upward ( 43.1% ) , followed by retrograde with bevel up ( 27.1% ) . the proportion of the two other combinations , that is , antegrade and retrograde with bevel downward , was 14.2% and 15.6% , respectively . the 15.6% with retrograde and bevel down were mainly treated in two countries ( spain and portugal ) . median blood flow was 350400  ml / min . in italy and spain , 40% and 38% of patients conversely , in slovenia and in poland 5455% of patients were treated with blood flows below 300  ml / min . figure 3 shows the distribution of patients according to the prescribed needle size , blood flow , and venous pressure levels . the primary outcome event ( i.e. , surgery for a new va during the follow - up period ) was observed in 1485 patients ( 21% ) . univariate survival analysis revealed a significant benefit for access survival for patients who are younger , nondiabetic , male , have lower body mass index , do not take platelet antiaggregants , do not have heart failure , and are able to assist with compression . a significant benefit was also seen for patients with fistula ( vs. graft ) , smaller needles , distal location of the access , and low venous pressure . with regard to cannulation technique , positive effects were observed for antegrade needle direction ( vs. retrograde ) , bevel up ( vs. down ) , nonalcohol - based disinfection , and application of local anesthesia . although not statistically significant , a potential survival benefit was indicated for higher blood flow ( p=0.056 ) and buttonhole technique ( vs. rope - ladder and area , p=0.11 ) . needle rotation did not affect the access survival ( p=0.81 ) , neither did access vintage ( age &lt; 1 month before baseline vs. 1 month before baseline ; p=0.29 ) . meier access survival curves according to blood flows , venous pressures , needle sizes , and cannulation techniques are presented in figure 4 . in a second step , after adjustment for age , gender , diabetes , va type , access location ( proximal vs. distal ) , dialysis vintage and heart failure , and incorporation of country differences , the use of a 16-g needle was associated with a significantly higher risk of access failure ( hazard ratio ( hr ) 1.21 ) compared with the use of a 15-g needle . very few ( 1.4% ) patients were treated with the even smaller 17-g needles , but the direction of the results is the same , that is , increased hr for smaller needle size . using a blood flow of 300350  ml / min as a reference , the hr tended to decrease as the blood flow increased . with regard to cannulation technique , both rope - ladder and buttonhole techniques performed significantly better than the area technique . considering antegrade with the bevel up as reference , the retrograde direction of the arterial needle with bevel down is associated with a significant increase of access failure risk of 18% . all other options , that is , antegrade direction with bevel down or retrograde direction with bevel up , were not associated with a hr significantly different from 1.00 . with regard to venous pressure , using as reference the range between 100 and 150  mm  hg , the hrs increased proportionally to 1.4 , 1.87 , and 2.09 with the increase of venous pressure from 150 to 200  mm  hg , 200 to 300  mm  hg , and &gt; 300  mm  hg , respectively ( all p0.008 ) . of note , venous pressures of &gt; 300  mm  hg are extreme cases and were only recorded in 0.6% of the patients . in addition , a venous pressure of &lt; 100  mm  hg was associated with a significantly higher hr of 1.51 . to investigate this further , we also looked for interaction effects between blood flow and venous pressure , as well as between arterial and venous pressures ; no significant associations were found . finally , the use of a tourniquet and not applying any pressure at the time of cannulation were associated with hrs of 1.30 and 1.25 ( p&lt;0.008 and &lt; 0.02 ) , respectively , compared with exertion of arm compression by the patient at the time of cannulation ( labeled  patient assistance ' in table 1 ) . in summary , this study revealed that area cannulation technique , albeit being identified as the most commonly used technique in this population of over 7000 patients , was inferior to rope - ladder and to buttonhole for maintenance of va functionality . with regard to the effect of needle and bevel direction , the combination of antegrade positioning of the arterial needle with bevel - up orientation was significantly associated with better access survival than retrograde positioning with bevel down . the use of larger needles tended to favor access patency , with 15  g being superior to 16 or 17  g. the application of arm pressure by the patient at the time of cannulation had a favorable effect on access longevity compared with not applying pressure or using a tourniquet . results pertaining to the type and location of the access and the technical parameters ( i.e. , blood flow and venous pressure ) were as follows : there was an increased risk for access failure for grafts vs. fistulas , proximal location vs. distal , right arm vs. left arm , blood flows below 300  ml / min vs. those in the range of 300350  ml / min , and for the presence of a venous pressure &gt; 150  mm  hg vs. pressures between 100 and 150  mm  hg . tissue reparative processes triggered by cannulation procedures may cause enlargement of the fistula and the formation of aneurisms and scars that , in turn , can favor the development of stenotic lesions and ultimately impact fistula survival . repetitive punctures at the va site cause vessel wall defects that are initially filled by thrombi before finally healing . of the three cannulation techniques , the buttonhole approach has the theoretical advantage of limiting the process of dilatation and fibrosis because the thrombus is displaced while being formed , favoring the formation of a cylindrical scar from the subcutaneous and vessel wall tissues . the rope - ladder technique may have the initial advantage of favoring progressive maturation along the entire length of the fistula , but it requires fistula with sufficiently long segments suitable for cannulation . the area puncture technique weakens the fistula wall and is associated with the least favorable consequences , that is , localized dilation , disruption of the vessel wall , and subsequent development of ( pseudo)aneurysms and strictures . despite this and the fact that area cannulation has been discouraged for over two decades , it was disheartening to observe that this was the predominant practice in almost two - thirds of patients . according to the ebpg and the clinical practice guidelines for va , the rope - ladder technique should be used for cannulation of grafts . specifically , according to the latter , this study showed a 22% lower risk for va failure in those patients whose va was cannulated with the buttonhole technique as opposed to area , confirming the results of a recently published randomized controlled clinical trial . although the buttonhole technique is associated with good results , one should also take into consideration that it is a practice performed in centers with highly trained personnel that work with strict protocols and that it may also be used for fistulas with only short segments available for cannulation . in our study , this practice is used in 22 centers , mainly in portugal , turkey , the united kingdom , and italy . research questions that arise from current guidelines address the effectiveness of structured cannulation training , increased remuneration for expert cannulators , and whether self - cannulation can lead to better outcomes . indeed , as buttonhole cannulation requires the designation of a reference nurse , especially for the initial 46 weeks , it is likely that this technique benefits from its association with centers offering the necessary training ( i.e. , centers capable of stemming the increased organizational effort and assigning the right cannulator to the right patient ) . in addition , once the tunnel is created , cannulation can be performed directly by patients . however , irrespective of the influence of cannulator training and center organizational issues , the underlying question to be addressed , optimally in a well - designed clinical study , is which cannulation techniques can be recommended to ensure long - term va functionality . this study showed that retrograde direction of arterial needle with bevel down is associated with the least favorable outcome . this is consistent with the findings of woodson and shapiro who reported that retrograde puncturing may be associated for an increased risk of hematoma formation , possibly owing to the related venous return of the blood ( i.e. , retrograde filling ) . antegrade puncturing , on the other hand , may be considered fistula - protective by the same reasoning , that is , tract closure through flow force . therefore , retrograde direction of the arterial needle is more likely to be associated with a higher risk for aneurism . despite recommendations by kdoqi to rotate the needle during insertion , the univariate analysis performed here found no evidence of any benefit of this practice . on the contrary , the authors share the opinion of many cannulators that the 180 rotation of the needle is unnecessary and may constitute an additional trauma to the va . further studies are needed to clarify whether rotation of the va needle during cannulation should be recommended or not . there are a number of possible reasons for the association of the higher failure risk with smaller needle sizes . while increased trauma and prolonged bleeding time are generally associated with the use of large needles , the use of small needles at the same blood flow results in a higher speed of the blood returning to the vasculature , possibly damaging the intima of the avfs . for example , at an operative blood flow of 350  ml / min , the maximum speed of the injected blood will be 8.79  m / s with a 17-g needle and 5.80  m / s with a 15-g needle ( presented by ralf jungmann at vascular access coursestockholm , 1112 october 2012 , stockholm , sweden ) . furthermore , the shear forces created by returning blood can have a role in inflammation and stenosis formation . stenotic fistula and graft lesions are associated with the induction of the expression of profibrotic cytokines , local inflammation , and neointimal proliferation . however , we can not exclude that this association may be a consequence of bias by indication . needles of smaller inner dimension are generally prescribed not only for a new va but also for problematic avfs , that is , those likely to fail in the following months . therefore , it is difficult to derive a conclusion from this association , but on the basis of figure 3 , 17-g needles are clearly linked to blood flow levels below 300  ml / min and , on the contrary , 14-g needles are mainly prescribed to patients with 350400  ml / min or greater blood flows . it is also of interest to underline that higher venous pressure is mainly associated with the 16-g needles , which have a wider distribution of different blood flows . measurement of venous pressure during dialysis is currently used as a surveillance tool within the dialysis session , and not as a standard monitoring strategy . this study showed a significant and proportionally increasing risk of va failure with venous pressures higher than 150  mm  hg . an increased hr was also detected for venous pressure below 100  mm  hg . as shown in figure 3 , an association between needle size , blood flow , and venous pressure is indicated in that for needle gauges 15 , 16 , and 17 low venous pressures appear to be associated with low blood flows . such an association could be an indication of stenosis in the artery . venous pressure is crucially dependent on the characteristics of the needle ( e.g. , the needle gauge , the length of the metallic portion , and the length and the thickness of the needle shaft ) , which vary among manufacturers . in this network , at the time of the study , the vast majority of the needles ( 85% ) were from a single producer and the length of the needle was 25  mm . the unexpectedly high hr associated with a venous pressure of under 100  mm  hg compared with 150200  mm  hg should motivate reflection on the currently accepted limits . one could consider integration of venous pressure monitoring into an algorithm for the detection of increased risk of access failure . this study has certain limitations over and beyond those inherent to observational studies , for example , that residual confounding can not be completely ruled out . being a retrospective study , patient data for those patients on dialysis before admission to the nephrocare clinic were not collectable , and thus robust information on the number of prior vaes , on their respective lengths , and on first cannulation was not available . particularly , the missing information on the length of the va , its depth , and the access flow constitute a major weakness because a particular cannulation technique could have been chosen on the basis of what is possible with the given access characteristics . in addition , the length of the access can influence the way in which the needles are placed . despite these missing data , we feel that this study has its merits , as it shows that traditional local practices have a significant influence on procedures exercised . a further limitation is that the va practice was surveyed in april 2009 and was assumed not to have been changed during the follow - up ( 31 march 2012 ) . however , as nursing practices in this field are strongly related to the clinic culture and experience , we have reason to believe that it is should not constitute a significant bias . of course , some cannulation particulars , such as needle size and arterial blood flow , may vary over time , in that smaller needle sizes and low blood flow rate are used for initial access use and that large needles are taken for mature accesses . however , we feel that the model selected here is also justified because it is an explanatory model , based on the association of baseline characteristics with access survival . other limitations are that we had follow - up of 65% of the patients and that most countries were in europe ( owing to deployment of the electronic reporting system ) . as reported , an association between clinical practice patterns and country has been detected , and consequently not all different practices were covered by our model . however , according to the results of this analysis , each country has a combination of practices that positively and negatively influence the va survival . for example , in romania , positive influences were the puncture direction being antegrade ( 82% ) , bevel orientation being predominantly upward ( 95% ) , and needle not rotated ( 84% ) ; negative associations were the use of area technique ( 77% ) , preferred needle size ( 75% with 16  g ) , and the use of blood flows &lt; 300  ml / min ( 47% ) . for this reason , intracountry correlations were considered using a sandwich estimator in the multivariate model . to assess the influence of individual center practices , we also performed a sensitivity analysis by applying the sandwich estimator at the center level . there were only negligible differences to the results obtained with the original model at the country level , raising our confidence that there is no severe confounding of the model by center practice effects . given the relevant impact of the investigated variables on the survival of the va , itself a key driver of hemodialysis patient survival , we believe it is time to organize a large - scale randomized clinical trial to facilitate the formulation of practical and comprehensive cannulation practice guidelines . as the associations between practice patterns and va survival reported here are mainly related to national procedures and only partially related to actual patient limitations , they offer some promising indications for improving clinical practice . in april 2009 , a cross - sectional survey was conducted in 171 dialysis units located in europe , the middle east , and africa to collect details on va cannulation practices on a clinic by clinic level . all patients who were on double - needle hemodialysis or online hemodiafiltration during the week of the survey were selected for analysis , as long as a fistula or graft was used for va , survey data were complete , and follow - up data were available in our clinical database . primary outcome was time until the first surgical access intervention resulting in the generation of a new access ( i.e. , as opposed to any surgical intervention done just for revision , thrombectomy , etc . , or any endovascular intervention ) . patients were censored for transplantation , death , loss of follow - up , or end of the follow - up period ( 31 march 2012 ) . information on cannulation retrieved from the survey comprised fistula type and location , cannulation technique , needle size , needle and bevel direction , needle rotation , blood flow , arterial and venous pressure , use of disinfectants , use of local anesthesia , and application of arm compression at the time of cannulation . to adjust for individual patient characteristics , the following information was extracted from the clinical database : patient age , gender and body mass index , prevalence of diabetes , and the use of ace inhibitors , platelet antiaggregants , and anticoagulants . in addition , the median blood flow prescription was documented at a center level at the time of the survey . for univariate analysis , kaplan  meier curves were calculated and comparisons were performed using the log - rank test . by combining univariate results with medical and statistical experience , a set of variables for multivariable analysis was determined . in particular , specific interaction terms ( e.g. , bevel vs. arterial needle direction ) were defined for statistical examination , and decisions were made regarding their inclusion or omission in the cox model depending on their significance or collinearity , respectively . a final cox model based on these variables was calculated , using the sandwich estimator to account for within - country correlation . step by step , the final model was reduced , setting a p - value of 0.1 for variable inclusion . all analyses were performed with sas v9.2 ( sas institute , cary , nc ) .</td>\n",
       "      <td>&lt;S&gt; hemodialysis patient survival is dependent on the availability of a reliable vascular access . in clinical practice , &lt;/S&gt; &lt;S&gt; procedures for vascular access cannulation vary from clinic to clinic . &lt;/S&gt; &lt;S&gt; we investigated the impact of cannulation technique on arteriovenous fistula and graft survival . &lt;/S&gt; &lt;S&gt; based on an april 2009 cross - sectional survey of vascular access cannulation practices in 171 dialysis units , a cohort of patients with corresponding vascular access survival information was selected for follow - up ending march 2012 . of the 10,807 patients enrolled in the original survey , access survival data were available for 7058 patients from nine countries . &lt;/S&gt; &lt;S&gt; of these , 90.6% had an arteriovenous fistula and 9.4% arteriovenous graft . &lt;/S&gt; &lt;S&gt; access needling was by area technique for 65.8% , rope - ladder for 28.2% , and buttonhole for 6% . &lt;/S&gt; &lt;S&gt; the most common direction of puncture was antegrade with bevel up ( 43.1% ) . &lt;/S&gt; &lt;S&gt; a cox regression model was applied , adjusted for within - country effects , and defining as events the need for creation of a new vascular access . &lt;/S&gt; &lt;S&gt; area cannulation was associated with a significantly higher risk of access failure than rope - ladder or buttonhole . &lt;/S&gt; &lt;S&gt; retrograde direction of the arterial needle with bevel down was also associated with an increased failure risk . &lt;/S&gt; &lt;S&gt; patient application of pressure during cannulation appeared more favorable for vascular access longevity than not applying pressure or using a tourniquet . &lt;/S&gt; &lt;S&gt; the higher risk of failure associated with venous pressures under 100 or over 150  mm  hg should open a discussion on limits currently considered acceptable . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>health is not only related to the absence of the disease , therefore we need to conceptualize and operationalize what health is . increasingly , we have come to understand that information about functional status is needed in order to appreciate the full picture regarding the health of an individual or a population . an individual 's health fundamentally includes their capacity to carry out the full range of actions , activities and tasks required to fully engage in all areas of human life . the health state of a person can be described in terms of capacity to carry out a set of tasks or actions . in addition , the health state also includes changes in body functions and/or structures arising from a health condition . the impact of the health state on a person 's life can be understood by measuring performance of tasks and actions in the person 's real - life or actual environment . the full picture of the health experience can further be appreciated by taking into cognizance the value that people place on levels of functioning in given domains in association with a health condition . plainly , the concept of functional status is integral to health and its achievement . two individuals with identical diagnoses may have utterly different levels of functioning that determine their actual health status . without fsi , our picture of the health of an individual , or a population , is flawed and incomplete . fsi has , of course , long been collected in various ways and used clinically , especially in rehabilitative medicine ; physical , occupational and speech and language therapy ; and in nursing home and home care settings . fsi is essential for needs assessment as well as the development and monitoring of rehabilitative interventions to restore or maintain functions . it is also essential in this area of health care because the aim of therapy is to assist patients in maximizing their capacities to perform activities needed for their lives . although no one doubts that restoring functioning is restoring health ( the ultimate purpose of all forms of health care ) some clinicians , focusing exclusively on acute - care needs , do not see the need to collect or utilize fsi . in most countries with a sophisticated health administrative data collection and utilization infrastructure , a wide variety of information what is often missing is information that would link diagnosis and treatment with health outcomes that are fully meaningful to the patient 's life , namely information about the presence of decrements in capacity to carry out tasks and actions in areas of life as well as how these decrements play out in the person 's actual , real - life environment ( deyo and patrick , 1989 ; lubetkin et al . , 2003 ) . there is growing recognition that there is a gap in health administrative records : the failure to collect or disseminate fsi across all health care settings . unless fsi becomes an essential part of administrative records , the potential value of these data will be lost , not merely to clinicians , but to health administrators concerned about management and quality of care issues , health researchers , and public health agencies . this insight is clearly expressed in a report by the national committee on vital and health statistics ( ncvhs ) ( 2001 ) :  without functional status information , the researchers , policymakers , and others who are already using administrative data have at best a rough idea of how people , individually and collectively , are doing and at worst they are making erroneous assumptions and decisions .  the report outlines in some detail the benefits of routinely collecting fsi across the entire health care delivery system and throughout all care settings . fsi can serve management needs of all the stakeholders in the health care system  clinicians , providers , payers , patients , and government regulatory bodies . this is true especially with respect to evaluating outcomes , comparing treatment modalities , and predicting and managing costs . this links directly to debates of modes of service provision , single or multiple payer , managed care , fee - for - service , or some hybrid mixture . the policy and research applications of fsi are evident for local health management and quality control , and in the broader arena of public health . policy decisions about priorities must be made at the level of individual clinics or hospitals , local or regional health care agencies , or at the level of government planning and budgeting . given the importance of getting the complete picture of health outcomes , fsi is an essential input into evidence - based policy decisionmaking . researchers in all areas of health and social policy , at all levels , need valid and reliable data about functional status in order to make informed decisions . for example , it is a matter of debate whether , as the world 's population lives longer and ages , they will be unhealthy and pose a greater burden on health systems . there is some evidence suggesting that elderly persons today are functioning at higher levels than before . without reliable information on levels of functioning , this debate would be unresolvable because it would not be possible to detect functional status , since the disease morbidity may not have changed very much . compression of morbidity occurs when disability or decrement in functioning is postponed more than longevity is extended , as for example with the effects of exercise or better eating habits . the direct test of compression ( or extension ) of morbidity depends on the effects of reduced health risks on cumulative lifetime disability ( fries , 1980 , 2001 ; vita et al . , , fsi is a crucial element for the description of health states and quantification of overall health status in individuals that can be aggregated to a summary measure of population health . at the who , the use of fsi is in this area , in particular , because this data ( collected by the world health survey now in the field in more than 70 countries ) feeds into ongoing endeavors to determine levels and distributions of health this survey would be inconceivable without information on health outcomes that describe health on multiple dimensions in terms of levels of functioning in a parsimonious set of domains . it is commonly known that the demographic trends toward an older population , at least in developed counties , will create unprecedented burdens on all age - sensitive social policies , such as social security and other pensions , retirement , unemployment , and long - term care . aging , according to a recent organization for economic cooperation and development ( oecd ) ( 2001 ) report , is the principal factor currently driving pension spending costs . since age - sensitive social programming constitutes between 40 and 60 percent of total public spending , the impact of aging is considerable . to comprehend the nature and magnitude of its social impact , those responsible for policies from transportation and housing to employment and taxation , will need reliable data on functional status and how it plays out in the lives of the aging population . for fsi to be available for this wide variety of uses , however , it must be routinely and consistently collected across the entire health care delivery system , preferably in some electronic format . nonetheless , before contemplating the systemwide changes required to collect fsi , a classification that provides a common language and framework to describe the universe of functioning and disability is required . in order to complement the classification scheme , a comprehensive coding system that creates consistent and comparable data across all settings of care and a method of routinely capturing and disseminating these data ( in a mode and manner consistent with social interests in preserving privacy ) linked to measurement tools for clinical and related encounters the foundation of a new structure for collecting fsi is , therefore , a standard classification and coding system that will make it feasible for fsi to be included in administrative data . as the ncvhs report stated :   while the international classification of diseases ( icd ) has served us well for more than a century in characterizing diagnoses , it is now time to complement it with a parallel system for characterizing functional status .  although the committee argued that more research , analysis , testing , and demonstration projects are required before final recommendations can be made , it concluded that :  the concepts and conceptual framework of the icf have promise as a code set for reporting functional status information in administrative records and computerized medical records . in the committee 's view , the icf is the only existing classification system that could be used to code functional status across the age span .  in this article , we want to briefly describe the extensive international developmental process that lead to the revision of the original international classification of impairments , disabilities and handicaps ( icidh ) ( world health organization , 1980 ) and produced the icf . we also want to describe the basic principles and structure of the icf , in particular , to show its value in the context of collecting fsi for administrative records . the primary mandate of who is the production and dissemination of reliable and timely information about the health of populations . who 's 1947 constitution requires that :  each member shall provide statistical and epidemiological reports in a manner to be determined by the health assembly .  countries have long reported causes of death or mortality statistics based on who 's ( 1992 ) \\n international statistical classification of diseases and related health problems ( icd-10 ) . though useful for calculating life expectancy for different countries , however , who recognized that these data did not capture the overall health status of living populations . missing was information about non - fatal health outcomes , i.e. , functioning and disability across all areas of life . to meet this need , who ( 1980 ) issued a tool for the classification of the consequences of disease , namely the icidh . a considerable academic literature built up around clinical and other uses of the icidh , but much of this literature was critical of the underlying model of disability . responding to these critiques and an international call for an updated version , who launched a revision process in 1993 to address what many viewed as an urgent international need for a framework for measuring and reporting the health as functional status at both individual and population levels . over the next 10 years , who 's international collaborating centers and governmental and non - governmental organizations , including groups representing persons with disabilities , engaged in the systematic revision of the icidh . from an exhaustive literature search of existing classifications and assessment tools , the who revision team developed a 3,000-plus item pool of potential classification domain names for areas of human functioning at the body , person , and social levels . all efforts were made to ensure that the icidh-2 , as it was initially named , would be a suitable classification for all domains of functioning associated with both physical and mental health conditions . adopting the strategy of computer software development , alpha and beta drafts were prepared from 1996 forward . the original 1980 icidh had only been approved for field - trial purposes . in light of that , the who team felt for icidh-2 to have the necessary credibility and legitimacy to serve as the international standard language of health and functioning , that the revision process should include several years of field trials and other tests . the first phase of field trials concentrated on the cross - cultural and linguistic applicability of the model and classificatory structure and language of the icidh-2 . the intent of this phase of field trials was to establish the conceptual and functional equivalence of the items contained within the classification . stn et al.(1999a , b ; 2000 ) provide the rationale for the methodologies and presentation and analysis of the 15-country field trials . these results fed into further international collaboration in which the who team relied on a global network of who collaborating centers , non - governmental organizations , disability groups , and individual experts and key informants . the next revision phase began in 1999 when a series of expert drafting teams were assembled in geneva to produce the beta 2 draft . this draft was used for the second round of international field trials , these focusing on questions of reliability , utility , and feasibility of use . once the results of these tests were collected and analyzed , a pre - final draft was produced in early fall 2000 as a result of an intensive editing process grounded in the expert input being received from around the world . the icidh-2 , unlike its predecessor , was from the outset developed in multiple languages , primarily to identify and respond to cross - cultural and linguistic differences that might affect the usefulness of the classification . the collaborating centers and others provided constant input at this stage as the language and classification structures were redrafted and refined in multiple iterations . the draft was put on the internet for comment from a wide range of individuals , including both providers and consumers . after presentation before the executive board in december 2000 , the classification was put on the agenda of the fifty - fourth world health assembly and renamed the icf . the new title reflected the philosophy of moving beyond the consequence of disease approach and highlighted functioning as a component of health . in may 2001 , it was unanimously endorsed , member states were urged   to use the icf in their research , surveillance and reporting as appropriate .  with its approval , the icf became a member of the who family of international classifications . whereas icd-10 provides the codes for mortality and morbidity , icf provides the codes to describe the complete range of functional states that capture the complete experience of health . the icd-10 and icf are , therefore , complementary and who encourages users to utilize both together , wherever applicable . this will ensure a more meaningful and complete picture of the health of people or populations . soon after its official release , who 's director general , gro harlem bruntland , announced that the icf is who 's framework for measuring health and disability at both the individual and population levels . who has already implemented icf as the basis for its extensive world health survey program , demonstrating its use as a global and universal tool . to improve health , tools are needed to measure health , and in particular to measure the changes in health brought about by interventions .  icf is the ruler with which we will take precise measurements of health and disability .  ( brundtland , 2002 . ) from the public health perspective , the usefulness of icf goes beyond that of the measuring of population health and the effectiveness of internationally coordinated interventions funded by initiatives , such as the global fund to fight aids , tuberculosis and malaria . in addition , with the icf as their framework , countries will be able to identify social factors such as education , transportation , or housing , both as determinants of health , and social factors influenced by improvements in health . making these links will further support the relationship between health and economic development . in short , we have   in the shape of a little red book , an extraordinarily versatile tool  a swiss army knife for health ministries , researchers and decision - makers .  ( brundtland , 2002 . ) undoubtedly the primary reason that icf can plausibly claim to be a universal tool for classifying states of functioning and disability is that the underlying model of the icf reflects our best understanding of the complex phenomena of functioning and disability in a manner that is , to the greatest extent possible , theory - neutral and therefore compatible with whichever theoretical account of how disability arises , at the individual and population levels , that evidence may confirm . it is the conceptual basis for the definition , measurement , and policy formulations for all aspects of disability . a paradigmatic shift in the thinking with regard to disability that is captured in the icf is the stress placed on health and levels of functioning . heretofore , disability has been construed as an all or none phenomenon : a distinct category to which an individual either belonged or not . the icf , on the other hand , presents disability as a continuum , relevant to the lives of all people to different degrees and at different times in their lives . disability is not something that happens only to a minority of humanity , it is a common ( indeed natural ) feature of the human condition . the icf is for all people , not just people traditionally referred to as disabled and isolated as a separate group . icf thus mainstreams the experience of disability and recognizes it as a universal human experience . by shifting the focus from cause to the full range of lived experiences , it places all health conditions on an equal footing , allowing them to be compared using a common metric  the ruler of health and disability . from emphasizing people 's disabilities , and labeling people as disabled , we now focus on the level of health and functional capacity of all people . decrements in functioning may be the result of decrements in intrinsic capacity or problems with body functions or structures ; or they can result from features of the person 's physical , human - built or social environment that lead to problems in performance over and above decrements in capacity . very likely , decrements in functioning are the result of both processes . yet , the extent to which intrinsic decrements in capacity or environmental factors are the cause is not a matter that can be determined a priori . moreover , icf is grounded in the principle of universality , namely that functioning and disability are applicable to all people , irrespective of health condition , and in particular that disability  or decrement in functioning at one or more levels  is not the mark of a specific minority class of people , but is a feature of the human condition , which is , epidemiologically speaking , over the lifespan , a universal phenomena . in addition , icf is committed to the principle of parity , which states that the functional status is not determined by background etiology , and in particular by whether one has a physical rather than mental health condition . much time , effort , and international collaboration has gone into the development of the icf . it is no longer plausible to insist that the icf is a medical classification of people with disability , that it reduces all issues of functional status to underlying medical conditions , that it ignores the often salient role of the physical and social environment in the creation of restrictions of participation experienced by persons with functional problems . the revision process has produced a classification that has already stood up to rigorous tests of validity , reliability , and cross - cultural applicability . it is , as the ncvhs has concluded ,   the only existing classification system that could be used to code functional status across the age span .  we now turn to the structure of icf as a classification system , in part to show why the committee has correctly assessed the value of the icf as a coding system for functional status , suitable for use in administrative records . the model that informs icf , portrays functioning and decrements in functioning , or disability , as a dynamic interaction between health conditions ( diseases , disorders , and injuries ) and contextual factors . contextual factors include environmental factors , that is , all aspects of the physical , human - built , social , and attitudinal environment that create the lived experience of functioning and disability . although not classified in icf , contextual factors also include personal factors such as sex , age , coping styles , social background , education , and overall behavior patterns that may influence how disability is experienced by the individual . the terms functioning and disability in the icf are the general or umbrella terms for , respectively , the positive and negatives aspects of the interaction between an individual ( with a health condition ) and that individual 's contextual factors ( environmental and personal factors ) . in the icf , health condition is the umbrella term for disease ( acute or chronic ) , disorder , injury or trauma . a health condition may also include other circumstances such as pregnancy , aging , stress , congenital anomaly , or genetic predisposition . the icf interactive model identifies three levels of human functioning : functioning at the level of body or body part , the whole person , and the whole person in their complete environment . these levels in turn define three aspects of functioning : body functions and structures , activities , and participation . disability similarly denotes a decrement in functioning at one or more of these levels  that is , an impairment , activity limitation or participation restrictions . table 2 shows the complete list of all of the chapters found in the three classifications included in icf . under each of these chapters are second , third , and in some instances , fourth levels of categories , arranged in a hierarchical , tree - branch - stem - leaf , arrangement . this structure makes it possible for icf to be used as a classification tool for systematically describing situations of human functioning and problems with functioning . this complex information is organized by icf by means of a hierarchical coding system , thereby creating a common international language for functioning and disability . icf organizes information by means of several classifications distributed into two parts : ( 1 ) a component of functioning and disability that includes the component of the body with the body function and body structure classifications , and the component of activities and participation that includes all domains denoting aspects of functioning from an individual and social perspective organized into a single classification , and ( 2 ) a component of contextual factors that has a list of environmental factors organized from the individual 's most immediate to the wider environment . the classifications in the first part identify all of the domains of functioning  from basic physiological functions and body structures , to simple and complex actions , tasks , social performances and relationships . the environmental factors list provides a tool for identifying those features of a person 's physical , human - built , social and attitudinal environment that , in interaction with the domains of functioning , constitute the complete lived experience of human functioning and disability . within the contextual factors part , besides the environmental factors , the icf recognizes the existence of personal factors as another component , but provides no classification of these . domains are a practical , meaningful set of related physiological functions , anatomical structures , actions , tasks , or areas of life . domains make up the different chapters and blocks within each component ( world health organization , 2001 ) . in order for these domains to capture descriptive information about functioning and disability in particular cases , they must be used in conjunction with qualifiers that record the presence and severity of a problem or decrement in functioning at the body , person , and social levels . for the classifications of body function and structure , the primary qualifier indicates the presence of an impairment and , on a five - point scale , the degree of the impairment of function or structure ( no impairment , mild , moderate , severe , and complete ) . in the case of the activity and participation list of domains , two essential qualifiers are provided to capture the full range of relevant information about disability . the performance qualifier is used to describe what an individual does in their current or actual environment , including whatever assistive devices or other accommodations the person may use to perform actions or tasks and whatever barriers and hindrances exist in the person 's actual environment . because the current environment always incorporates the overall social context , performance might be understood as involvement in the lived experience of disability . the capacity qualifier describes an individual 's inherent ability to execute a task or an action . operationally , this qualifier identifies the highest probable level of functioning of a person in a given functional domain at a given moment without any specific assistance . for measurement purposes , this level of capacity presumes a standardized assessment environment , namely one that reveals the inherent capacity of a person in a specific functional domain without any particular enhancements . the environmental factors list can be used to describe such a standard assessment environment in order to ensure that results across different studies can be compared by holding this environment constant . intuitively , the performance qualifier captures what people actually do in their lives , whereas the capacity qualifier identifies the person 's inherent capacity without explicit environmental facilitation ( or hindrance ) . who is developing a standard application guide that will operationalize the constructs of capacity and performance with respect to individual items that form the classification . table 3 shows how data can be organized to reflect the role of these two qualifiers used for the domains of the activity and participation classification . as a general matter of describing functioning and disability phenomena fully and accurately , the performance / capacity having access to both performance and capacity data enables icf users to determine the gap between capacity and performance . if capacity is less than performance , then the person 's actual or current environment has enabled him or her to perform better than what data about their capacity would predict : the environment has facilitated performance . on the other hand , if capacity is greater than performance , then some aspect of the environment is acting as a barrier to a level of performance that is feasible in a more suitable environment . icf thus makes it possible to measure the effect of a person 's environment on their decrement in functioning , given their health condition . the environmental factors classification can be used to identify specific features of the person 's actual environment that are barriers or facilitators in general for the person or with specific regard to each item of the person 's body functions , body structures or activities and participation that have been described . it can also be used , as previously stated , to describe specific testing environments where capacity has been measured . for its use as a classification of functional status relevant for health administrative records , icf provides a complete classification of both body and person level domains of functioning . given that it has been designed for a multiplicity of uses and users , there is far more in icf than could ever be plausibly integrated into a viable coding system for health records , although it remains the ultimate lexicon to which any coder , for clinical or research purposes , could turn . clearly , for implementation purposes in this area , a simplified checklist of items is needed . such a checklist was produced and used during the beta 1 and 2 field - testing phase in the revision process ( world health organization , 2001 ) . this checklist , which takes less than 30 minutes to complete , is currently being extensively tested in clinical studies in different disorders in order to study its feasibility , reliability , and concurrent validity with existing assessment instruments as part of a larger project to define core sets of items that may be used in rehabilitation settings for specific conditions and across several disorders ( stucki et al . , 2002 ) . the core sets of items with their corresponding scales could also be then converted into even shorter assessment instruments . the challenge for incorporating the icf into clinical and administrative records beyond a lexicon and framework lies in identifying this parsimonious set of domains or items that captures decrements in functioning across different health conditions and a smaller subset of domains or items that uniquely describe the decrements of functioning that typify a given health condition . in addition , the mapping of instruments ( that measure functioning and disability that are already in use ) onto icf categories will allow a ready crosswalk between measurements already being made at points of encounter to a common framework ( cieza et al . , 2002 ) . the use of the icf in larger population based surveys will also provide data on norms and distributions of health , functioning and disability that will enable the setting of appropriate thresholds for a multitude of purposes . table 4 maps the domains of the icf that have been included in different waves of the world health survey that ought to be included as a minimum or ideal set for information systems . these domains are also included on the icf checklist , which is designed to be a clinical tool . primary data collection strategies with regard to functional status , in a manner that is truly comparable , are in their infancy especially for international use and for use across population groups . further tools need to be developed , and standards and procedures established , so that these data become meaningful and usable . as a final issue , it must be mentioned that the icf has been conceived as a dynamic classification that will not only serve multiple users requiring different levels of detail , but also will continue to evolve with advancements in science . the classification is flexible in its structure such that it can be expanded in the level of detail ( for example , the fourth level ) for specific uses , or new codes added where gaps have been left in the numbering system . a set of operational rules will specify the procedure for this evidence - based expansion , adaptation , or revision of the classification . a common language for describing fsi is the key to ensuring comparability of data from a myriad of sources as well as in providing users with a tool for precise and accurate communication with each other . the recognition that a description of health and health - related outcomes must go beyond a narrow view of health restricted to the absence of disease , as well as that the definition of disability must move beyond the narrow impairment - based view that has been traditionally adopted to define a minority population , will go a long way in bridging the gap between health and disability data . it will also fill the void in existing health outcomes data while measuring the impact of interventions and monitoring them over time . health records must include functioning information in order to ensure a complete description of health states . the icf is the common language and framework that users will employ from now on . in the same way that all languages grow , evolve , and flourish over time and are adapted and modified to express new ideas , the icf will have a multitude of applications where it will be creatively used such that it continues to be a living classification . as with all new languages , it will be important to develop tools to learn this new language . toward this end , who is developing standardized application manuals and web - based learning courses that will use state - of - the art pedagogic methodology to assist end users . its usefulness in describing functional health status information will be one of the measures of its success .</td>\n",
       "      <td>&lt;S&gt; a common framework for describing functional status information ( fsi ) in health records is needed in order to make this information comparable and of value . &lt;/S&gt; &lt;S&gt; the world health organization 's ( who 's ) international classification of functioning , disability and health ( icf ) , which has been approved by all its member states , provides this common language and framework . the biopsychosocial model of functioning and disability embodied in the icf goes beyond disease and conceptualizes functioning from the individual 's body , person , and lived experience vantage points , thereby allowing for planning interventions targeted at the individual 's body , the individual as a whole or toward the environment . &lt;/S&gt; &lt;S&gt; this framework then permits the evaluation of both the effectiveness and cost effectiveness of these different interventions in devising programs at the personal or societal level . &lt;/S&gt;</td>\n",
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
       "model_id": "6c5cb8cb8b5046aa912c1688c181fa02",
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
       "model_id": "9651e5f32b174ba599d54df602d86973",
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
       "model_id": "46b73e09e1d344319faa79bd075c8807",
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
       "model_id": "9a739b169c5747fc9da638bea8cb4927",
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
    "The max input length of `google/pegasus-arxiv` is 1024, so `max_input_length = 1024`."
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
       "model_id": "6987785befd5426298b09380dfd340c4",
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
       "model_id": "b3ce474d632845e9994fa004d40b6d4f",
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
       "model_id": "d5b3e89f7e6f4803b087c23cf591ecc1",
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
       "model_id": "49351798969e49cebcbb19a74864712d",
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
      "Cloning https://huggingface.co/Kevincp560/pegasus-arxiv-finetuned-pubmed into local empty directory.\n"
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
      "The following columns in the training set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: abstract, article. If abstract, article are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
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
       "      [5000/5000 2:21:55, Epoch 5/5]\n",
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
       "      <td>2.650000</td>\n",
       "      <td>1.984835</td>\n",
       "      <td>40.698400</td>\n",
       "      <td>16.387000</td>\n",
       "      <td>25.009700</td>\n",
       "      <td>36.483100</td>\n",
       "      <td>215.294000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>2.131700</td>\n",
       "      <td>1.852423</td>\n",
       "      <td>43.643100</td>\n",
       "      <td>18.679400</td>\n",
       "      <td>26.757100</td>\n",
       "      <td>39.664200</td>\n",
       "      <td>224.646000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>2.059100</td>\n",
       "      <td>1.825366</td>\n",
       "      <td>43.670700</td>\n",
       "      <td>18.517600</td>\n",
       "      <td>26.601500</td>\n",
       "      <td>39.632500</td>\n",
       "      <td>225.894000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>2.010900</td>\n",
       "      <td>1.813821</td>\n",
       "      <td>44.124400</td>\n",
       "      <td>18.886600</td>\n",
       "      <td>26.831300</td>\n",
       "      <td>40.091300</td>\n",
       "      <td>229.656000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>1.989400</td>\n",
       "      <td>1.811831</td>\n",
       "      <td>44.286000</td>\n",
       "      <td>19.047700</td>\n",
       "      <td>27.112200</td>\n",
       "      <td>40.260900</td>\n",
       "      <td>230.586000</td>\n",
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
      "Saving model checkpoint to pegasus-arxiv-finetuned-pubmed/checkpoint-500\n",
      "Configuration saved in pegasus-arxiv-finetuned-pubmed/checkpoint-500/config.json\n",
      "Model weights saved in pegasus-arxiv-finetuned-pubmed/checkpoint-500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-500/special_tokens_map.json\n",
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/special_tokens_map.json\n"
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
      "Saving model checkpoint to pegasus-arxiv-finetuned-pubmed/checkpoint-1000\n",
      "Configuration saved in pegasus-arxiv-finetuned-pubmed/checkpoint-1000/config.json\n",
      "Model weights saved in pegasus-arxiv-finetuned-pubmed/checkpoint-1000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-1000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-1000/special_tokens_map.json\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: abstract, article. If abstract, article are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "Saving model checkpoint to pegasus-arxiv-finetuned-pubmed/checkpoint-1500\n",
      "Configuration saved in pegasus-arxiv-finetuned-pubmed/checkpoint-1500/config.json\n",
      "Model weights saved in pegasus-arxiv-finetuned-pubmed/checkpoint-1500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-1500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-1500/special_tokens_map.json\n"
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
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/special_tokens_map.json\n"
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
      "Saving model checkpoint to pegasus-arxiv-finetuned-pubmed/checkpoint-2000\n",
      "Configuration saved in pegasus-arxiv-finetuned-pubmed/checkpoint-2000/config.json\n",
      "Model weights saved in pegasus-arxiv-finetuned-pubmed/checkpoint-2000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-2000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-2000/special_tokens_map.json\n",
      "Deleting older checkpoint [pegasus-arxiv-finetuned-pubmed/checkpoint-500] due to args.save_total_limit\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: abstract, article. If abstract, article are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "Saving model checkpoint to pegasus-arxiv-finetuned-pubmed/checkpoint-2500\n",
      "Configuration saved in pegasus-arxiv-finetuned-pubmed/checkpoint-2500/config.json\n",
      "Model weights saved in pegasus-arxiv-finetuned-pubmed/checkpoint-2500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-2500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-2500/special_tokens_map.json\n"
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
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/special_tokens_map.json\n"
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
      "Deleting older checkpoint [pegasus-arxiv-finetuned-pubmed/checkpoint-1000] due to args.save_total_limit\n"
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
      "Saving model checkpoint to pegasus-arxiv-finetuned-pubmed/checkpoint-3000\n",
      "Configuration saved in pegasus-arxiv-finetuned-pubmed/checkpoint-3000/config.json\n",
      "Model weights saved in pegasus-arxiv-finetuned-pubmed/checkpoint-3000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-3000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-3000/special_tokens_map.json\n",
      "Deleting older checkpoint [pegasus-arxiv-finetuned-pubmed/checkpoint-1500] due to args.save_total_limit\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: abstract, article. If abstract, article are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "Saving model checkpoint to pegasus-arxiv-finetuned-pubmed/checkpoint-3500\n",
      "Configuration saved in pegasus-arxiv-finetuned-pubmed/checkpoint-3500/config.json\n",
      "Model weights saved in pegasus-arxiv-finetuned-pubmed/checkpoint-3500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-3500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-3500/special_tokens_map.json\n"
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
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/special_tokens_map.json\n"
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
      "Deleting older checkpoint [pegasus-arxiv-finetuned-pubmed/checkpoint-2000] due to args.save_total_limit\n"
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
      "Saving model checkpoint to pegasus-arxiv-finetuned-pubmed/checkpoint-4000\n",
      "Configuration saved in pegasus-arxiv-finetuned-pubmed/checkpoint-4000/config.json\n",
      "Model weights saved in pegasus-arxiv-finetuned-pubmed/checkpoint-4000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-4000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-4000/special_tokens_map.json\n",
      "Deleting older checkpoint [pegasus-arxiv-finetuned-pubmed/checkpoint-2500] due to args.save_total_limit\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: abstract, article. If abstract, article are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "Saving model checkpoint to pegasus-arxiv-finetuned-pubmed/checkpoint-4500\n",
      "Configuration saved in pegasus-arxiv-finetuned-pubmed/checkpoint-4500/config.json\n",
      "Model weights saved in pegasus-arxiv-finetuned-pubmed/checkpoint-4500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-4500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-4500/special_tokens_map.json\n"
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
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/special_tokens_map.json\n"
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
      "Deleting older checkpoint [pegasus-arxiv-finetuned-pubmed/checkpoint-3000] due to args.save_total_limit\n"
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
      "Saving model checkpoint to pegasus-arxiv-finetuned-pubmed/checkpoint-5000\n",
      "Configuration saved in pegasus-arxiv-finetuned-pubmed/checkpoint-5000/config.json\n",
      "Model weights saved in pegasus-arxiv-finetuned-pubmed/checkpoint-5000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-5000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/checkpoint-5000/special_tokens_map.json\n",
      "Deleting older checkpoint [pegasus-arxiv-finetuned-pubmed/checkpoint-3500] due to args.save_total_limit\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: abstract, article. If abstract, article are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
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
       "TrainOutput(global_step=5000, training_loss=2.2452167724609375, metrics={'train_runtime': 8517.4138, 'train_samples_per_second': 1.174, 'train_steps_per_second': 0.587, 'total_flos': 2.885677635649536e+16, 'train_loss': 2.2452167724609375, 'epoch': 5.0})"
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
   "execution_count": null,
   "metadata": {
    "id": "jj7tm3Hvir6_"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Saving model checkpoint to pegasus-arxiv-finetuned-pubmed\n",
      "Configuration saved in pegasus-arxiv-finetuned-pubmed/config.json\n",
      "Model weights saved in pegasus-arxiv-finetuned-pubmed/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-arxiv-finetuned-pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-arxiv-finetuned-pubmed/special_tokens_map.json\n"
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
       "model_id": "35165813c33245a29d483375b46ca5da",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Upload file pytorch_model.bin:   0%|          | 32.0k/2.13G [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
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
