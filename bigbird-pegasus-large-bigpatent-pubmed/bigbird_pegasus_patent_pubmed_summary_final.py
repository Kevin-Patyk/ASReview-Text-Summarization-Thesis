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
      "\u001b[K     |████████████████████████████████| 312 kB 6.9 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting transformers\n",
      "  Downloading transformers-4.17.0-py3-none-any.whl (3.8 MB)\n",
      "\u001b[K     |████████████████████████████████| 3.8 MB 33.5 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting rouge-score\n",
      "  Downloading rouge_score-0.0.4-py2.py3-none-any.whl (22 kB)\n",
      "Collecting nltk\n",
      "  Downloading nltk-3.7-py3-none-any.whl (1.5 MB)\n",
      "\u001b[K     |████████████████████████████████| 1.5 MB 33.0 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: torch in ./miniconda3/envs/fastai/lib/python3.8/site-packages (1.9.1)\n",
      "Requirement already satisfied: ipywidgets in ./miniconda3/envs/fastai/lib/python3.8/site-packages (7.6.4)\n",
      "Collecting pyarrow!=4.0.0,>=3.0.0\n",
      "  Downloading pyarrow-7.0.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (26.7 MB)\n",
      "\u001b[K     |████████████████████████████████| 26.7 MB 34.0 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: requests>=2.19.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (2.26.0)\n",
      "Requirement already satisfied: packaging in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (21.0)\n",
      "Collecting xxhash\n",
      "  Downloading xxhash-3.0.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (212 kB)\n",
      "\u001b[K     |████████████████████████████████| 212 kB 33.9 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: numpy>=1.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (1.20.3)\n",
      "Collecting responses<0.19\n",
      "  Downloading responses-0.18.0-py3-none-any.whl (38 kB)\n",
      "Requirement already satisfied: pandas in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (1.3.2)\n",
      "Requirement already satisfied: aiohttp in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (3.7.4.post0)\n",
      "Requirement already satisfied: tqdm>=4.62.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (4.62.2)\n",
      "Collecting fsspec[http]>=2021.05.0\n",
      "  Downloading fsspec-2022.2.0-py3-none-any.whl (134 kB)\n",
      "\u001b[K     |████████████████████████████████| 134 kB 28.7 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting huggingface-hub<1.0.0,>=0.1.0\n",
      "  Downloading huggingface_hub-0.4.0-py3-none-any.whl (67 kB)\n",
      "\u001b[K     |████████████████████████████████| 67 kB 4.4 MB/s  eta 0:00:01\n",
      "\u001b[?25hCollecting dill\n",
      "  Downloading dill-0.3.4-py2.py3-none-any.whl (86 kB)\n",
      "\u001b[K     |████████████████████████████████| 86 kB 4.1 MB/s  eta 0:00:01\n",
      "\u001b[?25hCollecting multiprocess\n",
      "  Downloading multiprocess-0.70.12.2-py38-none-any.whl (128 kB)\n",
      "\u001b[K     |████████████████████████████████| 128 kB 34.0 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting filelock\n",
      "  Downloading filelock-3.6.0-py3-none-any.whl (10.0 kB)\n",
      "Collecting sacremoses\n",
      "  Downloading sacremoses-0.0.47-py2.py3-none-any.whl (895 kB)\n",
      "\u001b[K     |████████████████████████████████| 895 kB 43.4 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: pyyaml>=5.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (5.4.1)\n",
      "Collecting regex!=2019.12.17\n",
      "  Downloading regex-2022.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (764 kB)\n",
      "\u001b[K     |████████████████████████████████| 764 kB 47.0 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting tokenizers!=0.11.3,>=0.11.1\n",
      "  Downloading tokenizers-0.11.6-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.5 MB)\n",
      "\u001b[K     |████████████████████████████████| 6.5 MB 38.5 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting absl-py\n",
      "  Downloading absl_py-1.0.0-py3-none-any.whl (126 kB)\n",
      "\u001b[K     |████████████████████████████████| 126 kB 34.5 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: six>=1.14.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from rouge-score) (1.16.0)\n",
      "Requirement already satisfied: joblib in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nltk) (1.0.1)\n",
      "Requirement already satisfied: click in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nltk) (8.0.1)\n",
      "Requirement already satisfied: typing_extensions in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from torch) (3.10.0.2)\n",
      "Requirement already satisfied: ipykernel>=4.5.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (6.2.0)\n",
      "Requirement already satisfied: widgetsnbextension~=3.5.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (3.5.1)\n",
      "Requirement already satisfied: ipython-genutils~=0.2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (0.2.0)\n",
      "Requirement already satisfied: nbformat>=4.2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (5.1.3)\n",
      "Requirement already satisfied: jupyterlab-widgets>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (1.0.0)\n",
      "Requirement already satisfied: traitlets>=4.3.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (5.1.0)\n",
      "Requirement already satisfied: ipython>=4.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (7.27.0)\n",
      "Requirement already satisfied: debugpy<2.0,>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (1.4.1)\n",
      "Requirement already satisfied: jupyter-client<8.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.1.12)\n",
      "Requirement already satisfied: matplotlib-inline<0.2.0,>=0.1.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (0.1.2)\n",
      "Requirement already satisfied: tornado<7.0,>=4.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.1)\n",
      "Requirement already satisfied: setuptools>=18.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (58.0.4)\n",
      "Requirement already satisfied: backcall in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.2.0)\n",
      "Requirement already satisfied: decorator in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (5.0.9)\n",
      "Requirement already satisfied: pickleshare in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.7.5)\n",
      "Requirement already satisfied: pexpect>4.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (4.8.0)\n",
      "Requirement already satisfied: jedi>=0.16 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.18.0)\n",
      "Requirement already satisfied: pygments in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (2.10.0)\n",
      "Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (3.0.17)\n",
      "Requirement already satisfied: parso<0.9.0,>=0.8.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jedi>=0.16->ipython>=4.0.0->ipywidgets) (0.8.2)\n",
      "Requirement already satisfied: pyzmq>=13 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jupyter-client<8.0->ipykernel>=4.5.1->ipywidgets) (22.2.1)\n",
      "Requirement already satisfied: jupyter-core>=4.6.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jupyter-client<8.0->ipykernel>=4.5.1->ipywidgets) (4.7.1)\n",
      "Requirement already satisfied: python-dateutil>=2.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jupyter-client<8.0->ipykernel>=4.5.1->ipywidgets) (2.8.2)\n",
      "Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbformat>=4.2.0->ipywidgets) (3.2.0)\n",
      "Requirement already satisfied: pyrsistent>=0.14.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (0.17.3)\n",
      "Requirement already satisfied: attrs>=17.4.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets) (21.2.0)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from packaging->datasets) (2.4.7)\n",
      "Requirement already satisfied: ptyprocess>=0.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from pexpect>4.3->ipython>=4.0.0->ipywidgets) (0.7.0)\n",
      "Requirement already satisfied: wcwidth in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=4.0.0->ipywidgets) (0.2.5)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (3.2)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (1.26.6)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (2021.5.30)\n",
      "Requirement already satisfied: notebook>=4.4.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from widgetsnbextension~=3.5.0->ipywidgets) (6.4.3)\n",
      "Requirement already satisfied: argon2-cffi in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (20.1.0)\n",
      "Requirement already satisfied: prometheus-client in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.11.0)\n",
      "Requirement already satisfied: Send2Trash>=1.5.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.8.0)\n",
      "Requirement already satisfied: nbconvert in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (5.6.1)\n",
      "Requirement already satisfied: terminado>=0.8.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.9.4)\n",
      "Requirement already satisfied: jinja2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (3.0.1)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (5.1.0)\n",
      "Requirement already satisfied: async-timeout<4.0,>=3.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (3.0.1)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (1.5.1)\n",
      "Requirement already satisfied: chardet<5.0,>=2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (4.0.0)\n",
      "Requirement already satisfied: cffi>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.14.0)\n",
      "Requirement already satisfied: pycparser in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from cffi>=1.0.0->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.20)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jinja2->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.0.1)\n",
      "Requirement already satisfied: entrypoints>=0.2.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.3)\n",
      "Requirement already satisfied: pandocfilters>=1.4.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.4.3)\n",
      "Requirement already satisfied: defusedxml in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.7.1)\n",
      "Requirement already satisfied: testpath in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.5.0)\n",
      "Requirement already satisfied: bleach in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (4.0.0)\n",
      "Requirement already satisfied: mistune<2,>=0.8.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.8.4)\n",
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
       "model_id": "c3f03b10be2a43dba104957838137b1c",
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
      "Fetched 3316 kB in 1s (2931 kB/s)[0mm\u001b[33m\u001b[33m\n",
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
    "model_checkpoint = \"google/bigbird-pegasus-large-bigpatent\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4RRkXuteIrIh"
   },
   "source": [
    "This notebook is built to run  with any model checkpoint from the [Model Hub](https://huggingface.co/models) as long as that model has a sequence-to-sequence version in the Transformers library. Here we picked the [`google/bigbird-pegasus-large-bigpatent`](https://huggingface.co/google/bigbird-pegasus-large-bigpatent) checkpoint. \n"
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
       "model_id": "8f1fd949b1394cd8af51974c16f12b7e",
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
       "model_id": "abbd0f2bca044fa68351e128f530b91d",
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
       "model_id": "e17c8d2db3d1494db377ec732088af08",
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
       "model_id": "24dcc6cbd1e54f1c95f0bf43da404796",
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
       "model_id": "c2cdb47796574d739c7d4356f5241b95",
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
       "model_id": "30e559b48f8042ff95a7a59676aa9acf",
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
       "      <td>fecundity is reduced in male patients with congenital adrenal hyperplasia ( cah ) due to 21-hydroxylase deficiency . recent studies suggested that development of testicular adrenal rest tumors ( tarts ) , which cause an obstruction of the seminiferous tubules , may play a major role [ 1 , 2 ] . in addition , suppression of the gonadal axis due to adrenal androgen excess might also cause reduced fertility . both pathomechanisms are thought to be a consequence of insufficient hormonal control . besides these somatic causes of impaired fertility in cah males , there might be aspects of psychosocial adaption and sexual well - being which may be additional factors for impaired fertility . however , up to now there are no studies investigating sexual well - being in male cah patients . sexual function is best measured by patient self - report avoiding interviewer bias and only patients can report on issues such as sexual interest and the extent to which sexual dysfunction has an adverse effect on their quality of life . the brief sexual function inventory ( bsfi ) provides an excellent tool to assess a self - reported measure of current sexual functioning . the aims of our two - year prospective study in adult male patients with congenital adrenal hyperplasia wereto investigate changes in hypothalamic - pituitary - testicular regulation by gnrh testing , to evaluate changes in sexual functioning and quality of life . to investigate changes in hypothalamic - pituitary - testicular regulation by gnrh testing , to evaluate changes in sexual functioning and quality of life . the subjects were adult male patients with confirmed classical cah due to 21-hydroxylase deficiency with regular hormonal follow - up at the outpatient clinic of the department of endocrinology of the charit campus mitte hospital , berlin . the study was approved by the ethics committee of the charit campus mitte berlin ( permit no . exclusion criteria were other diseases with impairment of gonadal capacity and other general and psychiatric diseases . all patients were seen at the outpatient clinic by two experienced endocrinologists ( m.q . ; m.v . ) on a regular basis every six months . in each patient physical examination , blood drawings ( between 0800 and 1000  h , 2  h after morning medication ) , questionnaires , and testicular ultrasounds were performed at study start ( baseline ) and two years later ( follow - up ) . the treating physicians tried to optimize treatment during the study period according to the endocrine practice guidelines . the standard medications for the treatment of 21-ohd deficiency are hydrocortisone ( hc ) , prednisolone ( pr ) , and dexamethasone ( dx ) [ 6 , 7 ] . since these glucocorticoids have different biological strengths , dosage for pr and dx were converted into hydrocortisone equivalent ( pr was converted 1 to 5 to hc , dx 1 to 70 to hc ) [ 8 , 9 ] . after conversion to the hydrocortisone equivalent dose , the daily total amount of hydrocortisone equivalent in milligrams was calculated as well as the total daily dose per body surface area ( mg / m ) . grey - scale and color doppler ultrasonography of the testes was obtained in longitudinal and transverse sections . circulating concentrations of 4-androstenedione ( ad ) ( beckmanncoulter , krefeld , germany ) , testosterone , dheas ( dpc biermann gmbh ; bad nauheim , germany ) , lh , fsh , renin concentrations , acth , and 17-hydroxy - progesterone ( 17-ohp ) ( mp biomedicals gmbh ; eschwege , germany ) were measured by commercially available assays . the gnrh stimulation test was performed by administering 100  g gnrh ( aventis pharma gmbh , frankfurt , germany ) as an i.v.bolus . serum fsh and lh levels were measured at 0 and 30  min after gnrh dose . we used the differences between peak and basal lh and fsh concentrations , referred to as max , as response variables to eliminate the additive effect of basal lh or fsh level on the peak . a normal lh response in gnrh testing was assumed if lh levels rise was at least 3-fold ; normal fsh response if fsh rises was at least 50% or above 3  psychometric evaluation of patients was performed using three validated self - assessment subjective health status ( shs ) questionnaires : the sf-36 , the brief form of the giessen complaint list ( gbb-24 ) , and the  hospital anxiety and depression scale  ( hads ) . in addition , the sexual functioning was assessed by the male brief sexual function inventory ( bsfi ) . all four questionnaires were presented as self - explanatory , multiple - choice self - assessments . the sf-36 questionnaire is the most widely used generic instrument to assess quality of life ( qol ) . it consists of eight multi - item domains representing physical functioning ( pf ) , role functioning physical ( rp ) , bodily pain ( bp ) , general health perception ( gh ) , vitality ( vt ) , social functioning ( sf ) , role functioning emotional ( re ) , and mental health ( mh ) . the domain scores range from 0 to 100 with higher values indicating better qol [ 12 , 13 ] . each item is scored as a number , with a maximum score of 21 for each subscale . higher scores indicate higher levels of anxiety or depression . a cut - off value of 8 is regarded as indicating mild impairment , and a cut - off value of 11 is indicative of severe impairment . the short form of the gbb-24 questionnaire consists of 24 items defining four subscales ( exhaustion tendency , gastric symptoms , pain in the limbs , and heart complaints ) , each including six items with ratings from 0 to 4 . in addition , a global score of discomfort ( gsd ) is calculated by adding the four subscale scores . the maximum value for each subscale is 24 , and for the global score 96 . higher scores indicate greater impairment of well - being . regarding control group data we calculated the z - scores by using reference data for sf-36 scores obtained from the german national health survey ( bundesgesundheits - survey 1998 , robert koch institut , berlin 2000 , public use file bgs 98 ) comprising a representative random sample of 7124 subjects from the german population aged between 18 and 79  yr . reference data for the hads ( n = 4410 ) and the gbb-24 ( n = 2076 ) were obtained from previously performed surveys [ 1517 ] . the brief sexual function inventory ( bsfi ) was used to assess perceived problems associated with sexual drive ( two items ) , erection ( three items ) , problem assessment ( three items ) , ejaculation ( two items ) , or overall satisfaction ( one item ) . each question was scored on a 5-point scale , ranging from 0 to 4 , with lower scores indicating worse sexual function . regarding control group data we calculated the z - scores by using normative data for the bsfi obtained from a representative random sample of 1185 subjects from the norwegian population aged between 20 and 79  yr . results are expressed as mean  standard deviation ( sd ) if not stated otherwise . the significance of data was determined by students t test in normally distributed and in not normally distributed data by mann - whitney - wilcoxon test where appropriate . eight patients were excluded due to testicular operations or other exclusion criteria . finally 20 patients were enrolled into the study . three patients did not participate in the 2-year follow - up visit and were not included into the 2-year follow - up analysis . clinical and genetic characteristics are shown in table 1 : 14 patients had salt - wasting cah and 6 patients had simple - virilizing cah . patients with salt - wasting cah were diagnosed within the first week after birth ; patients with simple - virilizing form were diagnosed in the first 7 years after birth . biochemical and hormonal parameters of the 17 patients at baseline and at the 2-year follow - up visit are presented in table 2 . over the study period bmi , systolic blood pressure , lipids , androgens , and androgen precursors did not change significantly in the whole cohort or in the sw and sv subgroups . of these , two were fathers ( 12.5% ) , one of one child and the other of two children . during the study period three adrenal crises occurred resulting in a calculated incidence of 8.8 adrenal crises per 100 patients / year , which is higher than the recently reported frequency in cah patients ( 4.8 crises per 100 patients / year )   and resembles more the frequency in patients with primary adrenal insufficiency ( 6.6 crises per 100 patients / year ) . decreased dheas levels were measured in 15 patients ( 88.2% ) at baseline and in all patients at follow - up ( 100% ) . nmol / l ) was observed in no patients at baseline ( 0% ) and in only one patient ( 5.9% ) at follow - up , whereas elevated levels of 17-ohp in serum ( &gt; 36  nmol / l ) were present in 4 patients ( 23.5% ) at baseline and 2 patients at follow - up ( 11.8% ) . the androstenedione to testosterone ( ad / t ) ratio as indicator of testicular testosterone production was normal ( &lt; 0.2 ) in 11 patients ( 64.7% ) at baseline and follow - up ; three patients ( 17.6% ) had an ad / t ratio &gt; 1 suggesting testosterone from predominantly adrenal origin ( table 2 ) . estradiol levels were within the normal male range ruling out any suppression of the hypothalamus - pituitary - gonadal axis by estradiol . total testosterone levels were decreased in 4 patients ( 23.5% ) at baseline and in 2 patients ( 11.8% ) at follow - up ; calculated free testosterone index was diminished in 5 patients ( 29.4% ) at baseline and in 8 patients ( 47.1% ) at follow - up . iu / l ) in all patients at baseline and in all but one patient at follow - up . basal fsh levels were elevated in three patients at baseline ( 17.6% ) and normal in all patients at follow - up . gnrh stimulation induced an adequate increase in lh in all but one patient at baseline ( 5.9% ) and in all but two patients ( 11.8% ) at follow - up . fsh failed to increase sufficiently by gnrh stimulation in two patients at baseline and at follow - up ( 11.8% ) . three patients ( 18% ) showed tart in testicular ultrasound with a size of 611  mm . in one patient tart regressed and was not detected after 2 years . in a subset of patients ( 6 of the 17 patients ) patients with an ad / t ratio below 0.2 , indicating sufficient adrenal suppression and a testosterone of testicular origin , showed significant lower 17-ohp and ad levels than patients with an ad / t ratio &gt; 0.2 ( table 3 ) . significantly more patients with an ad / t ratio &lt; 0.2 received dexamethasone . basal lh and fsh levels as well as testosterone levels were not different between the groups . however , the max increase in lh in gnrh testing was significantly higher in the patients with an ad / t ratio &lt; 0.2 than those with an ad / t ratio &gt; 0.2 ( table 3 ) . analysis of the qol questionnaires ( gbb-24 , hads , and sf-36 ) revealed no significant changes in z - scores during the 2-year study period in our adult male cah patient cohort ( figure 1 ) . however , all dimensions of the gbb-24 showed a trend to increased z - scores indicating an impairment of qol ( figure 1(a ) ) . similar results were found for the anxiety and depression z - scores of the hads questionnaires ( figure 1(b ) ) . z - scores of the sf-36 questionnaire showed a trend to impairments especially in the dimensions  physical functioning ,   general health perception ,  and  emotional role functioning  ( figure 1(c ) ) . the dimensions role  physical functioning  and the analysis of the participants ' z - scores revealed that male cah patients exhibited a slightly reduced sexual drive . no significant differences in bsfi z - scores were found between patients with ad / t ratio &lt; 0.2 or &gt; 0.2 . further analysis showed that ad levels significantly negatively correlated with z - scores of the dimension  sexual drive  ( p &lt; 0.05 ; figure 3 ) with higher ad levels associated with lower z - scores ( = impaired  sexual drive  ) . decreases in qol and sexual well - being were not correlated with the presence of tarts . development of tarts and suppression of the gonadal axis are possible factors that might cause reduced fertility in male cah patients [ 1 , 2 , 21 ] . it is suggested that adrenal - derived androgen excess due to insufficient hormonal control might be the underlying cause [ 2 , 3 ] . a recent study in adult male cah patients revealed a high prevalence of impaired leydig cell function and impaired spermatogenesis . however , the authors found no correlation between semen parameters , hormonal control , and tart prevalence or size . in our current study , the majority of patients showed basal testosterone and lh within the normal range of young healthy men   suggesting normal leydig cell function in most of the patients . after two years no significant differences were observed in our patients indicating stable therapeutic regimens . however , lh and fsh showed a more pronounced increase ( max ) after gnrh stimulation than reported in healthy normal males . we further subdivided our cohort into a group with good hormonal control and only testicular testosterone production indicated by an androstenedione / testosterone ( ad / t ) ratio &lt; 0.2 and a group with poorer disease control and mixed adrenal and testicular testosterone production indicated by an ad / t ratio &gt; 0.2 . the group with an ad / t ratio &gt; 0.2 showed a normal lh and fsh response ( max ) to gnrh compared to healthy young men . however , the group with an ad / t ratio &lt; 0.2 presented a significant higher lh response to gnrh testing . this resembled a prepubertal response in gnrh testing but might be also due to a suppressed hypothalamic - pituitary axis with a decreased release of gnrh from the hypothalamus . this might be caused by abundant adrenal androgens , which seems not to be the case in this group with ad / t ratio &lt; 0.2 . interestingly , the percentage of dexamethasone treated patients was significantly higher in the group with an ad / t ratio &lt; 0.2 compared to the group with a ratio &gt; 0.2 , but we did not find a significant difference in total daily glucocorticoid equivalent dose per body surface between the two groups . in addition , the amount of glucocorticoid used was approximately similar to that used in other recent studies with male cah patients [ 1 , 23 ] . we assume that total glucocorticoid doses were not too high because our patients showed still normal and not suppressed lh levels . in summary , this suggests that dexamethasone has a profound effect on the hypothalamic - pituitary feedback regulation . this is in accordance with previous reports that changing glucocorticoid medication from hydrocortisone to dexamethasone resulted in an increased fertility . besides these somatic causes of impaired fertility in cah males , there might be aspects of psychosocial adaption and sexual well - being which might be additional factors for impaired fertility . we performed this in a prospective fashion and used also quality of life questionnaires to detect possible other changes or general influences during the 2-year study period . during the study period our cah patients showed unchanged bmi , unchanged metabolic and hormonal parameters , and unchanged impaired z - scores in qol questionnaires . impaired qol in male cah patients has been shown in previous studies [ 7 , 2527 ] ; however , these had only cross - sectional and not a longitudinal design . we did not find differences in qol z - scores in patients that were on dexamethasone and prednisolone treatment compared to hydrocortisone only as previously reported . sexual drive ,   erections ,  and  ejaculations  were impaired in our cohort . it is important to point out that  overall satisfaction  should not be confused with the mean score of the functional domains of the bsfi , and additional factors might be involved not covered by the questions . it is known that patients with low scores on functional domains , for example , ejaculatory impairment as a side - effect of an anti - depressant drug , do not necessarily report reduced overall sexual satisfaction . we are the first to describe a clearly impaired sexual well - being in male cah patients by using an established sexual function questionnaire . interestingly , we observed that poor disease control , according to elevated androstenedione levels , was associated with a reduced  sexual drive . therefore , we believe that aspects of psychosocial adaption and sexual well - being might be important additional factors for impaired fertility in our male cah patients first , cah patients have had a chronic disease since their childhood , as well as having been exposed to exogenous glucocorticoids also during pubertal development . secondly , male cah patients have still a lower height than the average male population   and this might cause problems in psychosocial adaptation . however , a recent hungarian study showed that sexual activity was not clearly related to other anthropometric parameters such as height . ( 1 ) there is no normative data for germany for calculating z - scores for the bsfi , and we had to rely on normative data from norway . however , no significant differences in functional bsfi scores were found between the norwegian data and american data from the olmsted county . ( 2 ) there is increasingly reduced sexual function concerning drive , erection , ejaculation , and problem assessment with age with most of these age - related effects starting at &gt; 50 years old . however , our patients were all below the age of 50  y. ( 3 ) our study is a rather small cohort of male cah patients ; however , this is the first longitudinal study in adult male cah patients . in conclusion , we showed that male cah patients with a normal ad / t ratio showed an increased lh and fsh response in gnrh testing indicating possible decreased hypothalamic gnrh release by glucocorticoid therapy . secondly , we found that male cah patients had impaired sexual well - being , especially regarding erections , ejaculations , and sexual drive .</td>\n",
       "      <td>&lt;S&gt; \\n introduction . men with congenital adrenal hyperplasia ( cah ) due to 21-hydroxylase deficiency show impaired fecundity due to testicular adrenal rest tumors and/or suppression of the gonadal axis . &lt;/S&gt; &lt;S&gt; sexual well - being might be an additional factor ; however , no data exists . &lt;/S&gt; &lt;S&gt; patients and methods . &lt;/S&gt; &lt;S&gt; prospective longitudinal monocentric study included 20 male cah patients ( 14 salt wasting , 6 simple virilizing ; age 1849  yr ) . clinical assessment , testicular ultrasound , biochemical and hormonal parameters , three validated self - assessment questionnaires ( sf-36 , gbb-24 , and hads ) , and male brief sexual function inventory ( bsfi ) were analyzed at baseline and after two years &lt;/S&gt; &lt;S&gt; . results . &lt;/S&gt; &lt;S&gt; basal lh and testosterone levels suggested normal testicular function . &lt;/S&gt; &lt;S&gt; lh and fsh responses to gnrh were more pronounced in patients with a good therapy control according to androstenedione / testosterone ratio &lt; 0.2 . &lt;/S&gt; &lt;S&gt; this group had significant higher percentage of patients on dexamethasone medication . &lt;/S&gt; &lt;S&gt; gbb-24 , hads , and sf-36 showed impaired z - scores and no changes at follow - up . &lt;/S&gt; &lt;S&gt; bsfi revealed impairments in dimensions  sexual drive ,   erections ,  and  ejaculations ,  whereas  problem assessment  and  overall satisfaction  revealed normal z - scores . &lt;/S&gt; &lt;S&gt; androstenedione levels correlated ( p = 0.036 ) inversely with z - scores for  sexual drive  with higher levels associated with impaired &lt;/S&gt; &lt;S&gt;  sexual drive . &lt;/S&gt; &lt;S&gt;  conclusion . &lt;/S&gt; &lt;S&gt; male cah patients showed a partly impaired sexual well - being which might be an additional factor for reduced fecundity . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>eating disorders were described in the early descriptions of patients with asperger syndrome.1 asperger syndrome is a serious and chronic neurodevelopmental disorder , which is presently defined by social deficits , restricted interests , and relative preservation of language and cognitive ability.2 in diagnostic and statistical manual of mental disorders ( dsm)-iv , the syndrome was considered to be separate , but it fell under the broader category of pervasive developmental disorders . in dsm - v , asd is a disorder with persistent deficits in social interaction and communication skills , accompanied by restricted , repetitive patterns of behavior , interests , or activities and by atypical sensory reactivity.3 we know nowadays that eating disorders take various forms and are often presented in asd , complicating both diagnosis and therapy . rastam4 offered a summary of these disorders stating that abnormal eating behaviors are overrepresented in asd , including food refusal , pica , rumination , and selective eating . she considers the connection between anorexia nervosa ( an ) and asd as interesting and states that asperger syndrome is sometimes not recognized in female teenagers with eating disorders . there is a risk that autistic traits in girls with an are overlooked , which may lead to a simplification in the diagnostic consideration and therapeutic procedures . a contemporary review publication has explained that asds are overrepresented in individuals who develop an and also that asds are common in chronic cases of an.5 this comorbidity has been associated with a poorer prognosis.6 the research by baron - cohen et al7 confirms that girls with an have elevated autistic traits . the authors point out that clinicians should consider whether a focus on autistic traits might be helpful in the assessment and treatment of anorexia . early - onset an ( in children under the age of 12 years ) represents approximately 5% of all cases . it is a serious disorder jeopardizing the development of children in the somatic and psychosocial areas , and it seems that its incidence is growing.8 in individual research groups where early - onset is indicated , there is not always agreement as to the precise age definition of  early - onset  ; some authors describe premenarcheal girls , others refer to the age between 8 and 14 years.9 both an and early - onset eating disorders include syndromes of food avoidance emotional disorder and selective eating . specific psychopathology of early - onset an is very similar to the disorder onset in adolescence.10 an extensive study by halmi et al11 suggests that the predominant feature that precedes all an subtypes is global childhood rigidity , which is a trait that leads to resistance to changes . pooni et al12 found a higher incidence of autistic traits in individuals with early - onset eating disorders ( 816 years ) compared with typically developing peers , namely , repetitive and stereotyped behaviors , and also trends toward higher levels of autistic social impairment . coombs et al13 looked into the relationship between eating disorder psychopathology and autistic symptomatology in a non - clinical sample of school children aged 1114 years with no recorded psychiatric diagnoses , and found a significant relationship between the level of eating disorder symptomatology and asd symptomatology . according to karlsson et al14 eating disorders are common in asd but are often being overlooked . they developed a psychometrically and statistically valid swedish eating assessment for autism spectrum disorders questionnaire detecting eating disorders , which has been designed for individuals with asd aged 1525 years with normal intelligence ; in younger patients , clinical assessment has to suffice at the moment . the question of relationship , similarity , and connection between asd and an has diagnostic and therapeutic importance in clinical practice . apart from the diagnosis of clinical an syndrome , it is necessary to assess the development of cognitive and psychosocial traits of a child , including the possibility of identifying asd or prominent autistic traits . equally , in asd patients it is important to consider the occurrence of eating disorders associated with cognitive and psychosocial peculiarities . the therapeutic basis must respect these development peculiarities and make them part of the therapeutic program . the results imply that young people with an would benefit from a treatment approach tailored to the needs of individuals on the autism spectrum.15 kerbeshian and burd16 have shown an inspiring approach on the case history of a 12-year - old girl with high - functional autism and partial an . they have demonstrated that the treatment approaches used with individuals with neuropsychiatric developmental disorders might be effective in higher functioning individuals with eating disorders . therapeutic implications emphasize the need to improve cognitive and social functions,17 deficits in the field of mentalization,18 and the importance of focusing on working with the family.19 a girl aged 10 years and 9 months was admitted to a children s psychiatric clinic with an eating disorder and an underlying diagnosis of asperger syndrome . the patient s parents had degrees from technical universities and were healthy . the patient s sister , a grammar school student , was 2 years older and had asperger syndrome . she began to form individual words at 8 months and sentences from 24 months , but did not speak much until the age of 4 years . she began to speak fluently at the age of 4 years . at the age of 4.5 years she managed to fit into a small group of children in kindergarten . in the third year of school , she joined a partly new group in a language class and got a new class teacher . she began to dislike going to school ; she had mood swings , sometimes there were suicidal proclamations ; and she withdrew from her peers . at the end of the school year , she did not manage a school trip lasting several days , she ended up disorientated , and she ran away from the teachers several times . she said that she had been confused because of the change in the daily routine that she was used to at school . the diagnosis of asperger syndrome was considered for the first time at this point , yet the adhd diagnosis was erroneously established , and the girl was medicated with atomoxetine , 40 mg per day . when on the medication , she lost appetite and began to reduce the food intake . she had never been a great eater , but until the age of 9 years the parents never noticed any eating problems . she ate small portions of food at precisely the same times daily . in the year leading up to the children s psychiatric clinic admission , she grew 12 cm and her weight reduced by 5 kg . on admission to the children s psychiatric clinic , most of the time , the patient used a pseudo - adult language in conversation and spoke in a high - pitched voice with unnatural intonation . she said on several occasions that she wanted to be the skinniest girl in the world . at other times , she expressed her wish to be a model , fashion designer , and world - famous painter . the eating regimen at the ward was first accompanied by high tension , even affective seizures ; she refused food , and behaved in a bizarre way ( concealed food in clothes , escaped , cried loudly , proclaimed suicide ) . after several days of adapting to the regimen , she began to accept food , and the diet . the asperger syndrome diagnosis was confirmed using the autism diagnostic observation schedule ( ados)20 testing method and following an interpretation of the psychological examination . the girl had a prominently impaired perception of her own body , little interest in social contact , egocentric perception , and infantile expression . she would escape in an imaginary world in which she was a famous and respected artist . her introspection was minimal ; affective seizures grew in number with her growing weight . during the hospitalization , her weight increased by 8 kg following a plan , and she kept her eating regimen even during visits home . the parents requested consultation because of some peculiarities in their daughter s behavior and habits , emphasizing on eating problems . for about 6 months , she expressed her opinion that being fat meant being ugly and mean , and sometimes made tactless remarks about people around her . she insisted on specific odd arrangements when eating ; she had to sit at her own place , have her own dishes , and minimized the food and drink she took . she said repeatedly that she did not want to be fat and old , and she kept asking her parents strange questions on this topic again and again . in the patient s history , there was suspicion of asperger syndrome . according to the parents , she did not have any capability of empathy , never asked personal questions , considered mainly her own self , and was a  great egoist  . in social contacts , she was aloof and passive in relation to peers and adults ; she only critically commented on what was happening around her without becoming much involved . there were great problems in adapting to changes ( changes in routes , clothes , daily routine ) accompanied by negative reactions or even affective seizures with verbal and brachial aggression . her playing had elements of stereotypes ; there were finger mannerisms . in conversation , she showed signs of impaired communication and social interaction . using the ados testing method , concerning her eating disorder ( minimizing food intake , rigid eating habits , specific arrangements when eating ) , the parents were recommended to approach this as a symptom of the asd diagnosis . the occurrence of an in asperger syndrome has been described in literature.5 good understanding of the connection between these two disorders is crucial both for diagnosis and treatment . the therapeutic points of departure must respect the individual composition of symptoms and their mutual links . the picture of an is always critically influenced by the presence of a pervasive developmental disorder . the clinical guidance of patients with such comorbidities is always more demanding and requires experience with both diagnoses . the patients case histories illustrate the issues and make it possible to share the clinical experience . important issues are raised in the understanding of the comorbidity of these disorders and the implications for treatment . in 1985 , gillberg21 was the first to describe cases where a relationship between children s autism and an was established ( four cases of autistic boys whose close relatives suffered from an ) . the first clear clinical case report was submitted by rothery and garden22 who described the case of a 16-year - old girl with an who was previously diagnosed as having infantile autism . this case illustrates that an does occur in adolescents with autism and that it is important it is diagnosed , so that appropriate treatment can be given . similarly , fisman et al23 described the development of an in a high - functioning autistic adolescent 13-year - old girl . autism was diagnosed when the patient was 4 years old , and a change in eating habits started approximately a year prior to the admission . hospitalization was suggested because of continued weight loss accompanied by increasing refusal to eat and the failure to disengage parents from the patient s eating and weight preoccupation . this case study illustrates , besides a clear comorbidity of both diagnoses , an example where a combined psychotherapeutic and pharmacological strategy resulted in good improvement . kerbeshian and burd16 presented a case report of multiple comorbidities in a 12-year - old girl with high - functioning autistic disorder who developed tourette syndrome , obsessive  compulsive disorder , and an . our cases document the difficulty in diagnosing and treating patients with concurrent eating and pervasive developmental disorders . in our first case history of a nearly 11-year - old girl , there was a clear comorbidity of early - onset an and asperger syndrome ; the diagnostic criteria for both the disorders were fulfilled . compared with patients of similar age , the signs of an ( such as the disorder of the body schema , minimizing food intake , lack of introspection ) were more persistent and difficult to influence with the therapeutic procedures commonly applied in cases of eating disorders . the persistence and rigidity so typical of asperger syndrome hindered the work with the patient both in the therapeutic regimen and in the individual , group , and family therapy . the girl needed longer than the usual to adapt to the therapeutic regimen and to accept it , then she rigidly insisted on keeping it , both for herself and for her fellow patients with eating disorders . the success of the therapy depended on the selection of a suitable motivating approach to the patient taking into consideration her traits stemming from the underlying asperger syndrome diagnosis . in our second case history , the eating problem was rather more part of the core symptoms of asperger syndrome , even though the eating disorder was the original reason for the psychiatric assessment . the manipulation around food could be interpreted as a communication means in a child with impaired social and communication abilities . only if this therapy fails would we recommend a more targeted approach to the eating disorder . it is important that we notice these anorectic traits in patients , especially at a young age , which is not typical for the incidence of an . it follows from literature and our clinical experience that the comorbidity of eating disorders and asd is not unusual . our statement is based on the detailed description of two clinical cases of girls with asperger syndrome and symptoms of an . it is necessary to distinguish which symptoms are part of the underlying diagnosis and which are distinctive comorbid symptoms . both diagnosis and therapy should be performed by experts experienced in working with patients with both the diagnoses . we believe that the most efficient in infancy and adolescence is the combined therapeutic strategy , which involves a structured behavioral approach as well as psychotherapy , pharmacotherapy , and family therapy .</td>\n",
       "      <td>&lt;S&gt; eating disorders frequently occur in conjunction with autism spectrum disorders , posing diagnostic and therapeutic difficulties . &lt;/S&gt; &lt;S&gt; the comorbidity of anorexia nervosa and asperger syndrome is a significant clinical complication and has been associated with a poorer prognosis . &lt;/S&gt; &lt;S&gt; the authors are presenting the cases of an eleven - year - old girl and a five - and - a - half - year - old girl with comorbid eating disorders and asperger syndrome . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>co2mnsi samples were prepared and investigated completely in situ in an ultrahigh vacuum cluster consisting of sputtering chambers , an molecular beam epitaxy ( mbe ) chamber , and a srups chamber equipped with a he gas discharge lamp ( h=21.2  ev ) and a hemispherical energy analyzer with multi - channel spin filter27 ( energy resolution 400  mev , sherman function s=0.420.05 ( ref . first , an epitaxial buffer layer of the heusler compound co2mnga ( 30  nm ) was grown on the mgo(100 ) substrate by radio frequency ( rf)-sputtering at room temperature . by an optimized additional annealing process at 550  c l21 order is obtained as shown by high energy electron diffraction ( rheed ) and x - ray diffraction ( xrd ) . . induced by the buffer layer the co2mnsi thin films show already some degree of l21 surface order as deposited . by additional annealing the order is improved as demonstrated for the film surface by rheed ( fig . a low annealing temperature of ta=300c results in a significantly increased intensity of the characteristic rheed l21 superstructure peaks . however , by xrd no ( 111 ) peak , which is indicative for l21 order , is observed for ta&lt;400  c . this suggests that l21 order is present at the film surface , but not in the bulk of the thin film . for ta400  c the ( 111 ) peak appears in xrd . for ta500  c some ga from the buffer layer is observed by core - level haxpes to have diffused to the co2mnsi surface . the magnetic moments of all samples amount to 5  b per formula unit at 4  k and is reduced by 3% at room temperature , in agreement with theoretical predictions and experimental values measured on bulk samples29 . figure 2 shows in situ ups spectra of co2mnsi thin films annealed at different temperatures ta without spin analysis . the large acceptance angle of the spectrometer ( 10 ) and applied sample bias voltage of 10  v result in k|| values which cover the complete brillouin zone . the spectra of all samples are almost identical , only the broad hump at eef=2,900  mev vanishes and the peak at eef=1,150  mev is slightly broadened for the deposited and the ta=550  c sample . however , by spin analysis clear differences between the samples are revealed . figure 3 shows the spin polarization of mgo / co2mnga(30  nm)/co2mnsi(70  nm ) thin films annealed at different temperatures ta as measured by srups . a huge room temperature spin polarization of 9093% at the fermi energy at room temperature was obtained for samples annealed between 300  c and 450  c . in combination with the ups calculations discussed below , these exceptionally high values are the first direct observation of half - metallicity in the surface region of any heusler compound , which provide strong evidence for 100% spin polarization in the bulk of the thin films . with lower annealing temperatures the spin polarization is reduced at ef and slightly increased at higher binding energies , which can be explained by an energy broadening of the electronic states owing to reduced structural order . with higher annealing temperatures ups is a surface sensitive method and thus the results can not be directly associated with electronic bulk band structure properties . however , as will be shown below , band structure based calculations of photoemission spectra provide this link . as additional experimental input for such calculations , a comparison of spin - integrated ex situ haxpes with a photon energy of 6  kev of alox capped ( oxidation protection ) co2mnsi thin films and spin - integrated in situ ups ( uncapped films ) was carried out . owing to the increased information depth of haxpes , true surface states are typically not observed by this method . as shown in fig . 4 , the in situ spin - integrated ups and the haxpes results fundamentally agree although the information depth of both experiments varies from 2  nm to 20  nm . this provides evidence that true surface states like shockley or tamm states , which are mainly located at the first atomic layer30 , do not contribute to the ups data . we calculated the spin resolved bulk dos of co2mnsi using the spin polarized relativistic korringa  kohn  rostoker ( spr - kkr ) green function method implemented in the munich spr - kkr band structure programme package employing the perdew  ernzerhof functional32 shifts the upper edge to higher energies , but leaves the lower edge almost unchanged . for a comparison of our experimental data with ups- and haxpes calculations this electronic structure provides the basis for a one - step model of photoemission , which includes all matrix - element effects , multiple scattering in the initial and final states33 , and all surface - related effects in the excitation process . we used a recently developed relativistic generalization for excitation energies ranging from about 10  ev to more than 10  kev ( ref . 34 ) realized in the full spin - density matrix formulation for the photocurrent35 . in fig . 4 the calculations and the experimental spin - integrated ups and haxpes results are compared . nearly quantitative agreement for both , uv and hard x - ray photon energies , is obtained . particularly with regard to the small dos just below the fermi energy the agreement of the calculations with the high ups and haxpes intensities in this energy range is remarkable and is traced back to a bulk - like surface resonance as will be discussed below . the obtained agreement between the spin - integrated ups / haxpes experiments and calculations based on a half metallic bulk band structure represents already evidence for half - metallicity . additional strong evidence is provided by the analysis of the srups data . for the surface region we can estimate the position of the lower band edge of the minority gap directly from the experimental data by taking the maximum of the derivative of the minority spin intensity with respect to the energy , which is found at eef 500  mev . from previous surface sensitive x - ray magnetic circular dichroism experiments we estimated the position of the upper band edge to be at eef+400  mev ( ref . 5 the highest experimentally obtained spin polarization is shown together with the spin polarization derived directly from the calculated dos , the calculated photoemission asymmetry including all broadening effects considering bulk contributions only , and the calculated photoemission asymmetry including surface - related effects . the correspondence between the dos and calculated pure bulk - like ups spectrum becomes clear , if the influence of intrinsic life time broadening owing to electronic correlations and included experimental energy resolution ( e=400  mev ) is considered . it is obvious that these broadening effects within the bulk calculations reduce the expected ups spin polarization although the dos is half - metallic . however true surface states contribute to the layer - resolved photocurrent with an intensity distribution that is nonzero for the first atomic layer only . consequently , their contribution to the total spectral weight decreases with increasing number of layers generating the photocurrent . thus in general with increasing photon energies the combined effect of energy - dependent cross - sections and larger inelastic mean free path results in a reduced weight of surface state photoemission . however , the situation is very different for co2mnsi , where we identified in our calculations a resonance on the ( 001)-surface , which is embedded in the bulk continuum with a strong coupling to the majority bulk states . in our case this surface resonance extends over the first six atomic layers , which is similar to the case of w(110 ) , where we found a surface resonance revealing a considerable bulk contribution35 as well . the spectral weight of this surface resonance is much larger than that of a true surface state resulting in a significant contribution to the total intensity even at hard x - ray energies . co2mnsi samples were prepared and investigated completely in situ in an ultrahigh vacuum cluster consisting of sputtering chambers , an molecular beam epitaxy ( mbe ) chamber , and a srups chamber equipped with a he gas discharge lamp ( h=21.2  ev ) and a hemispherical energy analyzer with multi - channel spin filter27 ( energy resolution 400  mev , sherman function s=0.420.05 ( ref . first , an epitaxial buffer layer of the heusler compound co2mnga ( 30  nm ) was grown on the mgo(100 ) substrate by radio frequency ( rf)-sputtering at room temperature . by an optimized additional annealing process at 550  c l21 order is obtained as shown by high energy electron diffraction ( rheed ) and x - ray diffraction ( xrd ) . . induced by the buffer layer the co2mnsi thin films show already some degree of l21 surface order as deposited . by additional annealing the order is improved as demonstrated for the film surface by rheed ( fig . a low annealing temperature of ta=300c results in a significantly increased intensity of the characteristic rheed l21 superstructure peaks . however , by xrd no ( 111 ) peak , which is indicative for l21 order , is observed for ta&lt;400  c . this suggests that l21 order is present at the film surface , but not in the bulk of the thin film . for ta400  c the ( 111 ) peak appears in xrd . for ta500  c some ga from the buffer layer is observed by core - level haxpes to have diffused to the co2mnsi surface . the magnetic moments of all samples amount to 5  b per formula unit at 4  k and is reduced by 3% at room temperature , in agreement with theoretical predictions and experimental values measured on bulk samples29 . figure 2 shows in situ ups spectra of co2mnsi thin films annealed at different temperatures ta without spin analysis . the large acceptance angle of the spectrometer ( 10 ) and applied sample bias voltage of 10  v result in k|| values which cover the complete brillouin zone . the spectra of all samples are almost identical , only the broad hump at eef=2,900  mev vanishes and the peak at eef=1,150  mev is slightly broadened for the deposited and the ta=550  c sample . however , by spin analysis clear differences between the samples are revealed . figure 3 shows the spin polarization of mgo / co2mnga(30  nm)/co2mnsi(70  nm ) thin films annealed at different temperatures ta as measured by srups . a huge room temperature spin polarization of 9093% at the fermi energy at room temperature was obtained for samples annealed between 300  c and 450  c . in combination with the ups calculations discussed below , these exceptionally high values are the first direct observation of half - metallicity in the surface region of any heusler compound , which provide strong evidence for 100% spin polarization in the bulk of the thin films . with lower annealing temperatures the spin polarization is reduced at ef and slightly increased at higher binding energies , which can be explained by an energy broadening of the electronic states owing to reduced structural order . with higher annealing temperatures ups is a surface sensitive method and thus the results can not be directly associated with electronic bulk band structure properties . however , as will be shown below , band structure based calculations of photoemission spectra provide this link . as additional experimental input for such calculations , a comparison of spin - integrated ex situ haxpes with a photon energy of 6  kev of alox capped ( oxidation protection ) co2mnsi thin films and spin - integrated in situ ups ( uncapped films ) was carried out . owing to the increased information depth of haxpes , true surface states are typically not observed by this method . as shown in fig . 4 , the in situ spin - integrated ups and the haxpes results fundamentally agree although the information depth of both experiments varies from 2  nm to 20  nm . this provides evidence that true surface states like shockley or tamm states , which are mainly located at the first atomic layer30 , do not contribute to the ups data . we calculated the spin resolved bulk dos of co2mnsi using the spin polarized relativistic korringa  kohn  rostoker ( spr - kkr ) green function method implemented in the munich spr - kkr band structure programme package employing the perdew  ernzerhof functional32 shifts the upper edge to higher energies , but leaves the lower edge almost unchanged . for a comparison of our experimental data with ups- and haxpes calculations this electronic structure provides the basis for a one - step model of photoemission , which includes all matrix - element effects , multiple scattering in the initial and final states33 , and all surface - related effects in the excitation process . we used a recently developed relativistic generalization for excitation energies ranging from about 10  ev to more than 10  kev ( ref . 34 ) realized in the full spin - density matrix formulation for the photocurrent35 . in fig . 4 the calculations and the experimental spin - integrated ups and haxpes results are compared . nearly quantitative agreement for both , uv and hard x - ray photon energies , is obtained . particularly with regard to the small dos just below the fermi energy the agreement of the calculations with the high ups and haxpes intensities in this energy range is remarkable and is traced back to a bulk - like surface resonance as will be discussed below . the obtained agreement between the spin - integrated ups / haxpes experiments and calculations based on a half metallic bulk band structure represents already evidence for half - metallicity . additional strong evidence is provided by the analysis of the srups data . for the surface region we can estimate the position of the lower band edge of the minority gap directly from the experimental data by taking the maximum of the derivative of the minority spin intensity with respect to the energy , which is found at eef 500  mev . from previous surface sensitive x - ray magnetic circular dichroism experiments we estimated the position of the upper band edge to be at eef+400  mev ( ref . 5 the highest experimentally obtained spin polarization is shown together with the spin polarization derived directly from the calculated dos , the calculated photoemission asymmetry including all broadening effects considering bulk contributions only , and the calculated photoemission asymmetry including surface - related effects . the correspondence between the dos and calculated pure bulk - like ups spectrum becomes clear , if the influence of intrinsic life time broadening owing to electronic correlations and included experimental energy resolution ( e=400  mev ) is considered . it is obvious that these broadening effects within the bulk calculations reduce the expected ups spin polarization although the dos is half - metallic . true surface states contribute to the layer - resolved photocurrent with an intensity distribution that is nonzero for the first atomic layer only . consequently , their contribution to the total spectral weight decreases with increasing number of layers generating the photocurrent . thus in general with increasing photon energies the combined effect of energy - dependent cross - sections and larger inelastic mean free path results in a reduced weight of surface state photoemission . however , the situation is very different for co2mnsi , where we identified in our calculations a resonance on the ( 001)-surface , which is embedded in the bulk continuum with a strong coupling to the majority bulk states . in our case this surface resonance extends over the first six atomic layers , which is similar to the case of w(110 ) , where we found a surface resonance revealing a considerable bulk contribution35 as well . the spectral weight of this surface resonance is much larger than that of a true surface state resulting in a significant contribution to the total intensity even at hard x - ray energies . as shown in fig . 5 , the inclusion of the complete surface - related photoexcitation in the ups calculation results in perfect agreement with the experiment . if the surface resonance were not present , half - metallic behaviour would persist but the finite experimental resolution in photoemission would hinder the observation of a high spin polarization . because the surface resonance is strongly coupled to the band structure of the bulk , this provides evidence for the validity of our calculated half metallic bulk band structure of co2mnsi . and , from the spintronics applications point of view it is the room temperature spin polarization in the thin film surface region , which is relevant . in conclusion , investigating optimized thin films of the compound co2mnsi by in situ srups , we were able to demonstrate for the first time half - metallicity in combination with directly measured ( ) % spin polarization at room temperature in the surface region of a heusler thin film . novel band structure and photoemission calculations including all surface - related effects show that the observation of a high spin polarization in a wide energy range below the fermi energy is related to a stable surface resonance in the majority band of co2mnsi extending deep into the bulk of the material . our results show that careful thin film preparation can indeed result in a high spin polarization with a sufficient degree of stability in a surface region of several atomic layers . in particular it shows that the observed tunnelling magnetoresistance values are not limited by the intrinsic spin polarization of the heusler alloy and that potentially much larger values can be obtained by carefully optimized growth . fundamentally our observation paves the way for most powerful future spintronic devices on the basis of heusler materials . m.j . initiated and coordinated the project and wrote the paper . a.k . and m.j</td>\n",
       "      <td>&lt;S&gt; ferromagnetic thin films of heusler compounds are highly relevant for spintronic applications owing to their predicted half - metallicity , that is , 100% spin polarization at the fermi energy . however , experimental evidence for this property is scarce . &lt;/S&gt; &lt;S&gt; here we investigate epitaxial thin films of the compound co2mnsi in situ by ultraviolet - photoemission spectroscopy , taking advantage of a novel multi - channel spin filter . by this surface sensitive method , &lt;/S&gt; &lt;S&gt; an exceptionally large spin polarization of ( ) % at room temperature is observed directly . as a more bulk sensitive method , additional ex situ &lt;/S&gt; &lt;S&gt; spin - integrated high energy x - ray photoemission spectroscopy experiments are performed . &lt;/S&gt; &lt;S&gt; all experimental results are compared with advanced band structure and photoemission calculations which include surface effects . &lt;/S&gt; &lt;S&gt; excellent agreement is obtained with calculations , which show a highly spin polarized bulk - like surface resonance ingrained in a half metallic bulk band structure . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>attachment is a relatively stable emotional bond which is created between child and mother or those with whom an infant regularly interacts . parents responses to the signs of child 's attachment behavior and their availability in stressful situations , provides a safe place and condition for children , based on which , children organize their expectations from the environment . the attachment between child and primary caregiver ( usually mother ) would become internalized and later act as a mental model which is used by the adult person to use as a base for building friendship and romantic relationships ; it can affect the attitudes of people in their adulthood as well . adult attachment styles are subdivided into three categories : ( 1 ) secure : secure people are intimate and comfortable in making relationships , and they are sure that others would like them . ( 2 ) anxious - ambivalent : they have a strong desire for close relationships but also have many concerns of rejection . these people have a negative image of themselves , but a positive attitude toward others . ( 3 ) avoidance : for this group of people , self - reliance is the most valuable issue . hence , it can be said that attachment styles affect other aspects of one 's life and have an impact on persons relationships with other people after childhood . many researchers and authorities have shifted their focus toward the topics such as joy , happiness , life satisfaction , and positive emotions . according to many theories of emotions , one of the six great emotions is happiness ; the six great emotions include surprise , fear , anger , happiness , disgust , and worry . happiness is a type of conception about individual 's own life ; it includes items such as life satisfaction , positive emotions , and mood , lack of anxiety and depression and its different aspects of emotions . when people are satisfied with their living conditions and are frequently experiencing positive and less negative emotions , it is said that they are at high levels of mental health . increased levels of happiness is directly associated with the better status of health , appetite , sleep , memory , family relationships , friendships , family status , and ultimately mental health . the relationship between subjective well - being and emotion regulation with attachment styles in various studies has been explained . despite the important role of medical students in public health and the significance of their happiness which is related to their attachment styles , so far , this research was aimed to assess the relationship between attachment styles and happiness and demographic characteristics of medical students . this descriptive and analytical study was conducted on medical students in kurdistan university of medical sciences , in 2012 . as exclusion criteria , students who were unwilling to fill out a questionnaire and guest students since there were five independent variables in the study and it was needed to include 35 samples for each variable in the regression model , the calculated sample size was 175 people ; a total of 200 students were included in the study . samples were chosen through stratified sampling method ( different levels of education ) and each stratum was proportional to the size of each class . to collect the data , after obtaining permission from the ethics committee of kurdistan university of medical sciences , list of all medical students , which was classified by educational level , was obtained from education office . the samples were systematically selected from the list provided by education office ; they were selected in proportion to the number of students in each educational level ( physiopathology , extern , intern level ) . after taking their consent to participate in research and explaining the objectives , questionnaires were given to the participants . the questionnaires were filled out by the students and were collected the same day . before completing the questionnaire ( 47 questions ) , students were assured that all information will be confidential , and they were also asked to answer the questions accurately . they were allowed to ask their questions in case of facing any ambiguity in the questionnaire . this scale is developed by hazan and shaver ( 1987 ) and it has 15 items , with five items for each of the three types of secure attachment , ambivalent attachment , and avoidant attachment style . it is scored from never ( zero ) to almost always ( score = 4 ) . the score of each attachment subscale is obtained by calculating the mean of five items for each subscale . in various studies , the reliability of the questionnaire has been calculated from 0.78 to 0.81 ; moreover , its reliability in iranian culture was tested by boogar et al . , the obtained results for the entire test , the ambivalent , avoidant , and secure attachment styles were 0.75 , 0.83 , 0.81 , and 0.77 , respectively . to measure the happiness variable , the scale has 29 items which is scored on a range of zero to four ; it has five marks including life satisfaction with eight items , self - esteem with seven items , subjective well - being with five items , satisfaction with four items , and positive manner with three items . because two items have a correlation coefficients of &lt; 35% with any of the five other components , they are not included in any of the components , but they are included in the total score . the reliability of this scale among iranian students has been reported to be 0.93 . the collected data were entered into spss version 16 ( ibm , chicago il , usa ) . quantitative data were described using the mean and standard deviation ( sd ) , and string variables were described using frequency and percentage . the correlation between happiness score and attachment style scores were assessed using pearson 's correlation coefficient . the difference between the happiness score and the scores of different attachment styles in each sex were compared using independent tests . the scores for different educational levels were compared using one - way anova . finally , using multiple regressions ( enter method ) , happiness variable as the dependent variable and the score of different attachment styles , gender , educational level , and grade point average ( gpa ) as the independent variables , if applicable , were entered into the model . this scale is developed by hazan and shaver ( 1987 ) and it has 15 items , with five items for each of the three types of secure attachment , ambivalent attachment , and avoidant attachment style . it is scored from never ( zero ) to almost always ( score = 4 ) . the score of each attachment subscale is obtained by calculating the mean of five items for each subscale . in various studies , the reliability of the questionnaire has been calculated from 0.78 to 0.81 ; moreover , its reliability in iranian culture was tested by boogar et al . , the obtained results for the entire test , the ambivalent , avoidant , and secure attachment styles were 0.75 , 0.83 , 0.81 , and 0.77 , respectively . to measure the happiness variable , the scale has 29 items which is scored on a range of zero to four ; it has five marks including life satisfaction with eight items , self - esteem with seven items , subjective well - being with five items , satisfaction with four items , and positive manner with three items . because two items have a correlation coefficients of &lt; 35% with any of the five other components , they are not included in any of the components , but they are included in the total score . this scale is developed by hazan and shaver ( 1987 ) and it has 15 items , with five items for each of the three types of secure attachment , ambivalent attachment , and avoidant attachment style . it is scored from never ( zero ) to almost always ( score = 4 ) . the score of each attachment subscale is obtained by calculating the mean of five items for each subscale . in various studies , the reliability of the questionnaire has been calculated from 0.78 to 0.81 ; moreover , its reliability in iranian culture was tested by boogar et al . , the obtained results for the entire test , the ambivalent , avoidant , and secure attachment styles were 0.75 , 0.83 , 0.81 , and 0.77 , respectively . to measure the happiness variable , the revised oxford happiness inventory was used which had an overall reliability of 0.91 . the scale has 29 items which is scored on a range of zero to four ; it has five marks including life satisfaction with eight items , self - esteem with seven items , subjective well - being with five items , satisfaction with four items , and positive manner with three items . because two items have a correlation coefficients of &lt; 35% with any of the five other components , they are not included in any of the components , but they are included in the total score . the collected data were entered into spss version 16 ( ibm , chicago il , usa ) . quantitative data were described using the mean and standard deviation ( sd ) , and string variables were described using frequency and percentage . the correlation between happiness score and attachment style scores were assessed using pearson 's correlation coefficient . the difference between the happiness score and the scores of different attachment styles in each sex were compared using independent tests . the scores for different educational levels were compared using one - way anova . finally , using multiple regressions ( enter method ) , happiness variable as the dependent variable and the score of different attachment styles , gender , educational level , and grade point average ( gpa ) as the independent variables , if applicable , were entered into the model . the mean ( sd ) of participants age was 22.42 ( 2.45 ) years . of all , 122 students ( 61% ) were female and 185 persons ( 92.5% ) were single . a total of 89 students ( 44.5% ) were in basic sciences educational level and the majority of participants , i.e. , 97 students ( 48.5% ) had gpa of 1517 [ table 1 ] . the distribution of demographic variables in studied subjects overall , the mean ( sd ) score of happiness was 62.71 ( 17.61 ) , secure attachment style was 11.46 ( 2.56 ) , avoidant attachment style was 9.34 ( 3.32 ) , and ambivalent attachment style was 7.93 ( 3.47 ) . there was no significant relationship between gender and attachment styles , however , the happiness score was 67.2 ( 17.2 ) in men and 59.9 ( 17.36 ) in women , and the difference was statistically significant ( p = 0.005 ) . the avoidant attachment style was 9.48 ( 3.34 ) in singles and 7.6 ( 2.66 ) in married people , and the difference was also statistically significant ( p = 0.03 ) [ table 2 ] . the relationship between gender and marital status of the studied subjects with attachment styles and happiness scores there was no significant relationship between the happiness score and educational level . the score of secure attachment style in students with gpa of 1720 was about 9.91 ( 2.9 ) , which was lower compared to those with lower gpas ( p = 0.051 ) . no significant relationship was observed between happiness score and other attachment styles with students gpas . age was not significantly correlated with happiness scores ( p = 0.797 , r = 0.019 ) . in the multivariate analysis , the relationship between attachment styles and happiness scores were compared and the results showed that after controlling for important factors , the variables of secure attachment style ( p = 0.001 ) , male gender ( p = 0.004 ) , and gpa ( p = 0.047 ) were associated with higher happiness scores ( r = 0.180 ) [ table 3 ] . comparison of the relationship between happiness scores and attachment style and other variables using multiple regression analysis the most common attachment style among students was secure attachment style that was consistent with the results of other studies . secure attachment style leads to activation of a system which bowlby calls the discovery system . this system allows a person to explore his / her environment and experience its own ability to control the condition . secure attachment gradually creates a sense of mastery and ability to handle frustration , and finally , in the context of a secure attachment relationship , then the person is enabled to reflect his / her emotions and positive beliefs about personal values and effectiveness . positive perfectionism , self - esteem , personal control , greater happiness in relationships better emotional management , less stress , and greater job satisfaction are among the specifications of secure attachment style ; these features may be a positive prognostic factor in medical students who usually endure much stress . in our study , the minimum frequency was observed in ambivalent attachment style ; our finding was similar to other studies . in asgharinejad et al.s study as well as ahadi et al.s study avoidant attachment style was the most common and secure attachment was second common style . due to differences in statistical samples and scales , which have been used in these two studies , these differences can be justified . attachment theory is focused on cognitive schema ; the schema affects the organization of individual 's relations with others and his / her perceptions of the world around . attachments formed in childhood can affect adulthood and the attachment between child and primary caregiver ( usually mother ) is internalized and serves as a mental model . according to the mentioned explanations , we can conclude that attachment styles are formed based on schemas and inner experiences , experiences which obtained through interaction with parents and others over time , the role of these factors is much stronger than the effect of gender alone . according to our results , there was a significant relationship between avoidant attachment style and marital status , and avoidant attachment style was more common among single people than married ; so , avoidant attachment could be a barrier to marriage . finney and noler believe that adults with avoidant attachment style have the same characteristics as those with dismissive attachment style ( self - positive model , others negative , with a low anxiety , and high avoidance ) . people with avoidant attachment styles have a negative attitude toward others and have difficulty in communicating with others and maintaining relationships ; they have a high sense of self - esteem and put low values on close relationships with others , which confirms our findings . the results showed no significant relationship between attachment style and gpa of individuals ; however , secure attachment style was less common in participants with high gpa . individuals with a secure attachment style are better able to interact with the environment , so they are expected to have better educational status , but the results of our study did not confirm this idea . it might be that struggling to get a higher score , sometimes help individual to compensate for a sense of frustration and low self - control . it is also possible that the educational system would create an unhealthy competitive environment and promote negative behaviors such as blind imitation without critical thinking . on the other hand , in our study , it was not determined to which educational level and age range each gpa belongs . in addition , the effects of other factors were not considered , and they have not even been considered in other studies as well , and this is one of the limitations of our study . in sheikhmoonesi et al.s study the average score of subjects in the happiness inventory was 41.23 and the average score of happiness in students of tehran university of medical sciences in 2010 was 47.13 . based on these results , our students had higher levels of happiness which could be due to facilities , the status of their field of study and university , their future career perspectives , and their inner attitudes . on the other hand , the statistical sample size , the age range , and demographic conditions can justify these differences . in our study , secure attachment style was associated with higher happiness scores and this finding was consistent with the findings of other studies . people with secure attachment style are successful in making relationships with others and have positive attitudes about self and others ; the mentioned items are effective in creating higher levels of happiness . researches also show that people with insecure attachment styles are more affected with emotional and psychological challenges and with increasing the feeling of helplessness in the marital relationship , they will be at lower levels of happiness . in a study , girls with secure attachment style , compared with girls with avoidant attachment style , were more satisfied with relationships with their fathers . as another results of our study , there was a significant relationship between happiness scores and gender ; accordingly , the happiness scores in boys was higher than that in girls . in keshavarz et al.s study , contrary to the results of our study , there was a positive relationship between female sex and happiness that could be due to differences in the studied populations . we studied students , while in keshavarz et al.s study , yazd population ( males and females ) were studied . study , no significant relationship was observed between sex and happiness . however , in solymani 's et al . study , men achieved higher scores in subscales of life satisfaction and self - esteem while men had higher scores in a positive manner and inner satisfaction . to interpret these differences , it can be said that working and educational condition , society 's attitudes toward gender , which is strongly influenced by cultural factors , can affect a person 's happiness . in our study , there was a negative correlation between age and happiness scores ; however , this relationship was not significant . in sheikhmoonesi et al.s study the happiness scores in people aged below 22 years were higher than that in people aged more than 22 years . to justify the consistency between the two studies , we can note the similarities in the field of study and age range . in keshavarz et al.s study , older age was associated with greater happiness which could be due to differences in population and age range . in boogar et al.s study , job satisfaction among younger nurses was higher than that in older people . in our studied population , individuals at different ages are not facing the same stressors and expectations ; indeed , the course materials , environmental conditions , and people whom they are communicating with ( professors , personals working in different wards , and patients ) are different at any stage . life satisfaction is not an objective and stable trait , rather it is sensitive to situational changes and is shaped based on individual 's perceptions and perspectives . in multiple regression analysis which was performed with the control of key factors , variables of secure attachment style , gender , and gpa were associated with higher happiness scores . such an analysis has not been carried out in other studies and is one of the strengths of our study . the higher gpa was associated with higher happiness scores and other studies have not addressed this issue . there was higher level of dissatisfaction and expectation among people with lower gpas ; on the other hand , students with higher gpas are dealing with more stress of keeping current situation and they have more competition with others . moreover , mediocre gpa did not indicate higher dissatisfaction , and it might even signify less competitive pressure and family expectations ; this greatly originates from individual 's attitudes and expectations . perfection - seeking individuals may excessively get higher scores , but they are less satisfied and happy . according to our results , the satisfaction score was not significantly associated with educational level which was consistent with the results of sheikhmonesi et al.s study . every educational level brings up different external conditions and stressors which may have different effects depending on the internal characteristics , student 's ability to cope with environment , and individual 's expectation , behavior , and social interaction with others . based on the findings of this study , the most common attachment style was secure attachment style , which could be a positive prognostic factor in medical students , helping them to manage stress . the frequency of avoidant attachment style among single persons was higher than that in married people , which is mainly due to their negative attitude toward others and failure to establish and maintain relationships with others . the variables of secure attachment style , male gender , and average gpa were associated with higher happiness scores these factors can be taken into account while planning for promoting happiness levels in students .</td>\n",
       "      <td>&lt;S&gt; background : attachment theory is one of the most important achievements of contemporary psychology . role of medical students in the community health is important , so we need to know about the situation of happiness and attachment style in these students.objectives:this study was aimed to assess the relationship between medical students attachment styles and demographic characteristics.materials and methods : this cross - sectional study was conducted on randomly selected students of medical sciences in kurdistan university , in 2012 . to collect data , hazan and shaver 's attachment style measure and the oxford happiness questionnaire were used . &lt;/S&gt; &lt;S&gt; the results were analyzed using the spss software version 16 ( ibm , chicago il , usa ) and statistical analysis was performed via t - test , chi - square test , and multiple regression tests.results:secure attachment style was the most common attachment style and the least common was ambivalent attachment style . &lt;/S&gt; &lt;S&gt; avoidant attachment style was more common among single persons than married people ( p = 0.03 ) . &lt;/S&gt; &lt;S&gt; no significant relationship was observed between attachment style and gender and grade point average of the studied people . &lt;/S&gt; &lt;S&gt; the mean happiness score of students was 62.71 . &lt;/S&gt; &lt;S&gt; in multivariate analysis , the variables of secure attachment style ( p = 0.001 ) , male gender ( p = 0.005 ) , and scholar achievement ( p = 0.047 ) were associated with higher happiness score.conclusion:the most common attachment style was secure attachment style , which can be a positive prognostic factor in medical students , helping them to manage stress . &lt;/S&gt; &lt;S&gt; higher frequency of avoidant attachment style among single persons , compared with married people , is mainly due to their negative attitude toward others and failure to establish and maintain relationships with others . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>thirty patients were diagnosed with iac in the peking union medical college hospital ( pumch ) from january 2011 to september 2014 , during which 275 patients were diagnosed with cca . all patients were retrospectively reviewed and information was collected including their sex , age , symptoms , weight loss ( decreased &gt; 5% within 6 mo ) , and serological tests , including biochemical tests , tumor markers , and the sigg4 level . imaging characteristics including endoscopic retrograde cholangiopancreatography ( ercp ) , magnetic resonance cholangiopancreatography ( mrcp ) , computed tomography ( ct ) , b - ultrasound , and endoscopic ultrasonography ( eus ) were also collected . chicago , il ) . the primary outcome consisted of the clinical parameters that showed significant differences in iac and cca . differences between the groups were evaluated using the independent samples t test , the  test , the mann - whitney u , or the fisher test according to their characteristic . in all tests , p values receiver operating characteristic curves were used to estimate the diagnostic application of sigg4 levels ( youden index = sensitivity+specificity1 ) . data were analyzed using spss version 13.0 ( spss inc . , chicago , il ) . the primary outcome consisted of the clinical parameters that showed significant differences in iac and cca . differences between the groups were evaluated using the independent samples t test , the  test , the mann - whitney u , or the fisher test according to their characteristic . in all tests , p values receiver operating characteristic curves were used to estimate the diagnostic application of sigg4 levels ( youden index = sensitivity+specificity1 ) . thirty patients ( 21 male and 9 female ; median age 59.012.7 y ; ranging from 28 to 83 y ) were diagnosed with iac , with the criteria described in the introduction section , and 275 cca patients ( 170 male and 105 female ; median age 61.811.3 y ; ranging from 30 to 89 y ) were diagnosed with histopathology and/or cytology . there was no significant difference in the gender and the age between the 2 groups ( table 1 ) . demographic data and symptoms of iac and cca patients as shown in table 1 , a significantly higher number of iac patients experienced weight loss than cca patients ( 66.7% in iac vs. 45.1% in cca , p=0.025 ) . moreover , iac patients had a significantly higher level of weight loss than cca patients ( 7.58.1 vs. 3.24.0 kg , p=0.008 ) . on comparing the prognosis of the 2 groups , iac patients had a significantly longer survival time than cca patients ( p&lt;0.001 ) . cca patients demonstrated significantly higher positive rates of tumor markers , including ca199 , ca242 , and cea , compared with iac patients . positive rates of ca199 , ca242 , and cea in cca patients compared with iac patients were 81.5% versus 42.9% , 45.5% versus 4.5% , and 29.2% versus 7.1% , respectively . in addition , average serological levels of these tumor markers in positive cca patients were significantly higher than those in positive iac patients ( p&lt;0.05 in all cases ) ( table 2 ) . there were no significant differences in the serum biochemistry tests including alt , ast , ggt , alp , tbil , and dbil between iac and cca patients ( table 3 ) . tumor maker detection in the iac and the cca groups serological measurement for the liver function of iac and cca patients thirty - one cca patients were tested for their sigg4 level , among whom 16.1% ( 5/31 ) were found to have an elevated level with a range between 29 and 8230 mg / l and an average of 896.3 mg / l . almost 100% of the iac patients showed elevated sigg4 ranging between 1650 and 78,590 mg / l , with an average of 16028.6 mg / l . when a cutoff level was set at 6 times the upper normal limit , the area under the curve for sigg4 was 0.981 in receiver operating characteristic analysis and sigg4 had 100% specificity for iac . on the basis of the youden index calculation , the best cutoff value for sigg4 in this cohort was 1575 mg / bile duct  occupying lesions were detected with ercp , mrcp , ct , b - ultrasound , or eus . an occupying lesion was defined as a thickening of the bile duct wall with a very clear margin . as shown in table 4 , the thickening wall ( p=0.001 ) and the occupying lesion ( p&lt;0.001 ) of the duct were found significantly different in iac and cca by eus . imaging comparison of iac and cca patients by different radiologic methods an example of an image taken with endoscopic ultrasonography that exhibited an occupying lesion of the bile duct . aip was the most frequent comorbidity of iac and the incidence reached as high as 83.3% in this study . the imaging diagnosis for aip included diffused pancreatic enlargement , irregular narrowing of the main pancreatic duct , and bile duct strictures . among the 30 iac patients , however , only 10.2% of the cca patients were found to have pancreas involvement and presented as tumor invasion . kidney ( 20% ) and parotid gland or lacrimal gland ( 53.3% ) involvement were also present in iac patients , whereas none was found in cca . both groups had hepatic hilar lymph nodule hyperplasia , but the percentage of incidents in iac patients was significantly higher ( 56.7% vs. 30.5% , p=0.004 ) . other organ involvement in iac and cca patients when diagnosed or highly suspected , iac patients were treated with steroid therapy ( initial prednisolone dose as 30 mg / d for 2 wk ) . the average sigg4 and tbil levels of iac patients decreased to 6278.37 mg / l and 26.14 mol / l , respectively . prednisolone application resulted in a decrease in the sigg4 level in all iac patients , and a decrease in the bilirubin level was noticed in 80.77% of the iac patients . thirty patients ( 21 male and 9 female ; median age 59.012.7 y ; ranging from 28 to 83 y ) were diagnosed with iac , with the criteria described in the introduction section , and 275 cca patients ( 170 male and 105 female ; median age 61.811.3 y ; ranging from 30 to 89 y ) were diagnosed with histopathology and/or cytology . there was no significant difference in the gender and the age between the 2 groups ( table 1 ) . demographic data and symptoms of iac and cca patients as shown in table 1 , a significantly higher number of iac patients experienced weight loss than cca patients ( 66.7% in iac vs. 45.1% in cca , p=0.025 ) . moreover , iac patients had a significantly higher level of weight loss than cca patients ( 7.58.1 vs. 3.24.0 kg , p=0.008 ) . on comparing the prognosis of the 2 groups , iac patients had a significantly longer survival time than cca patients ( p&lt;0.001 ) . cca patients demonstrated significantly higher positive rates of tumor markers , including ca199 , ca242 , and cea , compared with iac patients . positive rates of ca199 , ca242 , and cea in cca patients compared with iac patients were 81.5% versus 42.9% , 45.5% versus 4.5% , and 29.2% versus 7.1% , respectively . in addition , average serological levels of these tumor markers in positive cca patients were significantly higher than those in positive iac patients ( p&lt;0.05 in all cases ) ( table 2 ) . there were no significant differences in the serum biochemistry tests including alt , ast , ggt , alp , tbil , and dbil between iac and cca patients ( table 3 ) . tumor maker detection in the iac and the cca groups serological measurement for the liver function of iac and cca patients thirty - one cca patients were tested for their sigg4 level , among whom 16.1% ( 5/31 ) were found to have an elevated level with a range between 29 and 8230 mg / l and an average of 896.3 mg / l . almost 100% of the iac patients showed elevated sigg4 ranging between 1650 and 78,590 mg / l , with an average of 16028.6 mg / l . when a cutoff level was set at 6 times the upper normal limit , the area under the curve for sigg4 was 0.981 in receiver operating characteristic analysis and sigg4 had 100% specificity for iac . on the basis of the youden index calculation , the best cutoff value for sigg4 in this cohort was 1575 mg / bile duct  occupying lesions were detected with ercp , mrcp , ct , b - ultrasound , or eus . an occupying lesion was defined as a thickening of the bile duct wall with a very clear margin . as shown in table 4 , the thickening wall ( p=0.001 ) and the occupying lesion ( p&lt;0.001 ) of the duct were found significantly different in iac and cca by eus . imaging comparison of iac and cca patients by different radiologic methods an example of an image taken with endoscopic ultrasonography that exhibited an occupying lesion of the bile duct . aip was the most frequent comorbidity of iac and the incidence reached as high as 83.3% in this study . the imaging diagnosis for aip included diffused pancreatic enlargement , irregular narrowing of the main pancreatic duct , and bile duct strictures . among the 30 iac patients , however , only 10.2% of the cca patients were found to have pancreas involvement and presented as tumor invasion . kidney ( 20% ) and parotid gland or lacrimal gland ( 53.3% ) involvement were also present in iac patients , whereas none was found in cca . both groups had hepatic hilar lymph nodule hyperplasia , but the percentage of incidents in iac patients was significantly higher ( 56.7% vs. 30.5% , p=0.004 ) . when diagnosed or highly suspected , iac patients were treated with steroid therapy ( initial prednisolone dose as 30 mg / d for 2 wk ) . the average sigg4 and tbil levels of iac patients decreased to 6278.37 mg / l and 26.14 mol / l , respectively . prednisolone application resulted in a decrease in the sigg4 level in all iac patients , and a decrease in the bilirubin level was noticed in 80.77% of the iac patients . iac was recently recognized as an independent disease from other igg4-related diseases , and there are no epidemiology data for iac based on a large population.6 differential diagnosis between iac and cca can be challenging as both diseases share several symptoms and signs.7 obstructive jaundice accompanied with skin pruritus , abdominal discomfort , and/or weight loss have been the most common symptoms in both iac and cca patients.810 iac patients may be positive for tumor markers , whereas cca patient can also exhibit elevated sigg4 . imaging studies can also demonstrate many similarities including obstruction , dilatation , and a thickening wall of the bile duct . in this study , we examined the clinical data collected from patients who were diagnosed with either iac or caa . weight loss in iac patients was one of the symptoms that was significantly different from that of cca patients . we observed similar incidences of iac in both male and female patients , which agrees with other studies reported earlier.11 no significant demographic differences were found between iac and cca patients . the production of igg4 is related to the expression of several immune genetic factors , such as mhcii , polymorphism of nuclear factor-b , and fc - receptor - like ( fcrl ) 3.10 other scholars proposed the  induction and progression  biphasic mechanism,12 in which decreased naive tregs may induce a th1 immune response with the release of proinflammatory cytokines to antigens such as self - antigen or microorganisms . subsequently , th2-type immune responses may be involved in the disease progression , resulting in the production of igg4 . in the iac diagnosis criteria proposed by japanese scholars , the minimum level of igg4 was set as 1350 mg / l.2 however , the specificity at this cutoff is not sufficient to distinguish iac and cca . oseini et al13 found that out of the 126 cca patients , 17 ( 13.5% ) had elevated sigg4 ( &gt; 1400 mg / l ) and 4 ( 3.2% ) had a &gt; 2-fold ( &gt; 2800 mg / l ) increase . in our study , 16.1% of the cca patients had an elevated sigg4 level ( range , 29 to 8230 mg / l ) , which could mislead to an iac diagnosis , although the level was significantly lower than that of the iac group . on the basis of this study , we concluded that the best cutoff value for sigg4 level was 1575 our study suggested a cutoff level that was 6-fold higher than the upper normal limit of igg4 , which was different from the 4-fold criteria proposed previously by oseini et al.13 ca199 presents in the fetal gastrointestinal and pancreatic epithelium , whereas its serum level in adults is very low . its expression is elevated in adenocarcinoma cells , and is released into the blood through the thoracic duct . therefore , it can be a useful marker for the diagnosis of pancreatic carcinoma , gastric carcinoma , cca , and intrahepatic cca.14 the serum level of ca199 can also be elevated in pancreatitis , obstructive jaundice , and sclerosing diseases , which may be produced by abnormal epithelial cells.1517 in our study , the serum ca199 level was found to be increased in most of the cca patients . in iac patients , the serum ca199 level also increased , but at a significantly lower incidence and a significantly lower level . similarly , cea and ca242 were also found in the normal tissue.18 the incidence and elevated levels of cea and ca242 in cca patients were significantly higher than those in iac patients . these findings were in line with the studies published earlier.14,19,20 space - occupying lesions of the biliary tract are often diagnosed by b - ultrasound , ct , mri , or ercp . cca patients exhibit similar imaging characteristics , such as dilatation , thickening wall , or occupying lesion , which makes it difficult to distinguish one from the other . on reviewing the cases in our study , we found that these 3 manifestations exhibited significantly differently under eus between iac and cca , which could make eus a valuable tool to distinguish these 2 diseases . this finding was consistent with a previous study.21 multiple organ involvement , including most commonly the pancreas,19 the kidney , and the salivary and the lacrimal glands , is characteristic of iac.22,23 in this study , we had similar observations : 83.3% of the iac patients had obvious involvements of the pancreas , 20% had involvement of the kidney , and 53.3% had involvement of the salivary or the lacrimal glands . in contrast , in cca patients , the biliary tract was always the only involved organ at an early stage and few had other organs involved even though neighboring tissue / organ invasion and distant metastasis could occur at middle and advanced stages . . however , obtaining pathologic samples by puncture or ercp brush before surgery is invasive and may not be suitable for all patients , such as patients who are old , patients with coagulopathy , or those with high bilirubin.24 in addition , the positive rate of brush check is low.25 therefore , clinical examination and experimental treatment are very important for a differential diagnosis . we observed complete response of iac patients to the steroid treatment , although in some cases the stent placement also played a role in the symptom alleviation . the retrospective nature of this study made it difficult to obtain data of a single variable from all patients . the case number of the iac was low , which may affect the significance of the study . the use of a stent in some patients could be another factor in alleviating symptoms in iac patient , which was not taken into account for the response to steroid treatment because of the limited numbers . our study suggested that 6-fold higher levels of sigg4 , tumor markers ( ca199 , cea , and ca242 ) , and other organ involvement could be used as reference criteria for the differential diagnosis of iac and cca . for difficult cases , experimental steroid treatment can be used for further diagnosis under appropriate conditions .</td>\n",
       "      <td>&lt;S&gt; background and aim : immunoglobulin g4-associated cholangitis ( iac ) shares many similar symptoms with cholangiocarcinoma ( cca ) . however , the treatment and the prognosis are substantially different . this study aimed to identify the important markers for the differential diagnosis of these 2 diseases.methods:thirty iac patients and 275 cca patients &lt;/S&gt; &lt;S&gt; were reviewed retrospectively for their clinical symptoms , serological tests , and imaging characteristics . &lt;/S&gt; &lt;S&gt; posttreatment responses were also studied.results:igg4 had 100% specificity for iac at a cutoff of 6 times the upper normal limit . &lt;/S&gt; &lt;S&gt; iac patients had a significantly higher incidence of weight loss ( p=0.025 ) and a higher level of weight loss ( p=0.008 ) than cca patients . &lt;/S&gt; &lt;S&gt; the positive rates of biological markers ca199 , ca242 , and cea in cca and iac were 81.5% versus 42.9% , 45.5% versus 4.5% , and 29.2% versus 7.1% , respectively . &lt;/S&gt; &lt;S&gt; levels of these tumor markers in cca were significantly higher than in iac ( p&lt;0.05 ) . &lt;/S&gt; &lt;S&gt; the thickened wall [ 17/18 ( 94.4% ) vs. 3/10 ( 30% ) , p=0.001 ] and the occupying lesion on the bile duct [ 1/18 ( 5.6% ) vs. 8/10 ( 80% ) , p&lt;0.001 ] were found to be significantly different in iac and cca , respectively , by endoscopic ultrasonography . &lt;/S&gt; &lt;S&gt; autoimmune pancreatitis was the most frequently observed comorbidity of iac ( 25/30 ) . &lt;/S&gt; &lt;S&gt; all iac patients respond positively to steroid treatment.conclusions:increased tumor markers , 6-fold higher levels of serum igg4 , and other organs involvement could be the reference factors for a differential diagnosis of iac and cca . &lt;/S&gt; &lt;S&gt; endoscopic ultrasonography might be an effective imaging tool for diagnosis , although clinical signs and symptoms of iac and cca are similar . &lt;/S&gt; &lt;S&gt; experimental steroid treatment can be useful in the diagnosis for certain difficult cases . &lt;/S&gt;</td>\n",
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
       "model_id": "4625c76655aa4ac28d5d4afedacab7b1",
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
       "model_id": "9c3bc492cb244fe888dab482d7db1fe9",
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
       "model_id": "930e451d6f6f492aa8358ad2b3d1b95c",
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
       "model_id": "d54517a325b74fe8956f300748429101",
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
       "model_id": "51eef9fc0a51459b8d6c5f117d6af185",
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
    "The max input length of `google/bigbird-pegasus-large-bigpatent` is 4096, so `max_input_length = 4096`."
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
       "model_id": "eeab31a705724a65bd66f343d6639682",
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
       "model_id": "c23f0aaa98ed447ebf0b6f74f528445e",
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
       "model_id": "c569e36d32b54a8e92e2cc40e854b4c5",
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
       "model_id": "11142bc2fbb94163bc2601f2bf4d68cc",
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
    "    f\"{model_name}-finetuned-pubMed\",\n",
    "    evaluation_strategy = \"epoch\",\n",
    "    learning_rate=2e-5,\n",
    "    per_device_train_batch_size=batch_size,\n",
    "    per_device_eval_batch_size=batch_size,\n",
    "    weight_decay=0.01,\n",
    "    save_total_limit=3,\n",
    "    num_train_epochs=5,\n",
    "    predict_with_generate=True,\n",
    "    fp16=True,\n",
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
      "Cloning https://huggingface.co/Kevincp560/bigbird-pegasus-large-bigpatent-finetuned-pubMed into local empty directory.\n"
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
   "execution_count": 29,
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
      "  warnings.warn('Was asked to gather along dimension 0, but all '\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='2500' max='2500' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [2500/2500 2:12:38, Epoch 5/5]\n",
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
       "      <td>2.119800</td>\n",
       "      <td>1.628522</td>\n",
       "      <td>43.057900</td>\n",
       "      <td>18.179200</td>\n",
       "      <td>26.421000</td>\n",
       "      <td>39.076900</td>\n",
       "      <td>214.924000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>1.693900</td>\n",
       "      <td>1.569553</td>\n",
       "      <td>44.067900</td>\n",
       "      <td>18.933100</td>\n",
       "      <td>26.840000</td>\n",
       "      <td>40.068400</td>\n",
       "      <td>222.814000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>1.619500</td>\n",
       "      <td>1.550577</td>\n",
       "      <td>44.735200</td>\n",
       "      <td>19.353200</td>\n",
       "      <td>27.241800</td>\n",
       "      <td>40.745400</td>\n",
       "      <td>229.396000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1.579800</td>\n",
       "      <td>1.540313</td>\n",
       "      <td>45.041500</td>\n",
       "      <td>19.501900</td>\n",
       "      <td>27.296900</td>\n",
       "      <td>40.951000</td>\n",
       "      <td>231.044000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>1.559200</td>\n",
       "      <td>1.540273</td>\n",
       "      <td>45.085100</td>\n",
       "      <td>19.548800</td>\n",
       "      <td>27.391000</td>\n",
       "      <td>41.112000</td>\n",
       "      <td>231.608000</td>\n",
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
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/transformers/trainer.py:1443: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.\n",
      "  nn.utils.clip_grad_norm_(\n",
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
      "Saving model checkpoint to bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-500\n",
      "Configuration saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-500/config.json\n",
      "Model weights saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-500/pytorch_model.bin\n",
      "tokenizer config file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-500/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-500/special_tokens_map.json\n",
      "tokenizer config file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/special_tokens_map.json\n"
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
      "Saving model checkpoint to bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-1000\n",
      "Configuration saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-1000/config.json\n",
      "Model weights saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-1000/pytorch_model.bin\n",
      "tokenizer config file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-1000/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-1000/special_tokens_map.json\n"
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
      "tokenizer config file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/special_tokens_map.json\n"
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
      "Saving model checkpoint to bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-1500\n",
      "Configuration saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-1500/config.json\n",
      "Model weights saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-1500/pytorch_model.bin\n",
      "tokenizer config file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-1500/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-1500/special_tokens_map.json\n"
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
      "tokenizer config file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/special_tokens_map.json\n"
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
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/transformers/trainer.py:1443: FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered. At that point, setting error_if_nonfinite=false will be required to retain the old behavior.\n",
      "  nn.utils.clip_grad_norm_(\n",
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
      "Saving model checkpoint to bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-2000\n",
      "Configuration saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-2000/config.json\n",
      "Model weights saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-2000/pytorch_model.bin\n",
      "tokenizer config file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-2000/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-2000/special_tokens_map.json\n"
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
      "tokenizer config file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/special_tokens_map.json\n"
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
      "Deleting older checkpoint [bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-500] due to args.save_total_limit\n"
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
      "Saving model checkpoint to bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-2500\n",
      "Configuration saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-2500/config.json\n",
      "Model weights saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-2500/pytorch_model.bin\n",
      "tokenizer config file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-2500/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-2500/special_tokens_map.json\n"
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
      "tokenizer config file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/special_tokens_map.json\n"
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
      "Deleting older checkpoint [bigbird-pegasus-large-bigpatent-finetuned-pubMed/checkpoint-1000] due to args.save_total_limit\n"
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
       "TrainOutput(global_step=2500, training_loss=1.7144691162109376, metrics={'train_runtime': 7964.96, 'train_samples_per_second': 1.255, 'train_steps_per_second': 0.314, 'total_flos': 1.1196507790727578e+17, 'train_loss': 1.7144691162109376, 'epoch': 5.0})"
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
      "Saving model checkpoint to bigbird-pegasus-large-bigpatent-finetuned-pubMed\n",
      "Configuration saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/config.json\n",
      "Model weights saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/pytorch_model.bin\n",
      "tokenizer config file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/tokenizer_config.json\n",
      "Special tokens file saved in bigbird-pegasus-large-bigpatent-finetuned-pubMed/special_tokens_map.json\n"
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
      "To https://huggingface.co/Kevincp560/bigbird-pegasus-large-bigpatent-finetuned-pubMed\n",
      "   f456a9c..d236099  main -> main\n",
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
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [],
   "name": "bigbird-pegasus-patent-pubmed-summary-final.ipynb",
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
