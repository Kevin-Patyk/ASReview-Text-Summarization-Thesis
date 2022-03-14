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
      "\u001b[K     |████████████████████████████████| 312 kB 8.0 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting transformers\n",
      "  Downloading transformers-4.17.0-py3-none-any.whl (3.8 MB)\n",
      "\u001b[K     |████████████████████████████████| 3.8 MB 105.2 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting rouge-score\n",
      "  Downloading rouge_score-0.0.4-py2.py3-none-any.whl (22 kB)\n",
      "Collecting nltk\n",
      "  Downloading nltk-3.7-py3-none-any.whl (1.5 MB)\n",
      "\u001b[K     |████████████████████████████████| 1.5 MB 34.0 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: ipywidgets in ./miniconda3/envs/fastai/lib/python3.8/site-packages (7.6.4)\n",
      "Requirement already satisfied: numpy>=1.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (1.20.3)\n",
      "Collecting multiprocess\n",
      "  Downloading multiprocess-0.70.12.2-py38-none-any.whl (128 kB)\n",
      "\u001b[K     |████████████████████████████████| 128 kB 31.3 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: aiohttp in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (3.7.4.post0)\n",
      "Collecting fsspec[http]>=2021.05.0\n",
      "  Downloading fsspec-2022.2.0-py3-none-any.whl (134 kB)\n",
      "\u001b[K     |████████████████████████████████| 134 kB 32.9 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting dill\n",
      "  Downloading dill-0.3.4-py2.py3-none-any.whl (86 kB)\n",
      "\u001b[K     |████████████████████████████████| 86 kB 6.6 MB/s  eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: pandas in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (1.3.2)\n",
      "Collecting pyarrow!=4.0.0,>=3.0.0\n",
      "  Downloading pyarrow-7.0.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (26.7 MB)\n",
      "\u001b[K     |████████████████████████████████| 26.7 MB 22.0 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting huggingface-hub<1.0.0,>=0.1.0\n",
      "  Downloading huggingface_hub-0.4.0-py3-none-any.whl (67 kB)\n",
      "\u001b[K     |████████████████████████████████| 67 kB 4.3 MB/s  eta 0:00:01\n",
      "\u001b[?25hCollecting xxhash\n",
      "  Downloading xxhash-3.0.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (212 kB)\n",
      "\u001b[K     |████████████████████████████████| 212 kB 37.5 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: tqdm>=4.62.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (4.62.2)\n",
      "Collecting responses<0.19\n",
      "  Downloading responses-0.18.0-py3-none-any.whl (38 kB)\n",
      "Requirement already satisfied: packaging in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (21.0)\n",
      "Requirement already satisfied: requests>=2.19.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from datasets) (2.26.0)\n",
      "Collecting tokenizers!=0.11.3,>=0.11.1\n",
      "  Downloading tokenizers-0.11.6-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.5 MB)\n",
      "\u001b[K     |████████████████████████████████| 6.5 MB 35.3 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting sacremoses\n",
      "  Downloading sacremoses-0.0.47-py2.py3-none-any.whl (895 kB)\n",
      "\u001b[K     |████████████████████████████████| 895 kB 40.3 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting filelock\n",
      "  Downloading filelock-3.6.0-py3-none-any.whl (10.0 kB)\n",
      "Collecting regex!=2019.12.17\n",
      "  Downloading regex-2022.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (764 kB)\n",
      "\u001b[K     |████████████████████████████████| 764 kB 36.5 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: pyyaml>=5.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (5.4.1)\n",
      "Collecting absl-py\n",
      "  Downloading absl_py-1.0.0-py3-none-any.whl (126 kB)\n",
      "\u001b[K     |████████████████████████████████| 126 kB 36.0 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: six>=1.14.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from rouge-score) (1.16.0)\n",
      "Requirement already satisfied: click in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nltk) (8.0.1)\n",
      "Requirement already satisfied: joblib in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nltk) (1.0.1)\n",
      "Requirement already satisfied: ipython>=4.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (7.27.0)\n",
      "Requirement already satisfied: widgetsnbextension~=3.5.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (3.5.1)\n",
      "Requirement already satisfied: ipykernel>=4.5.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (6.2.0)\n",
      "Requirement already satisfied: ipython-genutils~=0.2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (0.2.0)\n",
      "Requirement already satisfied: jupyterlab-widgets>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (1.0.0)\n",
      "Requirement already satisfied: traitlets>=4.3.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (5.1.0)\n",
      "Requirement already satisfied: nbformat>=4.2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipywidgets) (5.1.3)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (3.10.0.2)\n",
      "Requirement already satisfied: matplotlib-inline<0.2.0,>=0.1.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (0.1.2)\n",
      "Requirement already satisfied: debugpy<2.0,>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (1.4.1)\n",
      "Requirement already satisfied: tornado<7.0,>=4.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.1)\n",
      "Requirement already satisfied: jupyter-client<8.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets) (6.1.12)\n",
      "Requirement already satisfied: pygments in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (2.10.0)\n",
      "Requirement already satisfied: pickleshare in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.7.5)\n",
      "Requirement already satisfied: pexpect>4.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (4.8.0)\n",
      "Requirement already satisfied: jedi>=0.16 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.18.0)\n",
      "Requirement already satisfied: decorator in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (5.0.9)\n",
      "Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (3.0.17)\n",
      "Requirement already satisfied: setuptools>=18.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (58.0.4)\n",
      "Requirement already satisfied: backcall in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from ipython>=4.0.0->ipywidgets) (0.2.0)\n",
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
      "Requirement already satisfied: certifi>=2017.4.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (2021.5.30)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (3.2)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests>=2.19.0->datasets) (1.26.6)\n",
      "Requirement already satisfied: notebook>=4.4.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from widgetsnbextension~=3.5.0->ipywidgets) (6.4.3)\n",
      "Requirement already satisfied: argon2-cffi in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (20.1.0)\n",
      "Requirement already satisfied: Send2Trash>=1.5.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.8.0)\n",
      "Requirement already satisfied: jinja2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (3.0.1)\n",
      "Requirement already satisfied: prometheus-client in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.11.0)\n",
      "Requirement already satisfied: terminado>=0.8.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.9.4)\n",
      "Requirement already satisfied: nbconvert in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (5.6.1)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (5.1.0)\n",
      "Requirement already satisfied: chardet<5.0,>=2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (4.0.0)\n",
      "Requirement already satisfied: async-timeout<4.0,>=3.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (3.0.1)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from aiohttp->datasets) (1.5.1)\n",
      "Requirement already satisfied: cffi>=1.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.14.0)\n",
      "Requirement already satisfied: pycparser in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from cffi>=1.0.0->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.20)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from jinja2->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (2.0.1)\n",
      "Requirement already satisfied: bleach in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (4.0.0)\n",
      "Requirement already satisfied: testpath in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.5.0)\n",
      "Requirement already satisfied: pandocfilters>=1.4.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (1.4.3)\n",
      "Requirement already satisfied: defusedxml in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.7.1)\n",
      "Requirement already satisfied: entrypoints>=0.2.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.3)\n",
      "Requirement already satisfied: mistune<2,>=0.8.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets) (0.8.4)\n",
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
   "execution_count": 4,
   "metadata": {
    "id": "9weGF83Sir6z"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5783fb89e7cc4ea6b1c32968c80c1432",
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
   "execution_count": 5,
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
      "Fetched 3316 kB in 1s (4378 kB/s)\u001b[0m\u001b[33m\n",
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
   "execution_count": 6,
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
   "execution_count": 7,
   "metadata": {
    "id": "voXBC93bir61"
   },
   "outputs": [],
   "source": [
    "model_checkpoint = \"google/pegasus-large\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4RRkXuteIrIh"
   },
   "source": [
    "This notebook is built to run  with any model checkpoint from the [Model Hub](https://huggingface.co/models) as long as that model has a sequence-to-sequence version in the Transformers library. Here we picked the [`google/pegasus-large`](https://huggingface.co/google/pegasus-large) checkpoint. "
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
   "execution_count": 8,
   "metadata": {
    "id": "IreSlFmlIrIm"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7eda0de4f1d44f4c81e5e41d995f1a71",
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
       "model_id": "6d6853e308c8483d8bf72c4b84250d27",
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
       "model_id": "4cf26748b4ad4bb1b41ff38e422f75ed",
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
       "model_id": "c057498c551e408182184a832c8a6a02",
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
       "model_id": "06cc4aa431a648489aa05055cba416f5",
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
       "model_id": "20ac8008d3914bb0b6a22a90c9621e3d",
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
   "execution_count": 9,
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
     "execution_count": 9,
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
   "execution_count": 10,
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
     "execution_count": 10,
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
   "execution_count": 11,
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
   "execution_count": 12,
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
   "execution_count": 13,
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
       "      <td>acute ophthalmia neonatorum ( on ) continues to play an important role in causing severe ocular problems of newborns in countries where primary health care coverage is insufficient , as in the case of angola . unfortunately , in most clinical obstetric wards of luanda , the nation 's capital , there is no regular plan of on prophylaxis , nor is there any capacity for determining the microorganisms involved in the etiology of this entity . therefore , most of the acute conjunctivitis cases in the newborns are empirically diagnosed as on and treated without having information on the causative agent . although prenatal screening and treatment of pregnant women are very effective for the prevention of on , this approach can be difficult to implement in developing countries . in some african countries , at the time of delivery , a large percentage of expectant mothers have had little or no prenatal care . in industrialized countries , on prophylaxis was performed by silver nitrate eyedrops and currently by topical erythromycin , tetracycline ointment , and others . in other developed countries , emphasis is placed on maternal surveillance , and there is no systematic prophylaxis of on . the use of povidone - iodine ( p - i ) has been advocated for the less developed countries , but few well - designed prospective , randomized , and controlled clinical studies have been performed with this agent . even while doubts remain regarding the real efficacy of p - i for prevention of on caused by either chlamydia trachomatis ( ct ) or neisseria gonorrhoeae ( ng ) [ 4 , 5 ] , this substance is well tolerated , has a broader spectrum than other anti - infectious agents , does not induce resistance , and is more effective than some antibiotics that are more expensive for prophylaxis [ 6 , 7 ] . after the civil war in angola ( 19742002 ) , , we conducted a pilot study structured as a maternity ward prospective series and found that 12% of the newborns had signs of acute bilateral conjunctivitis . an additional problem was the inability to determine the microorganisms involved [ 911 ] because microbiological studies were not routinely performed at either maternity wards or eye centers . based upon these existing conditions , we decided to evaluate different tests that could provide necessary microbiological information to the ophthalmologists and then select the best one for implementation in a nationwide on prophylaxis program . our initial attempt was to use the simplest available diagnostic tool , standard staining of conjunctival swabs by giemsa and gram stains . however , this provided little useful information regarding the active on agent(s ) . following this , we attempted to use multiplex polymerase chain reaction ( pcr ) of maternal endocervical and neonatal conjunctival specimens for microorganism identification . the pcr assay has been previously validated with conjunctival and endocervical samples collected from this study ( manuscript accepted ) . in brief , we demonstrated that this technique could be used to identify the three major microorganisms , for example , ct , ng , and mycoplasma genitalium ( mg ) , involved in sexually transmitted infections ( stis ) . this was successful not only in endocervical samples , but also in conjunctival smears , where it never had been used . therefore , our recommendation for angola 's health authorities was to provide this kind of microbiological identification technique . the primary purpose of this work was to analyze the efficacy , limitations , and obstacles of 2.5% p - i eyedrops for the prophylaxis of on in angola . these data are intended to provide the local health authorities with enough information to develop a national prophylactic campaign against on based on the routine use of p - i eyedrops for every newborn . a secondary objective was to provide additional data on the etiological agents both in conjunctival smears from newborns and in endocervical samples from their mothers and the possible relationship of these agents with clinical risk factors . an interventional , randomized , and prospective study with a blinded , randomized control group was designed . the study was conducted at the general augusto n'gangula specialized hospital ( hgeag ) and the health center of samba ( css ) , both in luanda , angola . approval was provided by the ethical commission of the faculty of medicine of the agostinho neto university ( luanda , angola ) . after the explanation of the objectives and methodology of the investigation the study was performed in accordance with the ethical standards from the 1964 declaration of helsinki and its later amendments . the target population for this study consisted of 317 mothers and their newborns from the hgeag ( 173 ) and the css ( 144 ) , recruited from 7 december 2011 to 22 november 2012 . the inclusion criteria consisted of healthy children weighing at least 2.3  kg and a gestation period of at least 37 weeks . newborns not meeting the inclusion criteria and those with respiratory distress at birth were excluded . additionally , because of the possible deleterious effect of iodine , newborns were excluded if their mothers were diagnosed with thyroid disease according to clinical data contained in her medical record . maternal data were collected through a questionnaire that included age , race , education , parity , number of prenatal visits , pathology during pregnancy , and duration of rupture of the amniotic sac . neonates were randomly distributed into two groups , a and b , by blocked randomization with a fixed block size of 4 . newborns in group b received instillation of a drop of p - i 2.5% in the bottom of the lower sac of both eyes immediately after a basic eye examination and the collection of conjunctival smears within 3 hours of birth . custom 2.5% p - i eyedrops were prepared by a certified spanish pharmacy ( carreras , barcelona , spain ) following the standards of good manufacturing practices . ocular samples were obtained from both eyes of the newborns by vigorous swabbing across the inferior tarsal conjunctiva . all samples were taken by ophthalmologists and medical personnel previously trained in these procedures by an expert microbiologist from spain ( pm ) . samples were collected with flocked swabs in universal transport medium ( copan italia s.p.a . , brescia , italy ) , stored at 70c , and shipped to the department of microbiology and immunology at the hospital clnico universitario of valladolid , valladolid , spain . dna extraction was performed according to routine laboratory standards with the gxt dna / rna reagents in a genoxtract extractor ( hain lifescience , nehren , germany ) . a multiplex pcr assay that coamplified dna sequences of ct , ng , mg , and an internal control was performed using the bio - rad dx ct / ng / mg kit ( bio - rad , hercules , ca , usa ) , according to manufacturer 's instructions . the ophthalmologist responsible for the study ( ia ) administered the p - i eyedrops . the basic eye examination given to all newborns included pupil light responses , eyelid position and movement , appearance of the conjunctiva , corneal size and transparency , iris appearance and symmetry , lens transparency , quality and symmetry of retinal red reflex ( brckner test ) , and the size , position , and shape of the pupil . an information sheet with explanations about the signs of ophthalmia / acute conjunctivitis ( red eye and ocular secretions and/or eyelid edema ) was given to all participating mothers . they were also given instructions to return with their children for observation after discharge from the maternity ward in any case , even if their baby did not show these signs . the mobile phone number of each mother was noted ( only two mothers gave no indication of telephone contact ) . between the 5th day and 7th day postpartum , phone calls were made to the mothers to bring their infants for observation , especially if they presented ocular signs of on or any other ocular alterations . the minimum sample size was calculated taking in consideration the suspected prevalence of on detected in our previous study   and with an expected reduction of on in at least 30% in the prophylaxis group ( group b ) . to detect a 8% difference between treatment groups with a significance level of 5% and power of 80% , infection rates of the different on bacteria in the endocervical samples of the mothers and in the conjunctival samples of their newborns and the presence of clinical risk factors were compared by  test . the mean age of the 317 participating mothers was 25 years ( range : 1452 years ) , with the majority between 14 and 24 years . some of the mothers , 28% , had a basic level of education and 2.2% were illiterate and the remainders have good reading skill . parity varied from 0 to 9 births , with 82% of the mothers having had 3 previous deliveries . the number of prenatal consultations of the mothers ranged from 0 to 9 , with an average of 4 consultations . thirty - eight cases ( 1.2% ) did not have any prenatal consultation , and 112 cases ( 35% ) had less than four . a total of 96 mothers ( 30.4% ) referred some pathology during pregnancy , predominately urinary infection in 81 mothers ( 25.6% ) and vulvovaginitis in 13 others ( 4.1% ) . a total of 70 mothers ( 22% ) presented with premature rupture of membrane ( prom ) . thirty - one of them had more than 6 hours before delivery , and 39 had less than 6 hours . data were collected from 317 children , but a total of 72 newborns were excluded ( 22.6% ) from the study for low weight , respiratory distress , death , or transfer of the mother to a more specialized center for dystocic delivery ( table 1 ) . newborns , 123 females and 118 males , had a gestational age of 36 to 40 weeks , and an average weight of 3.260  kg . a total of 42 ( 17.1% ) had ocular pathology at the time of delivery , including 31 ( 12.6% ) with conjunctival hyperemia and 11 ( 4.4% ) with on signs , including conjunctival hyperemia plus eyelid edema and/or purulent secretion . three out of the 11 suspected on cases were born from mothers with urinary infections during pregnancy . for five of the newborns with signs of on , the amniotic sac did not rupture prematurely . for the other six due to technical difficulties in the preparation of the samples for transport to spain for pcr analysis , there was no information available regarding the presence or absence of ct , mg , and ng for 6 maternal endocervical samples and 13 newborn conjunctival smears . a total of 543 valid samples were analyzed , 232 from conjunctival smears and 311 from endocervical samples ( table 2 ) . the most common etiologic agent in newborns was ct ( n = 4 ) , followed by mg ( n = 2 ) and then ng ( n = 1 ) . pcr gave positive results in 28 mothers , with a predominance of mg ( n = 19 , 6.1% ) , followed by ct ( n = 8 , 2.1% ) and ng ( n = 2 , 0.5% ) . eleven of the 28 mothers ( 39.3% ) who were infected with ct , mg , or ng presented risk factors for mother - to - child transmission . the factors included prom at delivery time ( n = 6 , all positive for mg ) , vulvovaginitis ( n = 3 ) , urinary tract infection ( n = 1 ) , and urinary tract infection plus vaginitis ( n = 1 ) . of the cases with prom with &gt; 6 hours of labor , one had ng and 3 had mg . the mother - to - child transmission rates were 50% for both ct and ng and 10.5% for mg ( manuscript accepted ) . chi - square analysis showed no significant correlation between cases with external signs of acute conjunctivitis and the presence of urinary or vaginal infections in the mother at the time of delivery . the newborns were randomly distributed into group a , in which the newborns received no ocular prophylactic treatment ( n = 130 ) , and group b , in which the newborns received p - i prophylaxis in both eyes ( n = 115 ) . despite the efforts made with every mother to perform a follow - up visit within 7 to 10 days after delivery , only 16 children were evaluated , nine from group a and seven from group b. ten children , 7 from group a ( controls ) and 3 from group b ( p - i treated ) , did not show any ocular pathology . two ( one from each group ) had acute bilateral conjunctivitis after the third day postpartum . one from group a had a small conjunctival hemorrhage in one eye , and another , from group b , had jaundice of both conjunctivas . the nearly complete disappearance of on in developed countries has been the result of a combination of factors , including prophylactic measures and , above all , better prenatal care [ 13 , 11 , 13 ] . however , currently in angola and other west african nations , no prophylactic measures are used , and it is routinely very difficult to identify the pathogenic agents . thus , diagnosis of on is based on clinical signs , and systematic follow - up of children after birth is almost impossible . ng and ct are currently the most common on etiologic agents , accounting for 60% of all cases in countries without neonatal prophylaxis [ 1416 ] . although gonococcal infection is less common in developed countries , it continues to be a problem in developing countries such as kenya where on has an incidence as high as 4% of live births for ng and 8% for ct can infect the fetus by ascending from the vagina to the uterus and can be present in the newborn at birth as an acute conjunctivitis ; however , other agents usually have an incubation period of 414 days before clinical signs . the aim of this study was to evaluate the efficacy of intervening at birth by providing p - i as a prophylactic agent for on . the weak response of mothers returning for a second ophthalmic examination for their newborns prevented the achievement of this goal . maternal ignorance about the implications of on and cultural factors resulted in poor cooperation of mothers in the follow - up evaluations . even after careful oral and written explanations were given to mothers about the signs and symptoms of on and the ocular and systemic risk for the affected children , and even after most of the mothers were personally contacted by mobile phone , only a very small number of children , 5.7% of the global sample , returned for a second examination . these kinds of problems have been recognized by other authors [ 6 , 7 , 14 ] . since 1990 , p - i has been considered as a potential prophylactic agent of on . several studies have established that p - i has broad spectrum of action and is effective against most agents of on , unlike other prophylactic agents previously used [ 1621 ] . p - i does not induce bacterial resistance , has low toxicity , is of very low cost , and is stable for several months after opening . even more , the transient brownish staining of the ocular surface after instillation could be useful as an indicator of its effective application . in 2002 , isenberg reported that 2.5% p - i was not irritating to the eye . overall , the studies showed that when trained personnel applied p - i after hygienic and general care of the newborn , the results are superior to the other prophylactic agents [ 16 , 19 ] . nevertheless , a number of arguments have been raised against its use as a prophylactic agent , including the possibility that it could be confused with a detergent , the lack of effect of p - i against viruses , and lack of studies proving that it is safe in newborns [ 15 , 19 ] . our study also shows for the first time the frequency of mg infection and some associated clinical findings in a cohort of angolan mothers and their newborns ( manuscript accepted ) . mg is an emerging cause of stis and has been implicated in urogenital infections of men and women worldwide . mg has a prevalence of 7.3% and 2% in high- and low - risk populations , respectively . there is also evidence that this microorganism has the potential to cause ascending infection and may play an important role in the on [ 22 , 23 ] . besides the fact that on is a potential cause of blindness , it can also result in serious systemic complications when nasopharyngeal colonization during vaginal delivery evolves to otitis , pneumonitis arthritis , sepsis , and meningitis [ 13 , 15 , 24 ] . risk factors associated with on are genitourinary infection and prom [ 10 , 13 , 15 ] . in this study , a total of 108 mothers ( 34.1% ) presented some of these risk factors . according to world health organization recommendations , prenatal care of mothers should include identification and management of infections including hiv , syphilis , and other stis . regular screening for stis is not routine in prenatal care in angola although the majority of mothers included in this study did have the prenatal appointments recommended by who . most of them had 4 or more prenatal visits , and the percentage of mothers without prenatal care was considerably lower than the one obtained by our group in the same maternity ward three years before ( 1.2% versus 14.7% ) , showing an improvement of the health system of the country . prenatal studies in mothers of similar age as in this study are very important because the age group between 15 and 25 years of age is considered to be at risk of stis . actually , genital ct ( serotype d - k ) and ng infections are especially common in this age group of african women ; therefore , their neonates are also a risk group for on and systemic complications [ 10 , 13 , 15 ] . our data show that the prevalence of ct in our population was relatively low when compared to previous studies in angola and other african populations . other studies of pregnant women have shown prevalence rates from about 6% in tanzania to 13% in cape verde . the prevalence in our study was much lower , which confirms the relatively low presence of ct in angola . ng and ct infections have common clinical features in women . both produce silent clinical infections , are transmitted efficiently to newborns , and can lead to sterility or chronic infection . both generate silent carriers , causing severe clinical consequences over time . according to historical data , around 3% of newborns with ng ophthalmia will develop complete blindness if untreated , and 20% will have some degree of corneal damage . in africa , the prevalence rate of ng among pregnant women ranges from 0.02% in gabon to 7.8% in south africa . our results showed a lower prevalence of ng , 0.5% , among mothers , confirming our previous results ( manuscript accepted ) . the results of the present study also allow us to estimate the rates of mother - to - child transmission which were 50% for ct and ng and 10.5% for mg . to our knowledge , this is the first study that estimates the rate of mg transmission in angola . in our cohort , a total of 12 newborns presented signs of acute conjunctivitis at the first ophthalmic evaluation , although laboratory tests were positive for mg only in one case . on the other hand , there were six cases that did not show signs of acute conjunctivitis at the first observation but were nevertheless positive for ct , mg , or ng . neonatal infections are of great epidemiological importance , so focused screening efforts should be made to reduce the number of infected pregnant women and thereby the rate of vertical transmission . to a similar extent as seen for other major stis in africa , our work clearly shows a mirror effect when considering matches between infected newborns and their mothers . this result also indicates the need to locally improve public information about primary health care particularly that oriented to eye care . without this knowledge , the participation of community members in studies like ours will not be effective . in summary , we determined the prevalence and incidence of ct , ng , and mg , which are all implicated in angolan cases of newborn on . we also determined that the mother - to - child transmission rates of ct and ng were 50% and 10.5% for mg , which was identified for the first time in angolan newborns . unfortunately , due to low compliance in follow - up clinical assessments , we were unable to achieve the original goal of testing the efficacy and safety of p - i systematic prophylaxis for preventing on .</td>\n",
       "      <td>&lt;S&gt; \\n purpose . to determine the efficacy of povidone - iodine ( p - i ) prophylaxis for ophthalmia neonatorum ( on ) in angola and to document maternal prevalence and mother - to - child transmission rates . methods . &lt;/S&gt; &lt;S&gt; endocervical samples from mothers ( n = 317 ) and newborn conjunctival smears ( n = 245 ) were analysed by multiplex polymerase chain reaction ( pcr ) for chlamydia trachomatis ( ct ) , neisseria gonorrhoeae ( ng ) , and mycoplasma genitalium ( mg ) . &lt;/S&gt; &lt;S&gt; newborns were randomized into a noninterventional group and an interventional group that received a drop of p - i 2.5% bilaterally after conjunctival smear collection . &lt;/S&gt; &lt;S&gt; mothers were trained to identify signs of on and attend a follow - up visit . &lt;/S&gt; &lt;S&gt; results . &lt;/S&gt; &lt;S&gt; forty - two newborns had ocular pathology , and 11 ( 4.4% ) had clinical signs of on at the time of delivery . &lt;/S&gt; &lt;S&gt; maternal pcr was positive for mg ( n = 19 ) , ct ( n = 8) , and ng ( n = 2 ) . &lt;/S&gt; &lt;S&gt; six newborns were positive for ct ( n = 4 ) , mg ( n = 2 ) , and ng ( n = 1 ) . mother - to - child transmission rates were 50% for ct and ng and 10.5% for mg . only 16 newborns returned for follow - up . conclusions . &lt;/S&gt; &lt;S&gt; lack of maternal compliance prevented successful testing of prophylactic p - i efficacy in on prevention . &lt;/S&gt; &lt;S&gt; nevertheless , we documented the prevalence and mother - to - child transmission rates for ct , ng , and mg . these results emphasize the need to develop an effective angolan educational and prophylactic on program . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>the absence of heart sounds on auscultation at the pre - cordial area , confusion during the diagnosis of acute cholecystitis , acute appendicitis and splenetic lesion on the basis of clinical examination raises the suspicion of one rare condition , the situs inversus totalis . this rare condition is characterized by total transposition of thoracic and abdominal viscera , and the predicted incidence is one in 10,000 among the general population.[14 ] during the embryological development , a 270 degree clockwise rotation instead of normal 270 degree anti - clockwise of the developing thoraco - abdominal organs results in mirror image positioning of the abdominal and thoracic viscera . the association of situs inversus totalis with syndromes such as kartagener 's syndrome , cardiac anomalies , spleen malformations and other such clinical entities makes the clinical scenario extremely challenging for the concerned anaesthesiologist . this rare condition is well described by a few surgical and medical journals , but the anaesthetic implications and considerations have not been thoroughly explained by any anaesthesia speciality journal . we are reporting a case of situs inversus totalis who was operated for symptomatic cholelithiasis , with an aim of thorough discussion of the anaesthetic considerations and implications associated with such anatomical abnormalities . a 55-year - old female patient presented to surgical out - patient department ( opd ) with chief complaints of pain and left hypochondrium for the last 3 days accompanied by nausea and vomiting . she presented a big dilemma to the attending surgeon as all the clinical signs and symptoms pointed towards symptomatic cholelithiasis , but the pain in the left hypochondrium totally contradicted the diagnosis . it was only with the help of radiological investigation reports that we were able to confirm it as a case of situs inversus totalis . her chest x - ray typically depicted dextrocardia and spiral computed tomography ( ct ) scan showed transposition of all major abdominal organs . echocardiography was carried out and revealed normal cardiac parameters with an ejection fraction of 60% . pre - anaesthetic evaluation revealed mallampatti class ii patient with a slight retrognathia , a pulse rate of 74/min , blood pressure of 130/78 mmhg and normal profile of laboratory investigations . she had a history of intermittent respiratory tract infections , especially during the winter season . at present , the patient had no systemic complaint pertaining to any organ system except for the gall bladder disease . a decision to operate the patient tablet ranitidine 150 mg and tablet alprazolam 0.25 mg were administered as premedication a night before and 1 h prior to the surgical procedure with a sip of water . in the operation theatre , a good intravenous induction of anaesthesia was achieved with o2 , isoflurane , propofol 120 mg , butorphanol 1 mg and succinylcholine 100 mg . during induction , we encountered masseter spasm immediately after the administration of succinylcholine , which lasted for approximately 45 s , and disappeared spontaneously , but we were able to ventilate the patient with bag and mask during this episode . intubation was extremely difficult as the larynx was placed quite anteriorly and the laryngoscopic view could be best labelled as cormack lehane grade - iii . maintenance of anaesthesia was carried out with injection vecuronium and low concentration of isoflurane in oxygen and nitrous oxide mixture . the peri - operative period was characterized by frequent copious secretions from the respiratory tract that had to be cleared regularly for smooth ventilation . the surgical procedure was smooth and uneventful , and lasted for about 1 h. at the end of surgery , inj . neostigmine and glycopyrrolate were administered but extubation was delayed as the patient developed prolonged post - operative apnoea . therefore , extubation was performed only after establishing the return of all protective airway reflexes and establishment of regular breathing pattern . the recovery period and hospital stay were uneventful and the patient was discharged in a satisfactory condition on the eighth post - operative day and was called for follow - up after 2 weeks . situs inversus totalis is a rare condition , the etiologic factors for which are still not completely understood . there is no established gender or racial difference in its incidence , but genetic predisposition and familial occurrence point towards multiple inheritance patterns . the symptomatic pain of cholelithiasis in these anomalies presents in the left hypochondrium and may mimic other acute abdomen presentations such as hiatus hernia , pancreatitis , duodenal perforation , etc . the scenario of clinical diagnostic dilemma was similar in our patient also , for which we had to take assistance of the radiology department . the most preferred diagnostic technique involves chest and abdominal skiagrams as well as ct scans . the diagnostic parameters of simple skiagrams include the presence of dextrocardia , stomach bubble under the right dome of diaphragm , liver shadow on the left side and the findings of barium meal.[79 ] there was prolonged post - operative apnoea period in our patient . it was only after 30 min of administration of reversal agents that the patient developed a regular breathing pattern with the return of protective airway reflexes that we were able to successfully extubate the patient . our institute does not have the facilities of diagnosing the atypical cholinesterase levels , and this was a limiting factor in this presentation . hence , we recommended the patient to get these investigations done from the top - most research centre and to bring the requisite reports during the follow - up visits . a case of prolonged paralysis after administration of succinylcholine has been reported earlier also in a patient with situs inversus totalis . the challenging aspects for anaesthesiologists in successful management of patients with situs inversus totalis should be thoroughly evaluated , such as : \\n the association of situs inversus with other syndromes and diseases such as kartagener 's syndrome , mucociliary dysfunction , airway anomalies , etc . , which may predispose the patient to numerous varieties of airway difficulties and pulmonary infections that can have considerable implications during induction of anaesthesia and intubation . we encountered masseter spasm immediately after administration of succinylcholine , which was relieved spontaneously in less than 1 min , and there is no evidence in the literature where masseter spasm developed after injection of depolarizing neuromuscular blockers in patients with situs inversus totalis.the syndrome is associated with numerous cardiac anomalies such as atrial septal defects , ventricular septal defects , transposition of great vessels , absent coronary sinus , double - outlet right ventricle , total pulmonary anomalous venous defect and pulmonary valve stenosis either singly or in combination.the spinal deformities like split cord , spina bifida , meningomyelocele , scoliosis , etc . have been described in the literature , and one has to evaluate the patient carefully if any surgery is planned under neuraxial anaesthesia.the ecg electrodes have to be applied in the opposite direction as the changed surface electric polarity may give a false picture of peri - operative ischaemia.in case of cardiothoracic surgery , lung separation throws a challenging task due to transposition of thoracic viscera . insertion of a double - lumen tube will pose a multitude of challenges , and the successful intubation and separation of lungs can not be accomplished without the aid of fibreoptic bronchoscope . the transposition of the thoracic viscera also alters the various anatomical landmarks , and one has to be well acquainted with ultrasound - guided procedures if in case a need arises for invasive central venous cannulation and brachial plexus blockade.situs inversus in kartagener 's syndrome is invariably associated with mucociliary dysfunction . primary ciliary dyskinesia is present in 25% of the patients with situs inversus totalis with an increased probability of developing respiratory complications . there were plenty of mucoserous secretions during the peri - operative period , even though our patient had no active respiratory tract infection . we administered injection hydrocortisone 100 mg , anticipating bronchospasm due to any subclinical infective pathology . the role of bronchodilators , chest physiotherapy , postural drainage , antibiotics and incentive spirometry can not be underestimated and is mandatory in optimizing the pulmonary status before any surgical procedure.in case of cardiac arrhythmias and cardiac arrest , great care has to be taken while applying direct current with defibrillator pads on the right side . a successful resuscitation of such patients requires a thorough knowledge and skills on the part of the attending anaesthesiologist and intensivists . \\n  the association of situs inversus with other syndromes and diseases such as kartagener 's syndrome , mucociliary dysfunction , airway anomalies , etc . , which may predispose the patient to numerous varieties of airway difficulties and pulmonary infections that can have considerable implications during induction of anaesthesia and intubation . we encountered masseter spasm immediately after administration of succinylcholine , which was relieved spontaneously in less than 1 min , and there is no evidence in the literature where masseter spasm developed after injection of depolarizing neuromuscular blockers in patients with situs inversus totalis . the syndrome is associated with numerous cardiac anomalies such as atrial septal defects , ventricular septal defects , transposition of great vessels , absent coronary sinus , double - outlet right ventricle , total pulmonary anomalous venous defect and pulmonary valve stenosis either singly or in combination . the spinal deformities like split cord , spina bifida , meningomyelocele , scoliosis , etc . have been described in the literature , and one has to evaluate the patient carefully if any surgery is planned under neuraxial anaesthesia . the ecg electrodes have to be applied in the opposite direction as the changed surface electric polarity may give a false picture of peri - operative ischaemia . in case of cardiothoracic surgery insertion of a double - lumen tube will pose a multitude of challenges , and the successful intubation and separation of lungs can not be accomplished without the aid of fibreoptic bronchoscope . the transposition of the thoracic viscera also alters the various anatomical landmarks , and one has to be well acquainted with ultrasound - guided procedures if in case a need arises for invasive central venous cannulation and brachial plexus blockade . primary ciliary dyskinesia is present in 25% of the patients with situs inversus totalis with an increased probability of developing respiratory complications . there were plenty of mucoserous secretions during the peri - operative period , even though our patient had no active respiratory tract infection . we administered injection hydrocortisone 100 mg , anticipating bronchospasm due to any subclinical infective pathology . the role of bronchodilators , chest physiotherapy , postural drainage , antibiotics and incentive spirometry can not be underestimated and is mandatory in optimizing the pulmonary status before any surgical procedure . in case of cardiac arrhythmias and cardiac arrest , great care has to be taken while applying direct current with defibrillator pads on the right side . a successful resuscitation of such patients requires a thorough knowledge and skills on the part of the attending anaesthesiologist and intensivists . from the mentioned implications and considerations in a case of situs inversus totalis , it can be safely established that regional anaesthesia is the preferred choice for any infra - umbilical surgery as compared with administration of general anaesthesia , provided that there is no spinal anomaly or dysraphism . the precise diagnosis of situs inversus and a thorough pre - operative evaluation can minimize , to a large extent , the difficulties and the various potential challenges associated with its anaesthetic management . keeping in mind all these considerations and implications , it becomes easier and safer to successfully manage the patients of situs inversus totalis in the operation theatre and intensive care units .</td>\n",
       "      <td>&lt;S&gt; situs inversus totalis is a rare condition with a predicted incidence of one in 10,000 among the general population , the aetiologic factors for which are still not completely understood . in a patient with situs inversus totalis , not just the diagnosis of any acute abdomen pathology is difficult due to distorted anatomy and transposition of thoraco abdominal viscera but equally challenging is the anaesthetic management during the respective surgical procedure . we are reporting a patient who had situs inversus totalis and was operated for open cholecystectomy . &lt;/S&gt; &lt;S&gt; the present case report lays an emphasis on the potential difficulties during anaesthetic management and its various implications . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>child abuse is defined as any physical or emotional injury which causes harm or substantive risk of harm to the child 's health or welfare . it includes sexual abuse , neglect , or being physically dependent upon an addictive drug at birth . types of child abuse , include physical abuse , emotional abuse and neglect , healthcare neglect ( medical and dental ) , physical neglect , sexual abuse , failure to thrive , safety neglect , intentional poisoning , and munchausen syndrome by proxy ( fabricated or induced illness by parent ) . factors contributing to abuse include stress[35 ] ( e.g. life crises such as unemployment or homelessness ) , lack of a support network , substance / alcohol abuse , learned behavior ( many abusers were previously victims ) , and other forms of family violence in the home such as spousal or elderly abuse.[69 ] craniofacial , head , face , and neck injuries happen in more than half of the cases of child abuse . although the oral cavity is a frequent point of sexual abuse in children , obvious oral injuries or infections are rare . the american academy of pediatric dentistry defined dental neglect as the willful failure of a parent or guardian to seek and follow through with treatment necessary to ensure a level of oral health essential for adequate function and freedom from pain and infection . indeed , many dentists who regularly treat children assert that management of dental neglect is part of daily practice . however , based on the literature , dentists feel unprepared to play a child protection role and are unsure what to do if they suspect that a child has been maltreated.[1215 ] the aim of this study was to provide data on prevalence and factors of orofacial lesions relating child abuse in iran to lend evidence to support preventing child abuse . the overall approach was a case - note review of children with child abuse history recording by personnel of social services . the social emergency services are legally eligible to separate the victims of child abuse from their families as well as interview the children and their families in order to solve their problems . research ethical approval was sought from the central social service organization as well as ethics committee of isfahan university of medical sciences . this study was conducted in isfahan , iran . for study inclusion , participating children had to have identifiable records of abuse during 2007 - 2011 . inclusion criteria for children dictated that they had been exposed to at least one child abuse event during this period . the data collected retrospectively for each child , from their existing records , were as follows : \\n type of abuse : physical , sexualpatient characteristics : age , gender \\n  type of abuse : physical , sexual patient characteristics : age , gender descriptive statistics were used to describe child - related sociodemographic and clinical data . type of child abuse , gender , age , and the type of abuser was described using appropriate measures of spread . exploratory analyses including bivariate analyses and correlation were used to analyze the relationship of different variables such as child gender , age , abuse experience , abuse type as well as abuser type . all statistical tests were repeated using these data and found the same relationships between variables as identified for the original data set . for study inclusion , participating children had to have identifiable records of abuse during 2007 - 2011 . inclusion criteria for children dictated that they had been exposed to at least one child abuse event during this period . the data collected retrospectively for each child , from their existing records , were as follows : \\n type of abuse : physical , sexualpatient characteristics : age , gender \\n  type of abuse : physical , sexual patient characteristics : age , gender type of child abuse , gender , age , and the type of abuser was described using appropriate measures of spread . exploratory analyses including bivariate analyses and correlation were used to analyze the relationship of different variables such as child gender , age , abuse experience , abuse type as well as abuser type . all statistical tests were repeated using these data and found the same relationships between variables as identified for the original data set . data were obtained for 301 children from social services records . there was an equal gender distribution amongst children gender . the mean age of children when abuse had been occurred was 8 years ( sd = 1.68 ) which was categorized to seven age groups [ table 1 ] , and there were approximately an equal number of boys and girls [ table 2 ] . children had a high physical experience ( 66.1% ) ; of these children , at least 69% sustained trauma to the face and mouth . emotional abuse was 77.1% , neglect was 64.1% , and lower experience of sexual abuse which was 4.1% . age of children having child abuse records categorized to seven age groups the gender distribution of children having child abuse records during 2007 - 2011 exploratory bivariate analyses revealed a significant relationship between the frequency of abuse with gender and age , p = 0.008 and 0.015 , respectively . having problem such as being a mental retired and hyperactive child shows significant relationship with gender , p = 0.03 and 0.02 , respectively , which was in the favor of males . there was a strong relationship between gender and abuser which shows girls have been affected by stepfathers , p = 0.001 . there was also strong relationship between age of child and age of abuser , p = 0.001 . however , there were no significant differences regarding age of abuser and gender of the child . the mean age of children when abuse had been occurred was 8 years ( sd = 1.68 ) which was categorized to seven age groups [ table 1 ] , and there were approximately an equal number of boys and girls [ table 2 ] . children had a high physical experience ( 66.1% ) ; of these children , at least 69% sustained trauma to the face and mouth . emotional abuse was 77.1% , neglect was 64.1% , and lower experience of sexual abuse which was 4.1% . age of children having child abuse records categorized to seven age groups the gender distribution of children having child abuse records during 2007 - 2011 exploratory bivariate analyses revealed a significant relationship between the frequency of abuse with gender and age , p = 0.008 and 0.015 , respectively . having problem such as being a mental retired and hyperactive child shows significant relationship with gender , p = 0.03 and 0.02 , respectively , which was in the favor of males . there was a strong relationship between gender and abuser which shows girls have been affected by stepfathers , p = 0.001 . there was also strong relationship between age of child and age of abuser , p = 0.001 . however , there were no significant differences regarding age of abuser and gender of the child . the main purpose of this study was to obtain preliminary data to provide data on child abuse - related orofacial lesions in order to lend evidence to prevent child abuse . this was considered important as there is a paucity of current data on the incidence of child abuse in all over the world and there is recognized need to evaluate the evidence to support dentists in different aspects of service provision . results of previous studies show that trauma to the head and associated areas occurs in approximately 50% of the cases of physically abused children and soft tissue injuries  most frequently bruises  are the most common injury to head and face . these findings make it obvious that dentists are in a position to detect child abuse . the british study by skinner and castle ( 1967 ) documented the injuries to 78 abused children requiring medical attention ; of these children , at least 34 ( 43.5% ) sustained trauma to the face and mouth . this may be an underestimate since some of the bruises were reported without noting the region . the majority of the injuries were bruises , but they also included lacerations , bites , and abrasions which support the results of our study . however , in our study , 60% of children had trauma to the face and head which was a higher prevalence . national figures in usa indicate that as many as 1 million children are abused and/or neglected annually and of these about 1000 die each year . if we assume that half of these cases involve trauma to the head , as is indicated in the literature , our profession is definitely in a position to detect and assist substantial numbers of victim - abused children . in this way , we can help to refer them to social security agencies and prevent further continuing trauma to the children by bringing help to these troubled families . the majority of victim - abused cases in our study were lower than ten years old . stress and trauma of them permanently impacted any aspects of their future life , especially in girls . the cost of orofacial trauma makes heavy burden on social security agencies and their families . the main purpose of this study was to obtain preliminary data to provide data on child abuse - related orofacial lesions in order to lend evidence to prevent child abuse . this was considered important as there is a paucity of current data on the incidence of child abuse in all over the world and there is recognized need to evaluate the evidence to support dentists in different aspects of service provision . results of previous studies show that trauma to the head and associated areas occurs in approximately 50% of the cases of physically abused children and soft tissue injuries  most frequently bruises  are the most common injury to head and face . these findings make it obvious that dentists are in a position to detect child abuse . the british study by skinner and castle ( 1967 ) documented the injuries to 78 abused children requiring medical attention ; of these children , at least 34 ( 43.5% ) sustained trauma to the face and mouth . this may be an underestimate since some of the bruises were reported without noting the region . the majority of the injuries were bruises , but they also included lacerations , bites , and abrasions which support the results of our study . however , in our study , 60% of children had trauma to the face and head which was a higher prevalence . national figures in usa indicate that as many as 1 million children are abused and/or neglected annually and of these about 1000 die each year . if we assume that half of these cases involve trauma to the head , as is indicated in the literature , our profession is definitely in a position to detect and assist substantial numbers of victim - abused children . in this way , we can help to refer them to social security agencies and prevent further continuing trauma to the children by bringing help to these troubled families . the majority of victim - abused cases in our study were lower than ten years old . stress and trauma of them permanently impacted any aspects of their future life , especially in girls . the cost of orofacial trauma makes heavy burden on social security agencies and their families . this is the first study in iran to provide data for the evaluation of orofacial lesions relating child abuse . preliminary data suggest that there are strong evidence regarding the incidence of child abuse relating orofacial lesions which dentists should be aware of them . future trials may draw based on these useful baseline data to help their study design .</td>\n",
       "      <td>&lt;S&gt; background : family violence , including child abuse , neglect , and domestic violence , is a public health problem . &lt;/S&gt; &lt;S&gt; the aim of this study was to provide data on prevalence and factors of orofacial lesions relating child abuse in iran to lend evidence to support preventing child abuse.materials and methods : the overall approach was a case - note review of children having child abuse note , recording by personnel of social services . &lt;/S&gt; &lt;S&gt; research ethical approval was sought from the central social service organization . &lt;/S&gt; &lt;S&gt; this study was conducted in isfahan , iran ( 2011).result : the mean age of children , when abuse had been occurred was 8 years ( sd = 1.68 ) , and there were approximately an equal number of boys and girls . &lt;/S&gt; &lt;S&gt; children had a high physical experience ( 66.1%).of these children , at least 60% sustained trauma to the face and mouth . &lt;/S&gt; &lt;S&gt; emotional abuse was 77.1% , neglect was 64.1% , and lower experience of sexual abuse which was 4.1% . &lt;/S&gt; &lt;S&gt; there was a strong relationship between gender and abuser which shows girls have been affected by stepfathers ( p = 0.001).conclusion : preliminary data suggest that there are strong evidence regarding the incidence of child abuse relating orofacial lesions which dentists should be aware of them . &lt;/S&gt; &lt;S&gt; future trials may draw on these useful baseline data to help their study design . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>alopecia areata ( aa ) is common cause of reversible hair loss afflicting approximately 1 to 2% of the general population . a wide range of clinical presentations can occur , from a single patch of hair loss to complete loss of hair on the scalp ( alopecia totalis , at ) or over the entire body ( alopecia universalis , au ) . the cause of aa is unknown , although there is evidence to suggest that the link between lymphocytic infiltration of the follicle and the disruption of the hair follicle cycle in aa may be provided by a combination of factors , including cytokine release , cytotoxic t - cell activity , and apoptosis . it is also considered that a disequilibrium in the production of cytokines , with a relative excess of proinflammatory and th1 types , vs. anti - inflammatory cytokines may be involved in the persistence of aa lesions , as shown in human scalp biopsies . tumor necrosis factor - alpha ( tnf- ) is a multifunctional proinflammatory cytokine which has been implicated in the pathogenesis of several chronic inflammatory disorders with an autoimmune component . this cytokine is synthesized in epidermal keratinocytes along with several other cytokines and is known to be a very potent inhibitor of proliferation . the changes in serum tnf- levels were found in many diseases , such as psoriasis and systemic lupus erythematosus . in some of these diseases , serum tnf- concentration correlated with activity and intensity of the disease , and although it is well known that multiple cytokines simultaneously play role in aa , many authors have measured only one particular cytokine . our study has focused only on tnf- because there are only a few studies that have measured the serum levels of this cytokine with controversial results . therefore , the aim of our study was to evaluate serum levels of tnf- in aa patients and control subjects , and also to assess the difference between the localized and extensive forms of the disease such as at and au . the study included 60 patients with aa ( 36 females and 24 males ; median age , 35.6 years ) . forty - six patients had localized aa ( laa ) and 14 patients had at , au , or at / au . the patients who had received any treatment within previous 3 months were excluded from the study , as well as patients with any diseases based on the immune pathomechanism , which could influence serum concentrations of tnf-. control group consisted of 20 generally healthy subjects ( 11 females and 9 males ; median age , 32.6 years ) . serum levels of tnf- were measured by an enzyme - linked immunosorbent assay technique , using quantikine human tnf- immunoassay ( r and d system , minneapolis , mn , usa ) . standards and samples are pipetted into the wells and any tnf- present is bound by the immobilized antibody . after washing away any unbound substances , an enzyme - linked polyclonal antibody specific for tnf- is added to the wells . following a wash to remove any unbound antibody - enzyme reagent , a substrate solution is added to the wells and color develops in proportion to the amount of tnf- bound in the initial step . the test distribution was done by kolmogorov - smirnov test , and comparisons were performed by t - test . the study group comprised of 60 ( 36 females and 24 males ; the mean age was 35.6 years , ranging from 5 to 69 years ) patients with aa and 20 healthy controls ( 11 females and 9 males ; the mean age 32.6 years , ranging from 6 to 63 years ) . there were no significant difference in age and female / male ratio between the patients and controls ( p &gt; 0.05 ) . the mean duration of aa was 14.5  25.4 ( range , 1 - 119 months ) . in the total of patients with aa , 46 of them were laa and 14 were at , au , or at / au group . serum tnf- levels ranged from 8.8 to 17.0 pg / ml , with the highest values observed in the au patients . the mean serum tnf- in aa patients was 10.31  1.20 pg / ml ( mean  sd ) , whereas that of laa or extensive ( at , au , or at / au ) was 10.16  0.79 pg / ml or 10.40  1.03 pg / ml , respectively . patients with longer duration of the disease had higher concentration of tnf- , but not significantly [ figure 1 ] . correlation between the duration of the aa and concentration of tnf-,r = 0.034 ;  ( rho ) = 0.1142 ; 95% ci ( -0,144 ; 0,358 ) ; p &gt; 0.05;-n.s serum levels of tnf- in patients with aa were significantly higher than those in controls ( p = 0.044 ) . there was no significant difference in levels of tnf- between patients with laa and the extensive group ( p=0.2272 ) . serum concentrations ( meansd ) of tnf- in patients with aa , laa , at / au and in healthy controls recent progress in the understanding of aa has shown that the regulation of local and systemic cytokines plays an important role in its pathogenesis . hair loss may occur because proinflammatory cytokines interfere with the hair cycle , leading to premature arrest of hair cycling with cessation of hair growth . this concept may explain typical clinical features of aa such as a progression pattern in centrifugal waves and spontaneous hair regrowth in concentric rings , suggesting the presence of soluble mediators within affected areas of the scalp . tnf- is a multifunctional proinflammatory cytokine which has been implicated in the pathogenesis of many infections and inflammatory disorders . however , this cytokine not only acts as mediator of immunity and inflammation , but also affects not - immune responses within tissues such as cell proliferation and differentiation . in vitro studies have shown that tnf- , along with il-1 and il- , causes vacuolation of matrix cells , abnormal keratinization of the follicle bulb and inner root sheath , as well as disruption of follicular melanocytes and the presence of melanin granules within the dermal papilla . experiments in cultured human hair follicles by hoffmann et al . showed that tnf- completely abrogated hair growth . additionally , tnf- induced the formation of a club - like hair follicle , similar to catagen morphology of the hair bulb . a study by thein et al . examined cytokine profiles of infiltrating activated t - cells from the margin of involved aa lesions . it was found that t - cell clones from involved lesions inhibited the proliferation of neonatal keratinocytes . in examining the cytokine profiles and relating them to regulatory capacity , the authors found that t - cell clones that released high amounts of ifn- and/or tnf- inhibited keratinocyte growth . a limited number of studies in the literature have evaluated the serum levels of tnf- in patients with aa . the results presented in our study demonstrate that the mean serum levels of tnf- were significantly elevated in aa patients in comparison with healthy subjects . there was no significant difference in levels of tnf- between patients with laa and the extensive group . in contrast to our results , teraki et al . reported that serum levels of tnf- in patients with laa were significantly higher than those in patients with au . in the study of koubanova and gadjlgoroeva , serum levels of tnf- in patients with aa did not differ from that in controls . however , tnf- was lower in patients with severe form of aa than in patients with mild form . they hypothesized that similar levels of tnf- in patients with both forms of aa and controls may indirectly indicate the absence of systemic immunopathological reactions in patients with aa , and the lowering of tnf- level in the mild form may indicate the tendency to formation of immunodeficiency in patients with severe aa . in addition , lis et al . found that serum levels of stnf- receptor type i were significantly elevated in patients with aa in comparison with healthy subjects . as they conclude , these results indicate that immune mechanisms in aa are characterized by activation of t - cells and other cells , possibly keratinocytes . tnf- seems to be a useful indicator of the activity of aa and that it may play an important role in the development of this disease . further investigations are required to clarify the pathogenic role and clinical significance of tnf- and these findings may provide important clues to assist in the development of new therapeutic strategies for patients with aa .</td>\n",
       "      <td>&lt;S&gt; background : alopecia areata ( aa ) is a common form of localized , nonscarring hair loss . it is characterized by the loss of hair in patches , total loss of scalp hair ( alopecia totalis , at ) , or total loss of body hair ( alopecia universalis , au ) . the cause of aa is unknown , although most evidence supports the hypothesis that aa is a t - cell - mediated autoimmune disease of the hair follicle and that cytokines play an important role.aims:the aim of the study was to compare the serum levels of tumor necrosis factor - alpha ( tnf- ) in patients with aa and the healthy subjects and also to investigate the difference between the localized form of the disease with the extensive forms like at and au.materials and methods : sixty patients with aa and 20 healthy controls were enrolled in the study . &lt;/S&gt; &lt;S&gt; forty - six patients had localized aa ( laa ) , and 14 patients had at , au , or at / au . &lt;/S&gt; &lt;S&gt; the serum levels of tnf- were measured using enzyme - linked immunoassay techniques.results:serum levels of tnf- were significantly higher in aa patients than in controls ( 10.31  1.20 pg ml vs 9.59  0.75 pg / ml , respectively ) . &lt;/S&gt; &lt;S&gt; there was no significant difference in serum levels of tnf- between patients with laa and those with extensive forms of the disease.conclusion:our findings support the evidence that elevation of serum tnf- is associated with aa . &lt;/S&gt; &lt;S&gt; the exact role of serum tnf- in aa should be additionally investigated in future studies . &lt;/S&gt;</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>intestinal pseudo - obstruction is a clinical syndrome that involves bowel obstruction without any mechanical cause.1 it can be either acute or chronic and may occur in the small bowel or the colon . chronic intestinal pseudo - obstruction ( cip ) usually exhibits a relapsing clinical course and may either be idiopathic or occur due to other systemic disorders . pregnancy can be a precipitating factor of colonic pseudo - obstruction.2 recently , we experienced a case of a woman with intractable colonic pseudo - obstruction that was aggravated after pregnancy . endoscopic decompression could delay surgical treatment until a gestational age of 21 weeks and finally a full - term delivery could be achieved . a 31-year - old woman presented with abdominal pain , distension , and constipation at a fetal gestational age of 17 weeks . the patient 's symptoms aggravated at fetal gestational age of 10 weeks and her last pregnancy was aborted 2 years earlier at another hospital due to severe intestinal obstruction . she had suffered from unexplainable obstructive symptoms 5 years earlier and was diagnosed with cip because of repetitive obstructive symptoms without any visible obstructive lesion observed on abdominopelvic computed tomography ( ct ) , barium enema , or colonoscopy ( fig . she was most recently admitted to our hospital 5 months ago because of recurrent obstructive symptoms . because the intravenous administration of neostigmine was not effective , colonoscopic decompression was performed at that time . laboratory examination revealed a white blood cell count of 8,120/mm , a hemoglobin level of 9.9 g / dl , a platelet count of 281,000/mm , a blood sugar level of 99 mg / dl , a blood urea nitrogen level of 12.2 mg / dl , a creatinine level of 1.1 mg / dl , a sodium level of 139 meq / l , a potassium level of 3.8 meq / l , a chloride level of 104 meq / l , a calcium level of 8.7 mg / dl , a phosphorous level of 3.6 mg / dl , and a magnesium level of 1.8 mg / dl . neither radiologic study nor neostigmine administration could be performed because of the risk to the fetus . colonoscopy was performed to provide decompression because the patient 's pain and distension worsened after 5 days of conservative care . the colonoscope was passed just beyond the splenic flexure , at which point the lumen was found to be obstructed by a large fecal bezoar ( fig . 2 ) . attempts to break the fecal bezoar with an endoscopic snare ( olympus disposable electrosurgical snare sd-210u-10 ; olympus medical systems corp . , tokyo , japan ) met with limited success . after suctioning the retained gas and contents of the proximal colon , a drainage catheter ( enbd-7-liguory , 7 fr , 250 cm ; cook medical inc . , winston - salem , nc , usa ) was placed through the working channel of the endoscope . the catheter was irrigated periodically with normal saline to prevent obstruction and ensure that the intestinal contents and gas were drained effectively . the patient 's symptoms improved for a few days after decompression but recurred soon thereafter . additional two attempts at endoscopic decompression were made and a new drainage catheter was inserted in each of the attempts until surgery could be performed . subtotal colectomy with endileostomy was performed electively at a fetal gestational age of 21 weeks . ileorectal anastomosis could not be performed due to the enlarged uterus and was postponed until after delivery . the proximal portion from the cecum to the descending colon was dilated up to 16 cm in diameter and a focally narrowed transitional zone was observed around the sigmoid colon ( fig . full colonic sections were obtained for histopathological analysis , which revealed no apparent histopathological alteration in the muscularis propria or neural plexuses ( fig . however , immunohistochemistry for c - kit ( cd 117 ) revealed fewer interstitial cells of cajal ( iccs ) in dilated portions of the colon compared to the undilated part ( fig . stool passage was normalized after surgery and pregnancy was maintained and resulted in a full - term delivery . pregnancy was one of the most important precipitating factors contributing to cip in this case . elevated progesterone , prostaglandin , and glucagon levels as well as the pressure of the gravid uterus are suggested as factors that exacerbate the pseudo - obstruction experienced by the patient during pregnancy.2 constipation , progressive abdominal pain , and abdominal distension are common symptoms in pregnancy and may delay the diagnosis of pseudo - obstruction . moreover , abdominal radiographs and ct scans , which may be required for evaluation and are typically diagnostic , may be limited in the case of a pregnant woman due to the desire to limit radiation exposure to the fetus.3 fortunately , our patient had already been diagnosed with cip . intravenous administration of neostigmine is recommended in patients who do not show improvement after conservative management . neostigmine may lead to resolution of acute colonic pseudo - obstruction and may also be effective in acute exacerbations of cip.4,5 in this case , the patient 's unresponsiveness to neostigmine 5 months earlier and the possible risks to the fetus deterred the authors from using it again . endoscopic decompression can be another option for treating colonic pseudo - obstruction when neostigmine is not effective or contraindicated.6 efficacy of endoscopic decompression is increased when a cecal tube is inserted at the time of colonoscopy . although the cecum could not be intubated due to a large fecal bezoar in our case , placements of a decompression tube beyond the obstruction site was available . tubes for colorectal decompression include commercial tubes specifically aimed for decompression , tubes for enteroclysis , and nasogastric tubes . however , commercial tubes for colorectal decompression are not currently available in korea . moreover , all the tubes above should be inserted by seldinger technique under fluoroscopic guidance . in contrast , nasobiliary tubes can be easily inserted through the working channel of endoscopy without fluoroscopy . radiation exposure should be avoided since our patient was pregnant.7 although the small diameter of nasobiliary tubes limits effective decompression , periodic irrigation can prevent plugging of the tube and ensure effective drainage . colonoscopy may be relatively safe without large fetal risks during the second trimester with limited data during the other trimesters.8 repetitive colonoscopic decompression was able to relieve the patient 's symptoms until the second trimester of pregnancy when surgery was available . cip can be classified into three major categories based on the underlying pathological abnormality : enteric visceral myopathy , neuropathy , and mesenchymopathy.9 this classification is based on the involvement of smooth muscle cells , the enteric nervous system , and iccs , respectively . choe et al.10 examined the surgical specimens of patients who received surgery for intractable constipation and found that a substantial number of patients presented with a distinct transitional zone with segmental hypoganglionosis . another study by do et al.11 suggested a novel classification of hypoganglionosis patients into two groups : type i with a focally narrowed transition zone and type ii without a transition zone . our patient showed dilation of the proximal to mid colon with distinct narrowing around the sigmoid colon . however , pathologic studies of the narrowed segment did not reveal any abnormality in the ganglion cells or the neural plexuses . the only pathologic abnormality were marked reduction of iccs in multifocal areas of the dilated colon . it has been speculated that alterations in the icc network may result in impaired control of peristalsis resulting in cip . however , the diagnostic criteria of mesenchymopathic type of cip is poorly defined with reference to the mean number of iccs and distribution of the iccs . it is also not clear whether the multifocal decrease of iccs in our patient is the primary cause of cip or secondary to continuous colon dilation . in summary , we have reported a case of cip which was aggravated severely by pregnancy . however , in such cases , radiologic studies and administration of neostigmine may be limited due to the associated fetal risks . endoscopic decompression can be performed repetitively with minimal fetal risks until elective surgery can be performed .</td>\n",
       "      <td>&lt;S&gt; chronic intestinal pseudo - obstruction is a rare clinical syndrome which is characterized by intestinal obstruction without occluding lesions in the intestinal lumen and pregnancy is one of the important aggravating factors . here &lt;/S&gt; &lt;S&gt; , we report a case of a woman with intractable intestinal pseudo - obstruction that was precipitated by pregnancy . &lt;/S&gt; &lt;S&gt; she could not make any stool passage for more than 4 weeks until a fetal gestational age of 17 weeks was reached . &lt;/S&gt; &lt;S&gt; however , the patient could be maintained by repetitive colonoscopic decompressions and finally total colectomy could be performed successfully at a fetal gestational age of 21 weeks . &lt;/S&gt;</td>\n",
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
   "execution_count": 14,
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
     "execution_count": 14,
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
   "execution_count": 15,
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
     "execution_count": 15,
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
   "execution_count": 16,
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
   "execution_count": 17,
   "metadata": {
    "id": "eXNLu_-nIrJI"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0e2311d90b0647ef807560c714679a7b",
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
       "model_id": "3905f74f2248472f8ee6f0066c4a6e0a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/3.02k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "aa212d1a8a5b43548ffbb95c466d9a8e",
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
       "model_id": "555fee64b93b45528ffbb1b315acf42b",
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
   "execution_count": 18,
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
     "execution_count": 18,
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
   "execution_count": 19,
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
     "execution_count": 19,
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
   "execution_count": 20,
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
   "execution_count": 21,
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
    "The max input length of `google/pegasus-large` is 1024, so `max_input_length = 1024`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
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
   "execution_count": 23,
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
     "execution_count": 23,
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
   "execution_count": 24,
   "metadata": {
    "id": "DDtsaJeVIrJT"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8ccb92bca58f46d4b5d476d83c2a2635",
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
       "model_id": "07138111ac2547a9884a2ddfabfbcb86",
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
       "model_id": "122dc84cb3b54635a8e33b50d945e570",
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
   "execution_count": 25,
   "metadata": {
    "id": "TlqNaB8jIrJW"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "73a7aecf5713473ba9641a5ecc856a33",
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
   "execution_count": 26,
   "metadata": {
    "id": "phpGhdw_ir69"
   },
   "outputs": [],
   "source": [
    "batch_size = 2\n",
    "model_name = model_checkpoint.split(\"/\")[-1]\n",
    "args = Seq2SeqTrainingArguments(\n",
    "    f\"{model_name}-finetuned-Pubmed\",\n",
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
   "execution_count": 27,
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
   "execution_count": 28,
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
   "execution_count": 29,
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
      "Cloning https://huggingface.co/Kevincp560/pegasus-large-finetuned-Pubmed into local empty directory.\n"
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
   "execution_count": 30,
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
       "      [5000/5000 2:22:20, Epoch 5/5]\n",
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
       "      <td>2.065000</td>\n",
       "      <td>1.826186</td>\n",
       "      <td>37.198600</td>\n",
       "      <td>14.368500</td>\n",
       "      <td>23.715300</td>\n",
       "      <td>33.071300</td>\n",
       "      <td>218.902000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>1.955200</td>\n",
       "      <td>1.793346</td>\n",
       "      <td>38.066300</td>\n",
       "      <td>14.781300</td>\n",
       "      <td>23.841200</td>\n",
       "      <td>33.957400</td>\n",
       "      <td>217.488000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>1.898300</td>\n",
       "      <td>1.776832</td>\n",
       "      <td>38.397500</td>\n",
       "      <td>15.098300</td>\n",
       "      <td>24.024700</td>\n",
       "      <td>34.314000</td>\n",
       "      <td>222.320000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1.882000</td>\n",
       "      <td>1.768684</td>\n",
       "      <td>39.131100</td>\n",
       "      <td>15.416700</td>\n",
       "      <td>24.297800</td>\n",
       "      <td>35.078000</td>\n",
       "      <td>222.564000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>1.845600</td>\n",
       "      <td>1.766925</td>\n",
       "      <td>39.110700</td>\n",
       "      <td>15.412700</td>\n",
       "      <td>24.372900</td>\n",
       "      <td>35.123600</td>\n",
       "      <td>226.594000</td>\n",
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
      "Saving model checkpoint to pegasus-large-finetuned-Pubmed/checkpoint-500\n",
      "Configuration saved in pegasus-large-finetuned-Pubmed/checkpoint-500/config.json\n",
      "Model weights saved in pegasus-large-finetuned-Pubmed/checkpoint-500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/checkpoint-500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/checkpoint-500/special_tokens_map.json\n",
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/special_tokens_map.json\n"
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
      "Saving model checkpoint to pegasus-large-finetuned-Pubmed/checkpoint-1000\n",
      "Configuration saved in pegasus-large-finetuned-Pubmed/checkpoint-1000/config.json\n",
      "Model weights saved in pegasus-large-finetuned-Pubmed/checkpoint-1000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/checkpoint-1000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/checkpoint-1000/special_tokens_map.json\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "Saving model checkpoint to pegasus-large-finetuned-Pubmed/checkpoint-1500\n",
      "Configuration saved in pegasus-large-finetuned-Pubmed/checkpoint-1500/config.json\n",
      "Model weights saved in pegasus-large-finetuned-Pubmed/checkpoint-1500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/checkpoint-1500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/checkpoint-1500/special_tokens_map.json\n"
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
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/special_tokens_map.json\n"
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
      "Saving model checkpoint to pegasus-large-finetuned-Pubmed/checkpoint-2000\n",
      "Configuration saved in pegasus-large-finetuned-Pubmed/checkpoint-2000/config.json\n",
      "Model weights saved in pegasus-large-finetuned-Pubmed/checkpoint-2000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/checkpoint-2000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/checkpoint-2000/special_tokens_map.json\n",
      "Deleting older checkpoint [pegasus-large-finetuned-Pubmed/checkpoint-500] due to args.save_total_limit\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "Saving model checkpoint to pegasus-large-finetuned-Pubmed/checkpoint-2500\n",
      "Configuration saved in pegasus-large-finetuned-Pubmed/checkpoint-2500/config.json\n",
      "Model weights saved in pegasus-large-finetuned-Pubmed/checkpoint-2500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/checkpoint-2500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/checkpoint-2500/special_tokens_map.json\n"
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
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/special_tokens_map.json\n"
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
      "Deleting older checkpoint [pegasus-large-finetuned-Pubmed/checkpoint-1000] due to args.save_total_limit\n"
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
      "Saving model checkpoint to pegasus-large-finetuned-Pubmed/checkpoint-3000\n",
      "Configuration saved in pegasus-large-finetuned-Pubmed/checkpoint-3000/config.json\n",
      "Model weights saved in pegasus-large-finetuned-Pubmed/checkpoint-3000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/checkpoint-3000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/checkpoint-3000/special_tokens_map.json\n",
      "Deleting older checkpoint [pegasus-large-finetuned-Pubmed/checkpoint-1500] due to args.save_total_limit\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "Saving model checkpoint to pegasus-large-finetuned-Pubmed/checkpoint-3500\n",
      "Configuration saved in pegasus-large-finetuned-Pubmed/checkpoint-3500/config.json\n",
      "Model weights saved in pegasus-large-finetuned-Pubmed/checkpoint-3500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/checkpoint-3500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/checkpoint-3500/special_tokens_map.json\n"
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
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/special_tokens_map.json\n"
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
      "Deleting older checkpoint [pegasus-large-finetuned-Pubmed/checkpoint-2000] due to args.save_total_limit\n"
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
      "Saving model checkpoint to pegasus-large-finetuned-Pubmed/checkpoint-4000\n",
      "Configuration saved in pegasus-large-finetuned-Pubmed/checkpoint-4000/config.json\n",
      "Model weights saved in pegasus-large-finetuned-Pubmed/checkpoint-4000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/checkpoint-4000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/checkpoint-4000/special_tokens_map.json\n",
      "Deleting older checkpoint [pegasus-large-finetuned-Pubmed/checkpoint-2500] due to args.save_total_limit\n",
      "The following columns in the evaluation set  don't have a corresponding argument in `PegasusForConditionalGeneration.forward` and have been ignored: article, abstract. If article, abstract are not expected by `PegasusForConditionalGeneration.forward`,  you can safely ignore this message.\n",
      "***** Running Evaluation *****\n",
      "  Num examples = 500\n",
      "  Batch size = 2\n",
      "Saving model checkpoint to pegasus-large-finetuned-Pubmed/checkpoint-4500\n",
      "Configuration saved in pegasus-large-finetuned-Pubmed/checkpoint-4500/config.json\n",
      "Model weights saved in pegasus-large-finetuned-Pubmed/checkpoint-4500/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/checkpoint-4500/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/checkpoint-4500/special_tokens_map.json\n"
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
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/special_tokens_map.json\n"
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
      "Deleting older checkpoint [pegasus-large-finetuned-Pubmed/checkpoint-3000] due to args.save_total_limit\n"
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
      "Saving model checkpoint to pegasus-large-finetuned-Pubmed/checkpoint-5000\n",
      "Configuration saved in pegasus-large-finetuned-Pubmed/checkpoint-5000/config.json\n",
      "Model weights saved in pegasus-large-finetuned-Pubmed/checkpoint-5000/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/checkpoint-5000/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/checkpoint-5000/special_tokens_map.json\n",
      "Deleting older checkpoint [pegasus-large-finetuned-Pubmed/checkpoint-3500] due to args.save_total_limit\n",
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
       "TrainOutput(global_step=5000, training_loss=1.9571337890625, metrics={'train_runtime': 8543.0323, 'train_samples_per_second': 1.171, 'train_steps_per_second': 0.585, 'total_flos': 2.885677635649536e+16, 'train_loss': 1.9571337890625, 'epoch': 5.0})"
      ]
     },
     "execution_count": 30,
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
      "Saving model checkpoint to pegasus-large-finetuned-Pubmed\n",
      "Configuration saved in pegasus-large-finetuned-Pubmed/config.json\n",
      "Model weights saved in pegasus-large-finetuned-Pubmed/pytorch_model.bin\n",
      "tokenizer config file saved in pegasus-large-finetuned-Pubmed/tokenizer_config.json\n",
      "Special tokens file saved in pegasus-large-finetuned-Pubmed/special_tokens_map.json\n"
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
       "model_id": "6f99c3013dab45409455bca06a81335d",
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
   "name": "pegasus-large-pubmed-summary-final.ipynb",
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
