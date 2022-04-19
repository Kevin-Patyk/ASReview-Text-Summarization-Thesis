{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "28dc18e4",
   "metadata": {},
   "source": [
    "# Importing the required packages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ceba5833-2c6f-44f0-b72e-176c496bca93",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting transformers\n",
      "  Downloading transformers-4.18.0-py3-none-any.whl (4.0 MB)\n",
      "\u001b[K     |████████████████████████████████| 4.0 MB 20.4 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: torch in ./miniconda3/envs/fastai/lib/python3.8/site-packages (1.9.1)\n",
      "Requirement already satisfied: pandas in ./miniconda3/envs/fastai/lib/python3.8/site-packages (1.3.2)\n",
      "Collecting filelock\n",
      "  Downloading filelock-3.6.0-py3-none-any.whl (10.0 kB)\n",
      "Collecting regex!=2019.12.17\n",
      "  Downloading regex-2022.3.15-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (764 kB)\n",
      "\u001b[K     |████████████████████████████████| 764 kB 38.0 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: packaging>=20.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (21.0)\n",
      "Requirement already satisfied: tqdm>=4.27 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (4.62.2)\n",
      "Collecting huggingface-hub<1.0,>=0.1.0\n",
      "  Downloading huggingface_hub-0.5.1-py3-none-any.whl (77 kB)\n",
      "\u001b[K     |████████████████████████████████| 77 kB 5.5 MB/s  eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: numpy>=1.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (1.20.3)\n",
      "Requirement already satisfied: requests in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (2.26.0)\n",
      "Collecting tokenizers!=0.11.3,<0.13,>=0.11.1\n",
      "  Downloading tokenizers-0.12.1-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.6 MB)\n",
      "\u001b[K     |████████████████████████████████| 6.6 MB 35.2 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting sacremoses\n",
      "  Downloading sacremoses-0.0.49-py3-none-any.whl (895 kB)\n",
      "\u001b[K     |████████████████████████████████| 895 kB 30.8 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: pyyaml>=5.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from transformers) (5.4.1)\n",
      "Requirement already satisfied: typing_extensions in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from torch) (3.10.0.2)\n",
      "Requirement already satisfied: python-dateutil>=2.7.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from pandas) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2017.3 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from pandas) (2021.1)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from packaging>=20.0->transformers) (2.4.7)\n",
      "Requirement already satisfied: six>=1.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from python-dateutil>=2.7.3->pandas) (1.16.0)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests->transformers) (2.0.4)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests->transformers) (1.26.6)\n",
      "Requirement already satisfied: idna<4,>=2.5 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests->transformers) (3.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from requests->transformers) (2021.5.30)\n",
      "Requirement already satisfied: click in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from sacremoses->transformers) (8.0.1)\n",
      "Requirement already satisfied: joblib in ./miniconda3/envs/fastai/lib/python3.8/site-packages (from sacremoses->transformers) (1.0.1)\n",
      "Installing collected packages: regex, filelock, tokenizers, sacremoses, huggingface-hub, transformers\n",
      "Successfully installed filelock-3.6.0 huggingface-hub-0.5.1 regex-2022.3.15 sacremoses-0.0.49 tokenizers-0.12.1 transformers-4.18.0\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install transformers torch pandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "65a64f7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#transformers\n",
    "import transformers\n",
    "#pytorch\n",
    "import torch\n",
    "#pandas\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cc94c255",
   "metadata": {},
   "source": [
    "# Loading the model from transformers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6014e2f3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "82971f8868054d5180aa039f6ea4c33e",
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
       "model_id": "8366c660666840c3b761b15ec951db23",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/2.15G [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3ffb3565ee0b407984fbd492d625564c",
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
       "model_id": "14262903d5ce4732be040ae18ad291fc",
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
       "model_id": "c7ba14e5cb9e49269c4c3ba065171659",
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
       "model_id": "6c17db0221f04a488331bca9f278e2b7",
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
    "#importing pipeline from the transformers library\n",
    "from transformers import pipeline\n",
    "#loading the model and tokenizer\n",
    "summarizer = pipeline(\"summarization\", model=\"google/bigbird-pegasus-large-arxiv\", tokenizer = \"google/bigbird-pegasus-large-arxiv\", device = 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "143b3d0c-4567-4f48-9fcf-e4c395410afc",
   "metadata": {},
   "source": [
    "# Making a progress bar"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ac94534e-8489-47a6-bf21-841d0aa7f9c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def log_progress(sequence, every=None, size=None, name='Items'):\n",
    "    from ipywidgets import IntProgress, HTML, VBox\n",
    "    from IPython.display import display\n",
    "\n",
    "    is_iterator = False\n",
    "    if size is None:\n",
    "        try:\n",
    "            size = len(sequence)\n",
    "        except TypeError:\n",
    "            is_iterator = True\n",
    "    if size is not None:\n",
    "        if every is None:\n",
    "            if size <= 200:\n",
    "                every = 1\n",
    "            else:\n",
    "                every = int(size / 200)     # every 0.5%\n",
    "    else:\n",
    "        assert every is not None, 'sequence is iterator, set every'\n",
    "\n",
    "    if is_iterator:\n",
    "        progress = IntProgress(min=0, max=1, value=1)\n",
    "        progress.bar_style = 'info'\n",
    "    else:\n",
    "        progress = IntProgress(min=0, max=size, value=0)\n",
    "    label = HTML()\n",
    "    box = VBox(children=[label, progress])\n",
    "    display(box)\n",
    "\n",
    "    index = 0\n",
    "    try:\n",
    "        for index, record in enumerate(sequence, 1):\n",
    "            if index == 1 or index % every == 0:\n",
    "                if is_iterator:\n",
    "                    label.value = '{name}: {index} / ?'.format(\n",
    "                        name=name,\n",
    "                        index=index\n",
    "                    )\n",
    "                else:\n",
    "                    progress.value = index\n",
    "                    label.value = u'{name}: {index} / {size}'.format(\n",
    "                        name=name,\n",
    "                        index=index,\n",
    "                        size=size\n",
    "                    )\n",
    "            yield record\n",
    "    except:\n",
    "        progress.bar_style = 'danger'\n",
    "        raise\n",
    "    else:\n",
    "        progress.bar_style = 'success'\n",
    "        progress.value = index\n",
    "        label.value = \"{name}: {index}\".format(\n",
    "            name=name,\n",
    "            index=str(index or '?')\n",
    "        )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c1d4ed6e",
   "metadata": {},
   "source": [
    "# Import the dataset "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "cfced142",
   "metadata": {},
   "outputs": [],
   "source": [
    "#loading in the dataset\n",
    "dataset = pd.read_csv('meta_EA.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "af902fe1",
   "metadata": {},
   "source": [
    "# Inspecting the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8aeb7ee5-b412-48d6-9a84-625dbfa58b70",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1366"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#checking the number of rows\n",
    "dataset.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "918066de-69f4-4e13-ad04-aeac2a118c4f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#checking the number of columns\n",
    "dataset.shape[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d873464a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>doi</th>\n",
       "      <th>included</th>\n",
       "      <th>file</th>\n",
       "      <th>abstract</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>'Weakest Link' as a Cognitive Vulnerability Wi...</td>\n",
       "      <td>10.1002/smi.2571</td>\n",
       "      <td>0</td>\n",
       "      <td>article6246</td>\n",
       "      <td>introduction previous studies conducted in ma...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>\"Cerebellar Challenge\" for Older Adults: Evalu...</td>\n",
       "      <td>10.3389/fnagi.2017.00332</td>\n",
       "      <td>0</td>\n",
       "      <td>article8482</td>\n",
       "      <td>introduction the brain and body form a comple...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5-HTTLPR and BDNF Val66Met polymorphisms moder...</td>\n",
       "      <td>10.1111/j.1601-183X.2011.00715.x</td>\n",
       "      <td>0</td>\n",
       "      <td>article6376</td>\n",
       "      <td>one way susceptibility to stress may increase...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>A broader phenotype of persistence emerges fro...</td>\n",
       "      <td>10.3758/s13423-017-1402-9</td>\n",
       "      <td>0</td>\n",
       "      <td>article9165</td>\n",
       "      <td>the omission of an expected reward is known a...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>A Case Report of Compassion Focused Therapy (C...</td>\n",
       "      <td>10.1155/2018/4165434</td>\n",
       "      <td>0</td>\n",
       "      <td>article8086</td>\n",
       "      <td>introduction major depressive disorder is a c...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0  'Weakest Link' as a Cognitive Vulnerability Wi...   \n",
       "1  \"Cerebellar Challenge\" for Older Adults: Evalu...   \n",
       "2  5-HTTLPR and BDNF Val66Met polymorphisms moder...   \n",
       "3  A broader phenotype of persistence emerges fro...   \n",
       "4  A Case Report of Compassion Focused Therapy (C...   \n",
       "\n",
       "                                doi  included         file  \\\n",
       "0                  10.1002/smi.2571         0  article6246   \n",
       "1          10.3389/fnagi.2017.00332         0  article8482   \n",
       "2  10.1111/j.1601-183X.2011.00715.x         0  article6376   \n",
       "3         10.3758/s13423-017-1402-9         0  article9165   \n",
       "4              10.1155/2018/4165434         0  article8086   \n",
       "\n",
       "                                            abstract  \n",
       "0   introduction previous studies conducted in ma...  \n",
       "1   introduction the brain and body form a comple...  \n",
       "2   one way susceptibility to stress may increase...  \n",
       "3   the omission of an expected reward is known a...  \n",
       "4   introduction major depressive disorder is a c...  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#first 6 observations\n",
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b99885ea",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>doi</th>\n",
       "      <th>included</th>\n",
       "      <th>file</th>\n",
       "      <th>abstract</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1361</th>\n",
       "      <td>Women brothel workers and occupational health ...</td>\n",
       "      <td>10.1136/jech.57.10.809</td>\n",
       "      <td>0</td>\n",
       "      <td>article4717</td>\n",
       "      <td>study objectives this study examined working ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1362</th>\n",
       "      <td>Women have worse cognitive, functional, and ps...</td>\n",
       "      <td>10.1016/j.resuscitation.2018.01.036</td>\n",
       "      <td>0</td>\n",
       "      <td>article8032</td>\n",
       "      <td>introduction gender differences in cardiac ar...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1363</th>\n",
       "      <td>Work-focused cognitive-behavioural therapy and...</td>\n",
       "      <td>10.1136/oemed-2014-102700</td>\n",
       "      <td>0</td>\n",
       "      <td>article429</td>\n",
       "      <td>introduction sickness absence with mental dis...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1364</th>\n",
       "      <td>Workaholism as a Mediator between Work-Related...</td>\n",
       "      <td>10.3390/ijerph15010073</td>\n",
       "      <td>0</td>\n",
       "      <td>article8065</td>\n",
       "      <td>introduction workaholism can be defined as be...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1365</th>\n",
       "      <td>WT1 peptide vaccine in Montanide in contrast t...</td>\n",
       "      <td>10.1186/s40164-018-0093-x</td>\n",
       "      <td>0</td>\n",
       "      <td>article8822</td>\n",
       "      <td>background while the majority of adults with ...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                  title  \\\n",
       "1361  Women brothel workers and occupational health ...   \n",
       "1362  Women have worse cognitive, functional, and ps...   \n",
       "1363  Work-focused cognitive-behavioural therapy and...   \n",
       "1364  Workaholism as a Mediator between Work-Related...   \n",
       "1365  WT1 peptide vaccine in Montanide in contrast t...   \n",
       "\n",
       "                                      doi  included         file  \\\n",
       "1361               10.1136/jech.57.10.809         0  article4717   \n",
       "1362  10.1016/j.resuscitation.2018.01.036         0  article8032   \n",
       "1363            10.1136/oemed-2014-102700         0   article429   \n",
       "1364               10.3390/ijerph15010073         0  article8065   \n",
       "1365            10.1186/s40164-018-0093-x         0  article8822   \n",
       "\n",
       "                                               abstract  \n",
       "1361   study objectives this study examined working ...  \n",
       "1362   introduction gender differences in cardiac ar...  \n",
       "1363   introduction sickness absence with mental dis...  \n",
       "1364   introduction workaholism can be defined as be...  \n",
       "1365   background while the majority of adults with ...  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#last 6 observations\n",
    "dataset.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "739cc6bd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 6966 entries, 0 to 6965\n",
      "Data columns (total 5 columns):\n",
      " #   Column     Non-Null Count  Dtype \n",
      "---  ------     --------------  ----- \n",
      " 0   title      6966 non-null   object\n",
      " 1   abstract   6966 non-null   object\n",
      " 2   doi        6966 non-null   object\n",
      " 3   included   6966 non-null   int64 \n",
      " 4   full_text  6966 non-null   object\n",
      "dtypes: int64(1), object(4)\n",
      "memory usage: 272.2+ KB\n"
     ]
    }
   ],
   "source": [
    "#getting the structure of the data frame\n",
    "dataset.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "df164070",
   "metadata": {},
   "source": [
    "# Running the summarization method over the full text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f8721786",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0ebdc3fe49b54fd6aa72fac68435f8f9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(HTML(value=''), IntProgress(value=0, max=1366)))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/torch/_tensor.py:575: UserWarning: floor_divide is deprecated, and will be removed in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.\n",
      "To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor'). (Triggered internally at  /opt/conda/conda-bld/pytorch_1631630839582/work/aten/src/ATen/native/BinaryOps.cpp:467.)\n",
      "  return torch.floor_divide(self, other)\n",
      "/home/user/miniconda3/envs/fastai/lib/python3.8/site-packages/transformers/pipelines/base.py:996: UserWarning: You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset\n",
      "  warnings.warn(\n",
      "Your max_length is set to 500, but you input_length is only 332. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=166)\n",
      "Attention type 'block_sparse' is not possible if sequence_length: 332 <= num global tokens: 2 * config.block_size + min. num sliding tokens: 3 * config.block_size + config.num_random_blocks * config.block_size + additional buffer: config.num_random_blocks * config.block_size = 704 with config.block_size = 64, config.num_random_blocks = 3. Changing attention type to 'original_full'...\n",
      "Your max_length is set to 500, but you input_length is only 415. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=207)\n",
      "Your max_length is set to 500, but you input_length is only 206. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=103)\n"
     ]
    }
   ],
   "source": [
    "#making an empty list where the summaries will be stored\n",
    "summaries = []\n",
    "#creating the for loop\n",
    "for i in log_progress(dataset[\"abstract\"]):\n",
    "    summaries.append(str(summarizer(i, max_length = 500, min_length = 250, do_sample=False, truncation = True, no_repeat_ngram_size = 2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7fb6394e-30d2-409e-a74f-f1611d286e7c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[\"[{'summary_text': 'previous studies have documented a dramatic increase in the prevalence rate of depressive disorders over the past two decades an epidemiological study of depression in a sample of undergraduates suggested that the frequency rates of mild and severe depression among this population may exceed those observed in adults furthermore preliminary findings indicated that chinese undergraduates may be more likely to develop depression than their counterparts in western countries previous research has provided support for operationalizing the weakest link as a cognitive vulnerability factor for the analysis of the relationship between stress and depressive symptoms in chinese undergraduate students the current study examined the applicability of a weak link approach to assessing the impact of negative life events on the development of depressed symptoms by using a multi wave longitudinal design to measure the cognitive style questionnaire , ces d .<n> the results of this study showed that higher levels of stress levels and weaker link interactions were associated with greater increases in depressive symptom during the follow up interval in which the analyses were conducted <n> furthermore the effects of gender and cultural differences between countries appeared to vary between participants in both chinese and western samples in agreement with past research in these two countries _ keywords : depression ; etiological models ] the effect size for weakestlink interaction was found to be small however it was possible that small effect sizes should be expected when conducting studies examining the neural mechanisms of subjects with acognitive vulnerability to depression'}]\",\n",
       " \"[{'summary_text': 'a battery of simple tasks designed to probe sensorimotor cognitive performance emotional state and nonverbal reasoning is presented and evaluated the effectiveness of an internet based cerebellar challenge system zing in terms of its effectiveness compared with a life as usual control group the results support the conclusion that a systems approach to healthy aging is the most promising framework for understanding the degradation in multiple functions with age . <n> cognitive decline , cognitive enhancers 87.10.+e cognitive neuroscience and psychology unit + school of physics and astronomy + university of sheffield + hicks building + hounsfield road + shefffield s3 7rh + united kingdom + phone : ( + 44 ) 3177 3983 + e - mail: g.h.wells@sheffields.ac.uk + web :1 ( http://www. cognitive- neuroscience.org/) ] the cerebellum represents a significant component in the functional networks involving other brain and body structures and is central in adapting to internal changes the design of the intervention is discussed on theoretical grounds and on the basis of recent evidence regarding the beneficial effects of computerized cognitive training approaches using computer programs to boost core cognitive capabilities such as balance and fluid thinking a recent meta analysis for mci patients concluded that the interventions were effective in reducing the rate of decline in cognitive and affective performance and in improving the quality of daily life functioning'}]\",\n",
       " \"[{'summary_text': 'a growing body of research indicates that ruminative thinking helps them understand and solve problems a vulnerability that is associated with negative mood rumination one way to stress may increase vulnerability to the negative effects of stress is through intermediary phenotypes or the tendency to perseverate on problems rumination represents an important cognitive vulnerability for cognitive and behavioral efforts to improve mood recent studies have shown that the httlpr polymorphism moderates the effect of current life stress on rumination variation in a gene regulating brain derived neruotropic factor bdnf met polymorphisms derived neurotrophic factor is a protein involved in neuronal and synaptic development the relationship between current stress and rumination remains unclear . in order to investigate this hypothesis we genotyped four alleles val and met in two sets of individuals : one group included individuals with two s allelics and the other group had at least one copy of the high expressing long allle we performed a multiple regression analysis with rrs scores as the predictor variable in an attempt to assess the association between rumination and current adverse events we also used a measure of depression severity as a covariate to test for variability in the prediction of rumination across genes and to examine the causal nature of these relationships across genetic profiles we found that individual differences in val met genotypes were correlated with rumination but not directly related to aversive responses across the life span in agreement with previous studies that showed that genetic effects mediated by the interaction between genes were responsible for the development of negative events and positive events in response to depression and rum depression , and negative depression in depression while negative adverse event ; this result resulted in these results were consistent with the number of a higher levels of mortality and a significant differences between the incidence of both t and bdi ii and variability and individual and c and that  bd and their number <n> b - b th and these bp np ) lt and they were significant to b. b bi bi b b we were a number b is the individual  b to _ b with a b but b that was the most b and number number the b the group b number a   is not b was b a group  but the co co - the subgroup b group and group that had b by a high number and individuals that were the groups  the met b groups and both met met  while b c b most  and most individuals met and was a most most groups that also the bi and groups from the individuals from '}]\",\n",
       " \"[{'summary_text': 'it has been known for some time that individuals exhibiting frustration from lack of reward incentive downshifts or extinction display dramatically different patterns of behavior in response to these shifts in reward a series of studies showed that during extinction some human infants exhibit persistence accompanied by facial displays of anger whereas others quickly stop responding and exhibit expressions of sadness these distinct individual differences were consistent from four to at least months of age and suggest the possibility of differences in a broader phenotype indeed work by others suggests that children who express depression in reaction to challenging situations often exhibit an absence of control low self worth and a disposition for learned depression all of which are associated with extinction the present results suggest that the continuum in persistence emerges independently of motivation or arousal the effect of extinction might be related to broader phenotypic differences and may have important implications for such phenomenon as addiction relapse resilience under stress and susceptibility to depression here we report on the results of a study in which we measured the rates of acquisition , spatial learning and reacquisition of the learned response during acquisition and extinction in persistent mice that were fed ad libitum during the initial acquisition phase and were allowed to freely explore a long alley or open field with no other movement beyond the occasional alternate movements of paws and tail necessary to keep head nose above water the mice were deprived of food by giving them only min of access to food daily during all phases the forced swim test is commonly used to assess the propensity for depression and is a common screening method for the efficacy of putative antidepressants in humans in the case of depression .'}]\",\n",
       " \"[{'summary_text': 'the case report was approved by the ethical committee of the graduate school of medicine chiba university hospital case presentation case lisa was a female aged in her s adolescence<n> she was referred from her primary doctor to our cbt center at chi ba university hospitalised major complaint her major complaints was depression and depressed mood i.e. he overdosed on sleeping pill and i am so frustrated with myself and my mother i can t work well because i have not done anything to help my depression i do nt think that i ll ever work again i just have to keep working and keep trying to do better .<n> the main strategy in treatment of depressed patients is to introduce a framework of compassion and compassion based therapy ( cft ) in addition to psycho education about the importance of maintaining a positive attitude toward oneself and others and how to develop a good relationship with each other and with the medical treatment center because it is important for the patient to be well and happy and healthy and to have a happy life because depression is a serious problem all over the world and is classified as a mental health problem more than two million people in the us are diagnosed with depression every year and this rate is twice that of anxiety disorders so this problem is natural that mdd is regarded as social behavioural therapy is recommended by major treatment guidelines , however there are no reports of japanese patients with high self criticism and shame in japanese culture has been called a shame culture by cultural anthropologists gilbert pointed out that it can be a key role in this is an important role for developing the development of empathy and that the japanese self self and the patients in a patient and it was the most important to use the key part of a part to the first part in human self that was also the reason to create the self in compassion for others to reinforce the treatment and was one of other self which was not the client and also to make the other to others in our japanese and we developed the previous and our first and they were the strongest and most most in all the fifth session and were also we were a practical and one and a first to all to a group and those and all and in which the second session of all of and other  compassionate and these we we also in cf cfs and ng and while the fourth session in and so we was consistent and only the local group of _ first the ; cf and c and first was to we and and cf <n> we to  the four of which we in '}]\",\n",
       " \"[{'summary_text': '* purpose * : cancer patients undergoing surgery for the first time may require surgical support while being limited by other medical conditions and may have difficulty adjusting to daily life while they are in the hospital .<n> * methods *: a longitudinal one group design program for cancer survivors with surgically created stomas ( ostomies ) was designed to improve their self efficacy and hrqol in chronic care settings using a framework based on patient activation the development of a comprehensive patient education and education curriculum was the initial design phase of the program and the design included pre intervention and month follow up survey evaluations to evaluate effectiveness and participant satisfaction the study was conducted in tucson , az over a period of five sessions in which sixty eight participants participated in each session the efficacy of this program was evaluated using the consumer satisfaction survey ghaa ( http://www.ghas.org/) a self assessment instrument for detecting cronbach s alpha was used to measure the participants response to the curriculum and to identify potential barriers to achieving improvement in health and well being the same instrument was also used for both the pre and post intervention sessions to determine the most beneficial and problematic sessions for participants in both groups of randomized control and for those who did not complete the sessions the results indicated that participants improved their quality of life and appeared to be able to experience positive changes in their physical and social wellbeing while significantly improved patient efficacy for patients with self management strategies and improvement of alpha for patient management and alpha management for alphas for hr management in patients and care and patient alpha in hr qol o alpha and in alpha by the patient q and by alpha <n> the coh o and co - alpha scores were significantly improvement by patients by participants by both alpha alpha were significant by patient and a number of patients were the number by number for their hr scores by a second by co and their number and number in a total scores for self and average scores and outcomes by by mean scores at the mean and rate for number score scores  and mean score and that was significant for rate rate and as the total number scores while the rate score for a rate was significantly by rate by average and and  alpha score was higher for average number number as number was number ; number the second number is the group and was only the average for johnson and two by as a mean number to rate of number at least number alpha number a group  the  to number that rate is number rate number which was at rate as rate the the'}]\",\n",
       " \"[{'summary_text': 'we report the results of a community survey conducted in informal settlements in the peru city of lima to estimate the prevalence of noncommunicable diseases and related risk factors .<n> evidence was found to support the conclusion that informal settlement populations were significantly more likely than the overall population to have been diagnosed with a disease or to be suspected of having an associated risk factor due to their informality and lack of infrastructure for providing epidemiologic and other public health services in a comprehensive and equitable way , especially for those living in low income regions of developing countries where high rates of violence and inequality are prevalent _<n> i.e._intermittency to seek medical care and access to public services are among the main factors contributing to increasing prevalence and risk for a number of diseases including diabetes : diabetes is the third leading cause of death from cardiovascular disease in developed countries @xcite ; more than half of all deaths from stroke are associated with vascular disease and its prevalence is higher among women and young adults than among high income individuals the rate of stroke is highest among young and middle age individuals 1>15 - 24 years of age and among those aged between 15 and 25 years there was a correlation between obesity and diabetes prevalence in both urban settlements and low - income populations <n> this finding is consistent with an increase in obesity prevalence among older age population compared to the age of population older than 35 years and the median age for these age groups were between 35 - 35 and 35 to 55 years for the last age between 65 years ) s age age in th age with age ages ages age more age among age compared with the average age number ages of adult age  ages among ages in ageages ages and age aged age at least half age when disproportionate age as most ageage age than age rate in an age ratio of the number number age were most adult ages number in most ages  age that age over age ( age or age to age gender and number among  overweight and adult number and  adult  in adult gender age is age are age was more ages were more more number with most  of ages with adult and most most of youth age which age regardless age by the most adults in population ages that were the population age duration of multiple age averaged age while most number that was in number for age during the majority of those of weight and weight in their number is more weight among adult population in multiple weight weight of adults and duration in high age most weight that most'}]\",\n",
       " \"[{'summary_text': 'the relationship between ed depression and anxiety symptoms was examined using network analysis methods in a clinical sample of individuals with eds . changes in global network strength over the course of treatment<n> were assessed and assessed whether these changes corresponded to decreases in symptom severity , and whether such changes were associated with a change in network structure among participants with poorer outcomes rm manova analyses indicated that the degree to which depression exhibited a decrease in treatment outcome was higher among those who evidenced a greater reduction in connectivity within the network and that there was a significant effect of time across all measures reflecting the evolution of network structures the present results suggest that interventions targeting central symptoms may be more effective in reducing the severity of depression than targeting high frequency affective symptoms which in turn may lead to more efficient interventions and more precise understanding of underlying network dynamics which allow for the development of tailored interventions which may enhance the impact of research into the complex relationships between symptoms in the context of mental health and substance abuse disorder it is noted that these results are consistent with findings of van borkuko et al and others that implicating negative affect as a salient etiological and maintaining factor in eds furthermore results indicate that fatigue is a marker of resourcedepletion that limits capacity for self regulation and could be a risk for binge eating behavior'}]\",\n",
       " \"[{'summary_text': 'the incidence of depression after stroke or myocardial infarction has been suggested to be related to the underlying vascular pathogenesis being associated with white matter hyperintensities and multiple silent cerebral infarctions on magnetic resonance imaging in addition non specific mechanisms such as personality traits and lack of social support may also be involved in the pathogenesis of post stroke depression . in this study<n> we compared the one year cumulative incidence rates of depressive episodes in patients with a diagnosis of first ever hemispheric cerebral infarct and in comparison with the same number of non depressed patients after controlling for age sex and level of handicap in order to test for a specific relation between the two diseases mechanisms and to investigate the impact of vascular risk factors on the nature of the pathogen that causes depression in stroke patients<n> the results do not favour the hypothesis that specific neurobiological mechanisms play a major role with regard to depression pathogenesis in both conditions however we can not rule out the possibility that generalised brain damage may affect mood regulatory processes the so called vascular hypothesis as has recently been put forward by krishnan and coworkers this more general vascular mechanism may reflect the complex interacting effects of physical illness with psychological and social functioning in physically ill patients the interaction term for vascular disease with depression may be a way to account for the high prevalence of both stroke and heart attack in elderly patients'}]\",\n",
       " \"[{'summary_text': 'the search for a unique genetic biomarker of depression has not been clearly success as yet with recent mega analysis of single nucleotide polymorphisms failing to identify robust and replicable findings for any specific genetic factor instead a considerable body of research has focussed upon the interaction of genes and environment via the serotonin transporter short form of the httlpr plus the individual ss response to various stresses .<n> recent studies have shown that the short forms of both the genotype and the environmental stress can be associated with clinically significant depression and that psychological resilience has been identified as a buffer against depression by an active physiological process that reduces autonomic responses to extreme stress and promotes adaptation in adverse circumstances and recovery from trauma resilience was found to follow a distinct trajectory to recovery when individuals experience a traumatic event the frequency of reported negative childhood events increased with age that relationship was not significant at the bonferroni corrected level of p t for total zsds scores and was indicative of a more complete model between genotype environmental events and depression than was previously reported by a number of explanations including the presence of several genetic factors and mdd significantly lowers patients quality of life and has sensitivity of depressed in predicting depression in patients who may present with milder symptoms of parkinson disease and child abuse care giving to alzheimer stroke and hip fracture fracture in recent medical conditions'}]\"]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#checking the some beginning observations\n",
    "summaries[0:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "0cce9501-11f4-484f-90d9-1d99119d7e0f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['[{\\'summary_text\\': \"aging with a considerable increase in both the number and the duration of healthy life years is a priority up to older adults have worry sleep disturbances and cognitive decline deteriorate well being and increase the risk of dementia and more generally in improved quality of life of aging populations and in particular in the elderly their family members and society at large mental health and emotional regulation through meditation practice might help reduce these adverse psycho affective factors this training might in turn result in reduced risk or delayed onset of the disease or even prevented altogether . here<n> we define meditation as a broad variety of practices including complex emotional regulatory strategies in various context and regulation in brain and brain function especially in elderly meditula and most interesting in this context is the positive effect on aging from various regulation mechanisms including sympathetic mind control and emotion regulation ability which is particularly important for hippocampus and neuroticism and anxiety are associated with an increased cumulative incidence and risk for dementia risk by approximately stress has a detrimental impact on brain structure and function and also subclinical conditions such as stress insomnia worry insomnia and depression worry anxiety but also stress reduction and social exclusion concerns the impact of meditation on behavioral and blood measures will be demonstrated in our lab we conducted a pilot study comparing intervention using large samples to confirm that attention and attention training in three arms and frontal regions was randomized before and after intervention in six older adult expert buddhist s cohort compared after randomized intervention versus controls in gray matter volume and neuroimaging and fluorodesoxyglucose metabolism was also randomized to delay onset or brain metabolism in a randomized group versus control in longitudinal studies using a group of gray neuroimaging intervention on gray brain activity in , posterior tempo tempotal and tempo and cortical and posterior cortex and prefrontal region in frontal region and hippocampus or posterior ponsula or tempo ; ve posterior posterior region was the strongest in these interventions were randomized in most randomized after a proportion to be randomized neuroimaging in randomized for the frontal frontal neuroimaging interventions after gray cortical neuroimaging after the randomized interventions in neuroimaging neuroimaging to the gray tempo or frontal cortex in white frontal tempo posterior cortical or neuroimaging  partal region or prefrontal cortex or fm and limb tempo ( yama _ t vata \\' ) ie : th ] ( a month and a jun <n> rong  ( we will help to have randomized randomized  to reduce gray and peripheral neuroimaging as randomized by the intervention to a previous randomized and we showed the effects in cognitive interventions to  neuroimaging for randomized the main neuroimaging from the longitudinal interventions for gray \"}]',\n",
       " \"[{'summary_text': 'randomized controlled trials are the most widely used means to determine the effectiveness of an intervention in clinical practice therefore it is important that research participants reflect the population they represent therefore care research has a critical role in the recruitment of participants when decision making is rarely straightforward gps decisions when cbt decisions regarding patient eligibility <n> often gps may adopt a protective role by viewing the research process as a benefit patients may experience even when in trial control arm and may deem some patients unsuitable which may be the case for patients with life threatening comorbidities .<n> this article aims to address the following questions : a ) what role should be played by patients who are currently excluded from trial sites ? b ] how should gps inform all eligible patients regarding the results of randomized participants , especially if patients are not recruited if they are selected for a randomization stage which is not explicitly reported in a recent review of the quality of service for primary care in bristol and glasgow and if gps confirm the external validity of a study _<n> ab initio_. <n> what is the best way to include excluded patients into the bb84 trial design process and to support their decision regarding participation in this and other research trials where gps have to choose between treating comorbid conditions or treating chronic disorders without being informed about the presence of multiple mental disorders and being treated by a treatment centre or receiving treatment for bipolar disorder is currently identified by treating patients without treatment but being able to account for unspecific data on the proportion of unscreened data in b is a quarter of b and psw number for the remaining data without a fraction for b whereas a single age for each b - a number of two years without p - j - p s without b in both p and a p for both b without chis j and b for p j tt a j in p p in these data ; p to be p. a second section of p a c - b to p ( p is p while p we were in th p with a complete data for scotland scottish data with p number number to j j for two p if pt rs pf a list for all p without the number tp a first p b p which p by p that was p most p c scott b number b. p was not p only p after a total number a most c and c p r os b if a pair of c c. c in j. b j p'}]\",\n",
       " \"[{'summary_text': 'we report the results of a study combining the methodology for the assessment of gynaecological disorders in a large sample of indian women .<n> the analysis of the data suggests that the prevalence of sexually transmitted diseases is a main factor affecting the outcome of such studies and the quality of care provided to the women in this setting the study was conducted in india , southia and taiwan in the period from november 2000 to july 2001 <n> it has been hypothesized that complaint of bowel syndrome may represent wider issues of social stress in which concerns about loss of bladder are addressed in addition to their focus on the community studies linking the discharge of women with a high proportion of syndromic shaped bodily expressions and their response to medical interventions for their reproductive health the report of this study is the first step towards a systematic approach to assess the impact of sexual transmitted disorders and other diagnosis methods in general care settings for women and in particular for high - risk cases of chlamydia and gonorrhombic infections where there is substantial literature from west literature demonstrating a strong correlation between grti and depression in primarycare settings there<n> is evidence of both psychological and economic concerns regarding the use of private health care for these patients in westia ranged from poor and affordable care in mumbai and northia to rich families and associations with the population of southie as suffering from more than 10 types of mental disorders including trichosiasis and se tis syndrome which are characterized by unexplained bowel infections and common pathogens like syphilis and ceti ; these are found to be the number for a fraction of two years in goa and have been used for medical diagnosis in medical tests for gtis and a range in ctis tii and shane and had been the family and independent scores for each had the age in these classification in gti : the range for any gender and we were used as a number in any number with her gender - a family with any previous scores in many years se and she is found with pse ] with more _ cii levi ) with two - the separation in an independent score for two and were the time with an a more with one - p with many th and more in each the most most tp and while the birth rate for more - number which was the distribution for independent with four - which were not the identity for one with each with independent  while a two with most number and that were most with  the union  with other with which is known with both'}]\",\n",
       " \"[{'summary_text': 'there is growing evidence that positive mental health over time functions as a resilience resource and protects against both physical and mental illness and disease even further amplifies its significance for high levels of positivemental health are associated with heightened recovery and survival despite physicalillness and cortisol levels and cardiovascular disease risk we need to know more of the working mechanisms by which positive mentally health enacts its potential as mediate the relationship between scs and psychopathology . in the present study , we conducted a pilot study to assess the effect of indirect compassion interventions on the development of both high and low level of compassion in addition to indirect sfs scores and hadronization strategies with a focus on high functioning groups and individuals with mental disorders we found no significant difference between the best and worst outcomes of these randomized control studies in which we compared direct and indirect interventions with the state of art in treatment and prevention of mental and physical illness in more than half of our participants we also conducted an analysis to compare indirect and direct interventions and found significant differences between these models in terms of how effectively we were able to mitigate the effects of chronic disease and how well we succeeded in recovery even for a significant subset of participants <n> + + * keywords : * _ psychological well - being ; adaptive functioning ]'}]\",\n",
       " \"[{'summary_text': 'the role of depression in the context of widowhood remains unclear largely owing to two factors first a focus on cross sections rather than longitudinal data and second that use separation into years after death of one s spouse and remarriage or separation for children more likely to occur at higher loss for physical well being increased health outcomes including decreased physical and mental health .<n> analysis of data in a variety of countries with two main lines of first comparison finds that women have higher levels of somatic components compared to each other and recover to levels on par with single age populations compared with korean counterparts while men do not explore whether distinguishing somatic scores into somatic mood components allows us to distinguish gender transitions across international contexts we find that widows in china may be more affected than those in united states while it is likely that these transitions are related to depression and men have more access to care in england and south china than in europe and france where most research has focused on long period transitions for re marriage and children who remain attached to england , north china and taiwan where many people have long periods of silence and are easily accessible to treatment and care for those who have lost someone close to death or are experiencing transitions to lower quality of life in many countries where we use the link for depression over time and context to determine the effect of long duration for widows and others who are not directly connected to remarry loss ces for each year of transition hausman and more than fixed numbers for united state scores with lower baseline scores for the respective age and those were used for a single population in france and china ; ypm scores are used to be used on the age scores in south american samples and united united : d scores at year b and most age scale for all age levels for most years in world d and the number of united age in most population cle th age com b in all united c scores and r r mean mean the most most time with the time in american state in age for both d for many in u and we are more more with most b we have most r and u are the population for b t and a more to have the mean for these are a subset for any more most d in those are most american and b for people who were the more in y are american c c are also most are j and c and were more for more b that were most more _ d  most mean d c in d r b c ] d we were a most in b. d'}]\",\n",
       " \"[{'summary_text': '* abstract * a mother s ability to cope with the outcome of learning the result of her genetic test for a brca mutation can depend on her cognitive state and the resources available to her in the face of anticipated distress following genetic testing and subsequent impacts on psychological outcome .<n> the process of predicting how one will feel following future events has been shown to have powerful effects on health decision making despite the influence that affective forecasting can have on major life choices individuals can overestimate the strength of this reaction which can contribute to information processing biases the association between psychological and behavioral outcomes can lead individuals to make decisions based on expectations of how they will react emotionally to learning their test result even for those who anticipated negative or uninformative results recent work demonstrates that familial cancer risk information for the benefit of families and their offspring can be disclosed to their minor aged children in an effort to reduce negative anticipation among those testing positive for their carrier status and subsequently experiencing poorer outcomes across a range of medical decisions from disease prevention to end of life care * keywords : * genetic screening for hereditary breast and ovarian cancer @xmath0division of cancer and preventive medicine , los alamos national laboratory + 6100 jefferson ave + new mexico 87131 + department of health and human sciences + massachusetts institute of technology + amherst ma 01003 + division of radiation science and biomedical engineering + national institute for medical research and biostatistics + university of california at san berkeley + san francisco + california 94106 - 95 - 2575 - 97 - 94 - 1 - 99 - 98 - 00 - 000 - 0105'}]\",\n",
       " \"[{'summary_text': 'this exploratory study outlines the nature of working conditions in the sex industry in israel .<n> results indicate that women working in a high morbidity rate t he industry have access to adequate health care when they report a step of discomfort in other pcd infections<n> furthermore , the majority of women in this industry had no trauma in their past attempts to escape from the exploitative environment and no medical care was available for them if they had interacted with other workers and had suffered harassment or violence in connection with their working environment a third of respondents had worked in an environment where they were threatened with violence or had been the subject of physical or sexual abuse while a fourth had never had to deal with any kind of violence and one third were exposed to violence while working at least once per week at an office or at a train station a quarter of the women reported being forced into the industry by some organization who threatened their health and well being one quarter were forced to engage in forced forced sex work and the other quarter worked for less than a week with little or no paid sick leave a significant fraction of workers reported having worked more than one hour per day and many had multiple jobs over long periods of time without any paid days off<n> this study indicates the existence of a correlation between the number of registered sex workers per capita and an index of occupational health risk for s forced sexual industry workers a factor of two higher than any other epidemiological measure including the ratio of hiv versus std _ p_.'}]\",\n",
       " \"[{'summary_text': 'when compared to men post stroke women are typically older presenting with a higher rate of cognitive functional deficits and experience higher in hospital mortality even after adjusting for baseline differences in demographics and clinical variables there is paucity of data looking at these detailed outcomes after ca which has been limited to crude scales such as the cerebral performance category scale @xmath0 this study aims to examine gender differences at hospital discharge after cardiac arrest using indepthcognitive functional and psychiatric outcomes methods .<n> ca gender specific differences between m psms and ci p value were evaluated via the lawton physical self maintenance cpc ces d scores for a prospective discharge analysis columbia university s institutional review board data collection following a protocol for targeted temperature management crabant claile b submitted to _<n> j.phys._a doi : 10.1016/j.jph.2015.11.007 * keywords * asystole * ttm administration , discharge cognitive status rbans immediate memory semantic awareness and delayed memory outcome methods are attached in a supplementary file level of care provided by the simplified therapeutic intervention scoring system outcome measures were administered by a single reviewer who is a board certified neuropsychologist and is trained by an interdisciplinary team for the assessment of neuropsychological status'}]\",\n",
       " \"[{'summary_text': 'the integrated work and sickness programme ( awac ) aims to integrate people with common mental disorders into the labour market .<n> the aim of this study is to evaluate the impact of the integration process on work participation and quality of life at months follow up for a labelled integrated model with a main focus on people who have suffered from common disorders as a consequence of their long absence over the past decade and more than six times the sixth working age population at any age group we found no significant differences in the employment outcomes of ips participants and long term participants compared to control participants who did not have access to a wac the main objective was to assess the effectiveness of an integrated approach for the treatment and recovery of common mentally disorders in a context where these disorders are a major cause of loss of labour supply and increase in state expenditure through sickness absence and disability the results indicate that the case study was conducted in norway , as well as in other european countries and the united states of america with data collected at or close to the end of last year <n> + + * keywords : * integratedwork and illness prevention ; longterm unemployment insurance ( noi)*. +<n> * pacs nos:* 87.23.ge;87.19.hh'}]\",\n",
       " '[{\\'summary_text\\': \"the relationship between work and emotional well - being has long been a subject of debate and research .<n> recent studies have shown that there are significant benefits to be found in the fields of psychology and social science , especially for those working in low - paid jobs where the cost of unemployment can be quite high the number of employed individuals has increased dramatically in recent years @xcite the purpose of this study is to investigate if and to what extent work affects the psychological state of the employees and the quality of services provided to them by the employment services industry through the use of a model of workaholism and its influence on work performance and satisfaction at a large technology andeccoecco basedecco firm with a focus on reducing the unemployment rate in urban environments <n> the model is designed to have a strong work - related component and a weak emotional component _<n> i.e._the more work a worker does the greater the impact she has on herself and on the organization she s  good feeling \\'\\' and thus the less likely she will suffer from depression or other mental health issues such as substance abuse or unemployment pacs numbers : 07.05.mh ; 89.90.+n + e-mail:kentslepian@ehu.edu + http://arxiv.org/abs/cond-neg/papers.html\"}]',\n",
       " \"[{'summary_text': '* abstract * : residual leukemia remains a significant obstacle to the development of new targeted therapies for cancer vaccination because it is mediated by the inhibition of wt gene expression in the majority of patients with minimal residual disease while intensive induction chemotherapy is difficult to tolerate with comorbidities and impaired performance status in clinical trials .<n> we report the results of a pilot study in which a novel strategy for pursuit in cd <n> ic leukemia with a vaccination strategy based on an ifn receptor agonist has been developed that is capable of recognizing laas after presentation on class i mhc molecules and subsequently elimination of malignant cells bearing these antigens can increase frequencies with which antigen specific effector and memory t cells are generated to enhance the effectiveness of vaccination with peptide vaccination for the treatment of adult acute leukemia younger than years of age while avoiding cytotoxic deep cycle chemotherapy with cytarabine based allogeneic stem cell consolidation each option carries significant morbidity and may be difficult or impossible totol in poly sloan digital sky survey ( psdss ) or other cancer organizations with limited financial support from the national research institute for leukemia and sickle cell disease ( nrimd)the work of the authors is supported in part by grant - in - aid for scientific research ( nims - dfs-00 - 0087 ]'}]\"]"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#checking the some ending observations\n",
    "summaries[1355:1366]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2aa37ec-1774-4483-8c68-2b988605a495",
   "metadata": {},
   "source": [
    "# Putting the enhanced abstracts in the data frame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "cc44793b-f509-485b-8c3f-7f320884982e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#putting the summaries in the data frame\n",
    "dataset[\"abstract\"] = summaries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "aa3fe4c3-64f5-4bb5-925e-6605e8421106",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>doi</th>\n",
       "      <th>included</th>\n",
       "      <th>file</th>\n",
       "      <th>abstract</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>'Weakest Link' as a Cognitive Vulnerability Wi...</td>\n",
       "      <td>10.1002/smi.2571</td>\n",
       "      <td>0</td>\n",
       "      <td>article6246</td>\n",
       "      <td>[{'summary_text': 'previous studies have docum...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>\"Cerebellar Challenge\" for Older Adults: Evalu...</td>\n",
       "      <td>10.3389/fnagi.2017.00332</td>\n",
       "      <td>0</td>\n",
       "      <td>article8482</td>\n",
       "      <td>[{'summary_text': 'a battery of simple tasks d...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5-HTTLPR and BDNF Val66Met polymorphisms moder...</td>\n",
       "      <td>10.1111/j.1601-183X.2011.00715.x</td>\n",
       "      <td>0</td>\n",
       "      <td>article6376</td>\n",
       "      <td>[{'summary_text': 'a growing body of research ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>A broader phenotype of persistence emerges fro...</td>\n",
       "      <td>10.3758/s13423-017-1402-9</td>\n",
       "      <td>0</td>\n",
       "      <td>article9165</td>\n",
       "      <td>[{'summary_text': 'it has been known for some ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>A Case Report of Compassion Focused Therapy (C...</td>\n",
       "      <td>10.1155/2018/4165434</td>\n",
       "      <td>0</td>\n",
       "      <td>article8086</td>\n",
       "      <td>[{'summary_text': 'the case report was approve...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0  'Weakest Link' as a Cognitive Vulnerability Wi...   \n",
       "1  \"Cerebellar Challenge\" for Older Adults: Evalu...   \n",
       "2  5-HTTLPR and BDNF Val66Met polymorphisms moder...   \n",
       "3  A broader phenotype of persistence emerges fro...   \n",
       "4  A Case Report of Compassion Focused Therapy (C...   \n",
       "\n",
       "                                doi  included         file  \\\n",
       "0                  10.1002/smi.2571         0  article6246   \n",
       "1          10.3389/fnagi.2017.00332         0  article8482   \n",
       "2  10.1111/j.1601-183X.2011.00715.x         0  article6376   \n",
       "3         10.3758/s13423-017-1402-9         0  article9165   \n",
       "4              10.1155/2018/4165434         0  article8086   \n",
       "\n",
       "                                            abstract  \n",
       "0  [{'summary_text': 'previous studies have docum...  \n",
       "1  [{'summary_text': 'a battery of simple tasks d...  \n",
       "2  [{'summary_text': 'a growing body of research ...  \n",
       "3  [{'summary_text': 'it has been known for some ...  \n",
       "4  [{'summary_text': 'the case report was approve...  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#checking the first 6 observations\n",
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "271c948c-9101-47ed-a814-9eb2ad711f96",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>doi</th>\n",
       "      <th>included</th>\n",
       "      <th>file</th>\n",
       "      <th>abstract</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1361</th>\n",
       "      <td>Women brothel workers and occupational health ...</td>\n",
       "      <td>10.1136/jech.57.10.809</td>\n",
       "      <td>0</td>\n",
       "      <td>article4717</td>\n",
       "      <td>[{'summary_text': 'this exploratory study outl...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1362</th>\n",
       "      <td>Women have worse cognitive, functional, and ps...</td>\n",
       "      <td>10.1016/j.resuscitation.2018.01.036</td>\n",
       "      <td>0</td>\n",
       "      <td>article8032</td>\n",
       "      <td>[{'summary_text': 'when compared to men post s...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1363</th>\n",
       "      <td>Work-focused cognitive-behavioural therapy and...</td>\n",
       "      <td>10.1136/oemed-2014-102700</td>\n",
       "      <td>0</td>\n",
       "      <td>article429</td>\n",
       "      <td>[{'summary_text': 'the integrated work and sic...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1364</th>\n",
       "      <td>Workaholism as a Mediator between Work-Related...</td>\n",
       "      <td>10.3390/ijerph15010073</td>\n",
       "      <td>0</td>\n",
       "      <td>article8065</td>\n",
       "      <td>[{'summary_text': \"the relationship between wo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1365</th>\n",
       "      <td>WT1 peptide vaccine in Montanide in contrast t...</td>\n",
       "      <td>10.1186/s40164-018-0093-x</td>\n",
       "      <td>0</td>\n",
       "      <td>article8822</td>\n",
       "      <td>[{'summary_text': '* abstract * : residual leu...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                  title  \\\n",
       "1361  Women brothel workers and occupational health ...   \n",
       "1362  Women have worse cognitive, functional, and ps...   \n",
       "1363  Work-focused cognitive-behavioural therapy and...   \n",
       "1364  Workaholism as a Mediator between Work-Related...   \n",
       "1365  WT1 peptide vaccine in Montanide in contrast t...   \n",
       "\n",
       "                                      doi  included         file  \\\n",
       "1361               10.1136/jech.57.10.809         0  article4717   \n",
       "1362  10.1016/j.resuscitation.2018.01.036         0  article8032   \n",
       "1363            10.1136/oemed-2014-102700         0   article429   \n",
       "1364               10.3390/ijerph15010073         0  article8065   \n",
       "1365            10.1186/s40164-018-0093-x         0  article8822   \n",
       "\n",
       "                                               abstract  \n",
       "1361  [{'summary_text': 'this exploratory study outl...  \n",
       "1362  [{'summary_text': 'when compared to men post s...  \n",
       "1363  [{'summary_text': 'the integrated work and sic...  \n",
       "1364  [{'summary_text': \"the relationship between wo...  \n",
       "1365  [{'summary_text': '* abstract * : residual leu...  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#checking the last 6 observations\n",
    "dataset.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "36e6d761-27d8-4094-abfe-bfb99419491c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#removing the full text column\n",
    "dataset_final = dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "27e52e7b-0b6b-4c79-a74f-c0c345a1d3ef",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>doi</th>\n",
       "      <th>included</th>\n",
       "      <th>file</th>\n",
       "      <th>abstract</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>'Weakest Link' as a Cognitive Vulnerability Wi...</td>\n",
       "      <td>10.1002/smi.2571</td>\n",
       "      <td>0</td>\n",
       "      <td>article6246</td>\n",
       "      <td>[{'summary_text': 'previous studies have docum...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>\"Cerebellar Challenge\" for Older Adults: Evalu...</td>\n",
       "      <td>10.3389/fnagi.2017.00332</td>\n",
       "      <td>0</td>\n",
       "      <td>article8482</td>\n",
       "      <td>[{'summary_text': 'a battery of simple tasks d...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5-HTTLPR and BDNF Val66Met polymorphisms moder...</td>\n",
       "      <td>10.1111/j.1601-183X.2011.00715.x</td>\n",
       "      <td>0</td>\n",
       "      <td>article6376</td>\n",
       "      <td>[{'summary_text': 'a growing body of research ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>A broader phenotype of persistence emerges fro...</td>\n",
       "      <td>10.3758/s13423-017-1402-9</td>\n",
       "      <td>0</td>\n",
       "      <td>article9165</td>\n",
       "      <td>[{'summary_text': 'it has been known for some ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>A Case Report of Compassion Focused Therapy (C...</td>\n",
       "      <td>10.1155/2018/4165434</td>\n",
       "      <td>0</td>\n",
       "      <td>article8086</td>\n",
       "      <td>[{'summary_text': 'the case report was approve...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0  'Weakest Link' as a Cognitive Vulnerability Wi...   \n",
       "1  \"Cerebellar Challenge\" for Older Adults: Evalu...   \n",
       "2  5-HTTLPR and BDNF Val66Met polymorphisms moder...   \n",
       "3  A broader phenotype of persistence emerges fro...   \n",
       "4  A Case Report of Compassion Focused Therapy (C...   \n",
       "\n",
       "                                doi  included         file  \\\n",
       "0                  10.1002/smi.2571         0  article6246   \n",
       "1          10.3389/fnagi.2017.00332         0  article8482   \n",
       "2  10.1111/j.1601-183X.2011.00715.x         0  article6376   \n",
       "3         10.3758/s13423-017-1402-9         0  article9165   \n",
       "4              10.1155/2018/4165434         0  article8086   \n",
       "\n",
       "                                            abstract  \n",
       "0  [{'summary_text': 'previous studies have docum...  \n",
       "1  [{'summary_text': 'a battery of simple tasks d...  \n",
       "2  [{'summary_text': 'a growing body of research ...  \n",
       "3  [{'summary_text': 'it has been known for some ...  \n",
       "4  [{'summary_text': 'the case report was approve...  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#double checking to make sure it worked\n",
    "dataset_final.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "94ed5a96-9077-493e-af20-4918cb15827d",
   "metadata": {},
   "source": [
    "# Exporting the final dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "b35241fc-c6c9-4a1a-84ae-208e7f7e94dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset_final.to_csv(\"meta_EA.csv\")"
   ]
  }
 ],
 "metadata": {
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
 "nbformat_minor": 5
}
