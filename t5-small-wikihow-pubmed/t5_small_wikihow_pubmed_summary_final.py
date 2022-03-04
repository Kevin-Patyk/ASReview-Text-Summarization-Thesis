# -*- coding: utf-8 -*-
"""t5-small-wikihow-pubmed-summary-final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10yHzCTOGAxD0Vgpbfv2KnqKOYUzg2C10

If you're opening this Notebook on colab, you will probably need to install 🤗 `Transformers` and 🤗 `Datasets` as well as other dependencies. 

* `datasets`
* `transformers`
* `rogue-score`
* `nltk`
* `pytorch`
* `ipywidgets`

*Note*: Since we are using the GPU to optimize the performance of the deep learning algorithms, `CUDA` needs to be installed on the device.
"""

! pip install datasets transformers rouge-score nltk ipywidgets

"""When using `nltk`, `punkt` also needs to be installed. I guess it is not installed automatically. Not having `punkt` will result in an error during the analysis."""

import nltk
nltk.download('punkt')

"""If you're opening this notebook locally, make sure your environment has an install from the last version of those libraries.

To be able to share your model with the community and generate results like the one shown in the picture below via the inference API, there are a few more steps to follow.

First you have to store your authentication token from the Hugging Face website (sign up [here](https://huggingface.co/join) if you haven't already!) then execute the following cell and input your username and password:
"""

from huggingface_hub import notebook_login

notebook_login()

"""Then you need to install `Git-LFS`.

If you are not using `Google Colab`, you may need to install `Git-LFS` manually, since the code below may not work and depending on your operating system. You can read about `Git-LFS` and how to install it [here](https://git-lfs.github.com/).
"""

! apt install git-lfs

"""Make sure your version of `Transformers` is at least 4.11.0 since the functionality was introduced in that version:"""

import transformers

print(transformers.__version__)

"""You can find a script version of this notebook to fine-tune your model in a distributed fashion using multiple GPUs or TPUs [here](https://github.com/huggingface/transformers/tree/master/examples/seq2seq).

# Fine-tuning a model on a summarization task

In this notebook, we will see how to fine-tune one of the [🤗`Transformers`](https://github.com/huggingface/transformers) model for a summarization task. We will use the [PubMed Summarization dataset](https://huggingface.co/datasets/ccdv/pubmed-summarization) which contains PubMed articles accompanied with abstracts.

![Widget inference on a summarization task](https://github.com/huggingface/notebooks/blob/master/examples/images/summarization.png?raw=1)

We will see how to easily load the dataset for this task using 🤗 `Datasets` and how to fine-tune a model on it using the `Trainer` API.
"""

model_checkpoint = "deep-learning-analytics/wikihow-t5-small"

"""This notebook is built to run  with any model checkpoint from the [Model Hub](https://huggingface.co/models) as long as that model has a sequence-to-sequence version in the Transformers library. Here we picked the [`deep-learning-analytics/wikihow-t5-small`](https://huggingface.co/deep-learning-analytics/wikihow-t5-small) checkpoint.

## Loading the dataset

We will use the [🤗 `Datasets`](https://github.com/huggingface/datasets) library to download the data and get the metric we need to use for evaluation (to compare our model to the benchmark). This can be easily done with the functions `load_dataset` and `load_metric`.
"""

from datasets import load_dataset, load_metric

raw_datasets = load_dataset("ccdv/pubmed-summarization")
metric = load_metric("rouge")

"""The `dataset` object itself is [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which contains one key for the training, validation and test set:"""

raw_datasets

"""To access an actual element, you need to select a split first, then give an index:"""

raw_datasets["train"][0]

"""Since the `pubmed` data is extremely large, we are going to remove rows so that we have a training set of 8,000, a validation set of 2,000, and a test set of 2,000. """

raw_datasets["train"] = raw_datasets["train"].select(range(1, 8001))
raw_datasets["validation"] = raw_datasets["validation"].select(range(1, 2001))
raw_datasets["test"] = raw_datasets["test"].select(range(1, 2001))

"""To get a sense of what the data looks like, the following function will show some examples picked randomly in the dataset."""

import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

show_random_elements(raw_datasets["train"])

"""The metric is an instance of [`datasets.Metric`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Metric):"""

metric

"""You can call its `compute` method with your predictions and labels, which need to be list of decoded strings:"""

fake_preds = ["hello there", "general kenobi"]
fake_labels = ["hello there", "general kenobi"]
metric.compute(predictions=fake_preds, references=fake_labels)

"""## Preprocessing the data

Before we can feed those texts to our model, we need to preprocess them. This is done by a 🤗 `Transformers` `Tokenizer` which will (as the name indicates) tokenize the inputs (including converting the tokens to their corresponding IDs in the pretrained vocabulary) and put it in a format the model expects, as well as generate the other inputs that the model requires.

To do all of this, we instantiate our tokenizer with the `AutoTokenizer.from_pretrained` method, which will ensure:

- we get a tokenizer that corresponds to the model architecture we want to use,
- we download the vocabulary used when pretraining this specific checkpoint.

That vocabulary will be cached, so it's not downloaded again the next time we run the cell.

For this we need `sentencepiece` installed.
"""

! pip install sentencepiece

"""Now we can instantiate the tokenizer."""

from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

"""By default, the call above will use one of the fast tokenizers (backed by Rust) from the 🤗 `Tokenizers` library.

You can directly call this tokenizer on one sentence or a pair of sentences:
"""

tokenizer("Hello, this one sentence!")

"""Depending on the model you selected, you will see different keys in the dictionary returned by the cell above. They don't matter much for what we're doing here (just know they are required by the model we will instantiate later), you can learn more about them in [this tutorial](https://huggingface.co/transformers/preprocessing.html) if you're interested.

Instead of one sentence, we can pass along a list of sentences:
"""

tokenizer(["Hello, this one sentence!", "This is another sentence."])

"""To prepare the targets for our model, we need to tokenize them inside the `as_target_tokenizer` context manager. This will make sure the tokenizer uses the special tokens corresponding to the targets:"""

with tokenizer.as_target_tokenizer():
    print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))

"""If you are using one of the five T5 checkpoints we have to prefix the inputs with "summarize:" (the model can also translate and it needs the prefix to know which task it has to perform)."""

if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""

"""We can then write the function that will preprocess our samples. We just feed them to the `tokenizer` with the argument `truncation=True`. This will ensure that an input longer that what the model selected can handle will be truncated to the maximum length accepted by the model. The padding will be dealt with later on (in a data collator) so we pad examples to the longest length in the batch and not the whole dataset.

The max input length of `deep-learning-analytics/wikihow-t5-small` is 512, so `max_input_length = 512`.
"""

max_input_length = 512
max_target_length = 256

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["abstract"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

"""This function works with one or several examples. In the case of several examples, the tokenizer will return a list of lists for each key:"""

preprocess_function(raw_datasets['train'][:2])

"""To apply this function on all the pairs of sentences in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command."""

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

"""Even better, the results are automatically cached by the 🤗 `Datasets` library to avoid spending time on this step the next time you run your notebook. The 🤗 `Datasets` library is normally smart enough to detect when the function you pass to map has changed (and thus requires to not use the cache data). For instance, it will properly detect if you change the task in the first cell and rerun the notebook. 🤗 `Datasets` warns you when it uses cached files, you can pass `load_from_cache_file=False` in the call to `map` to not use the cached files and force the preprocessing to be applied again.

Note that we passed `batched=True` to encode the texts by batches together. This is to leverage the full benefit of the fast tokenizer we loaded earlier, which will use multi-threading to treat the texts in a batch concurrently.

## Fine-tuning the model

Now that our data is ready, we can download the pretrained model and fine-tune it. Since our task is of the sequence-to-sequence kind, we use the `AutoModelForSeq2SeqLM` class. Like with the tokenizer, the `from_pretrained` method will download and cache the model for us.
"""

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

"""Note that  we don't get a warning like in our classification example. This means we used all the weights of the pretrained model and there is no randomly initialized head in this case.

To instantiate a `Seq2SeqTrainer`, we will need to define three more things. The most important is the [`Seq2SeqTrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Seq2SeqTrainingArguments), which is a class that contains all the attributes to customize the training. It requires one folder name, which will be used to save the checkpoints of the model, and all other arguments are optional:
"""

batch_size = 2
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-pubmed",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
    seed = 42,
)

"""Here we set the evaluation to be done at the end of each epoch, tweak the learning rate, use the `batch_size` defined at the top of the cell and customize the weight decay. Since the `Seq2SeqTrainer` will save the model regularly and our dataset is quite large, we tell it to make three saves maximum. Lastly, we use the `predict_with_generate` option (to properly generate summaries) and activate mixed precision training (to go a bit faster).

The last argument to setup everything so we can push the model to the [Hub](https://huggingface.co/models) regularly during training. Remove it if you didn't follow the installation steps at the top of the notebook. If you want to save your model locally in a name that is different than the name of the repository it will be pushed, or if you want to push your model under an organization and not your name space, use the `hub_model_id` argument to set the repo name (it needs to be the full name, including your namespace: for instance `"sgugger/t5-finetuned-xsum"` or `"huggingface/t5-finetuned-xsum"`).

Then, we need a special kind of data collator, which will not only pad the inputs to the maximum length in the batch, but also the labels:
"""

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

"""The last thing to define for our `Seq2SeqTrainer` is how to compute the metrics from the predictions. We need to define a function for this, which will just use the `metric` we loaded earlier, and we have to do a bit of pre-processing to decode the predictions into texts:"""

import nltk
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

"""Then we just need to pass all of this along with our datasets to the `Seq2SeqTrainer`:"""

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

"""We can now finetune our model by just calling the `train` method:"""

trainer.train()

"""You can now upload the result of the training to the Hub, just execute this instruction:"""

trainer.push_to_hub()

"""You can now share this model with all your friends, family, favorite pets: they can all load it with the identifier `"your-username/the-name-you-picked"` so for instance:

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("sgugger/my-awesome-model")
```
"""