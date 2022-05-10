# Accelerating Systematic Reviews: Developing a Pipeline for Creating Enhanced Abstracts

**Author**: Kevin Patyk

**Supervisors**: [Ayoub Bagheri](https://www.uu.nl/medewerkers/ABagheri) & [Rens van de Schoot](https://www.rensvandeschoot.com/)

Department of Methodology and Statistics, Utrecht University, the Netherlands

In colloboration with [ASReview](https://asreview.nl/).

<p align="center">
  <img 
    height="150"
    src="Elas/ASReviewLogo.png"
  >
</p>

# Introduction 

Systematic reviews are valuable because they summarize what is and is not known on a
specific topic using a rigorous and transparent methodology. However, the screening process of
systematic reviews is time-consuming and prone to human error. Recent developments in machine
learning have sought to facilitate the screening process through the use of automated technologies.
One such program is ASReview, which uses reinforcement learning to reduce the number of
articles that need to be screened manually. Although ASReview performs well in previous studies,
it is only able to present the abstract to the user, which may not provide enough information to
make a decision about inclusion status. The goal of this thesis is to establish a pipeline for
converting PDF documents to a clean text format, which can then be used to automatically make
summaries of the full text (enhanced abstracts). In total, 15 text summarization algorithms are
tested and evaluated using an open test database. Then, the best performing and most practical
algorithm is used to generate summaries of available full texts used for a meta-analysis on
depression. Finally, a simulation study is conducted to determine how much time the automated
summaries save during the screening process in comparison to the original abstracts and full text.
The results show that the pipeline is successful in converting PDFs into a clean text format which
can be used to make enhanced abstracts for use in systematic reviews. The simulation study
demonstrates that the enhanced abstracts performed marginally worse relative to the original
abstracts and full text, but still save a significant amount of time compared to manual screening.
Follow-up research is needed to draw more concrete conclusions about the performance of the
enhanced abstracts. Areas of improvement and suggestions for future research are provided.

# Data 

The data used in this study are publicly available.

For Stage 1, the [PubMed](https://github.com/armancohan/long-summarization) data collected by Cohan et al. (2018) is used. 

For Stages 2 & 3, the systematic review data from [“Psychological theories of depressive relapse and recurrence”](https://osf.io/r45yz/) collected by Brouwer et al. (2019) is used. 

The final datasets containing the metadata with full text and metadata with enhanced abstracts could not be posted online due to publisher licensing restrictions. 

# Content: Stage 1 - Model Evaluation & Selection

This folder contains the `Python` (version 3.10) scripts in `ipynb` & `.py` formats for evaluating the 15 text summarization algorithms.

In order to make summaries, the graphics processing unit (GPU) is used to optimize the models. In order to use the GPU to optimize deep learning models, NVIDIA's [`CUDA`](https://developer.nvidia.com/cuda-downloads) (version 11.6) is used.  

* `bart-base-pubmed`
* `bart-large-cnn-pubmed`
* `bart-large-pubmed`
* `bigbird-pegasus-large-arxiv-pubmed`
* `bigbird-pegasus-large-bigpatent-pubmed`
* `distilbart-cnn-12-3-pubmed`
* `distilbart-cnn-12-6-pubmed`
* `distilbart-cnn-6-6-pubmed`
* `distilbart-xsum-12-1-pubmed`
* `pegasus-arxiv-pubmed`
* `pegasus-cnn-dailymail-pubmed`
* `pegasus-large-pubmed`
* `t5-base-pubmed`
* `t5-small-pubmed`
* `t5-small-wikihow-pubmed`

The models, along with evaluation results, are also uploaded to the [author's Hugging Face profile](https://huggingface.co/Kevincp560). 

# Content: Stage 2 - Preprocessing & Model Application

This folder contains the `R` (version 4.1.2) scripts in `.Rmd` and `.html` formats used during preprocessing. Additionally, it contains the Python (version 3.10) script in `ipynb` & `.py` formats for creating the the summaries using the selected algorithm from Stage 1.

The scripts for identifying articles and parsing the PDF files were provided by [Bianca Kramer](https://www.uu.nl/medewerkers/bmrkramer).

* `Part 1 - Gathering Article Information`: This folder contains the `R` script which iterates through the metadata provided by the depression dataset and obtains article information, such as licensing and determining if the URL is a PDF.
* `Part 2 - Downloading Articles`: This folder contains the `R` script which is used to download all of the articles that have the URL as a PDF.

Before moving on to the third stage, the PDF files need to be parsed. [GROBID](https://github.com/kermitt2/grobid) is used to parse the PDF files into XML format before importing and cleaning the text in `R`. How to install and use GROBID can be done by reading the [documentation](https://grobid.readthedocs.io/en/latest/Install-Grobid/).

* `Part 3 - Preprocessing`: This folder contains the `R` script which is used to import, clean, and merge the full text with the metadata prior to making summaries. 

* `Part 4 - Algorithm Application`: This folder contaisn the `Python` scripts which create summaries of the full text articles. Similar to Stage 1, NVIDIA's `CUDA` is used to optimize the deep learning models.

# Content: Stage 3 - Simulation Study

The simulation study is conducted using [ASReview](https://asreview.nl/). The ASReview GitHub page can be found [here](https://github.com/asreview/asreview). For information about installation and usage, see the ASReview [documentation](https://asreview.readthedocs.io/en/latest/). The simulation study is conducted using the default settings of the simulation mode.

<p align="left">
  <img 
    height="100"
    src="Elas/ElasWorker.png"
  >
</p>

This folder contains the `output` (results) and `scripts` folders. 

To obtain the results of the simulation, you would normally run this on your terminal:

```
sh jobs.sh
````

However, this will not be possible without the final datasets the containing the metadata with full text and metadata with enhanced abstracts could not be posted online due to publisher licensing restrictions. 

# Content: Flow Diagram and ROUGE Figures

These folders contain the figures used in the final manuscript.

# Content: Manuscript

This folders contains the final manuscript in `PDF` format. 

# Acknowledgements

This work would not have been possible without the help and support of my amazing
supervisors, [Ayoub Bagheri](https://www.uu.nl/medewerkers/ABagheri) and [Rens van de Schoot](https://www.rensvandeschoot.com/). Thank you for always being supportive,
available, and incredibly helpful throughout the course of this study. I would also personally like
to thank [Bianca Kramer](https://www.uu.nl/medewerkers/bmrkramer) for providing me with invaluable assistance with identifying, downloading, and parsing the PDF files. Furthermore, a very special thank you to [Jelle Teijema](https://teije.ma/) for helping me with the simulation study and identifying areas for improvement. Lastly, I would like to thank the entirety of the [ASReview team](https://asreview.nl/about/) for being supportive and curious about the work. I am very excited and interested to see how this work is incorporated into ASReview in the future.

<p align="center">
  <img 
    height="200"
    src="Elas/ElasBalloons.png"
  >
</p>

# References

Brouwer, M. E., Williams, A. D., Kennis, M., Fu, Z., Klein, N. S., Cuijpers, P., & Bockting,
C. L. (2019). Psychological theories of depressive relapse and recurrence: A systematic
review and meta-analysis of prospective studies. Clinical Psychology Review, 74,
101773.

Cohan, A., Dernoncourt, F., Kim, D. S., Bui, T., Kim, S., Chang, W., & Goharian, N. (2018).
A discourse-aware attention model for abstractive summarization of long documents.
arXiv preprint arXiv:1804.05685

# License 

The content in this repository is published under the MIT license.

# Contact

For any questions or remarks, please send an email to kvn.ptk@gmail.com.
