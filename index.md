# <b>MATHEMATICAL METHODS in DATA SCIENCE</b>

**Author:** [Sebastien Roch](https://people.math.wisc.edu/~roch/), Department of Mathematics, University of Wisconsin-Madison

**Full title:** Mathematical Methods in Data Science: Bridging Theory and Applications with Python

```{image} 9781009509404i.jpg
:width: 125px
:align: right
```
**Print copy:** A print version of this textbook has been published by Cambridge University Press. It is available to order [here](https://www.amazon.com/Mathematical-Methods-Data-Science-Applications/dp/1009509403). This online version will remain available and maintained. Additional resources can be found on the [publisher's website](https://www.cambridge.org/highereducation/books/mathematical-methods-in-data-science/6CB866F77A7CA33109EF99910CFA40BC#overview). See [here](MMiDS_Errata.pdf) for a list of typos. The book is based on Jupyter notebooks that were developed for
[MATH 535: Mathematical Methods in Data Science](https://people.math.wisc.edu/~roch/mmids/), a one-semester advanced undergraduate and master's level course offered at [UW-Madison](https://math.wisc.edu/). The online version was generated using [Jupyter Book](https://jupyterbook.org/stable/). See [here](https://executablebooks.org/en/latest/gallery/) for a collection of other such books.

**Description:** This textbook on the **mathematics of data and AI** has several intended audiences:

- *For students majoring in math or other quantitative fields like physics, economics, engineering, etc.*: it is meant as an invitation to data science and AI from a rigorous mathematical perspective.

- *For mathematically-inclined students in data science related fields (at the undergraduate or graduate level)*: it can serve as a mathematical companion to machine learning, AI, and statistics courses.

Content-wise it is a second course in **multivariable calculus**, **linear algebra**, and **probability** motivated by and illustrated on data science applications. As such, the reader is expected to be familiar with the basics of those areas, as well as to have been exposed to proofs -- but no knowledge of data science is assumed. Moreover, while the emphasis is on the mathematical concepts and methods, coding is used throughout. Basic familiarity with [Python](https://docs.python.org/3/tutorial/index.html) will suffice. The book provides an introduction to some specialized packages, especially [Numpy](https://numpy.org), [NetworkX](https://networkx.org), and [PyTorch](https://pytorch.org/).


```{tableofcontents}
```

````{important}
 To run the code in this book, you need to import the
 following librairies.

```python
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import torch
import mmids
```

The file `mmids.py` is [here](https://raw.githubusercontent.com/MMiDS-textbook/MMiDS-textbook.github.io/main/utils/mmids.py). All datasets can be downloaded on the [GitHub page of the notes](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/tree/main/utils/datasets).

Jupyter notebooks containing just the code are provided at the end of each chapter.
Running them in [Google Colaboratory](https://colab.google) is recommended.  The notebooks are also available in slideshow format. The slideshows were created using Jupyter; 
hence, instructors can create their own tailored version directly from the notebooks. 
````

```{note}
If you find typos (in the online or print version), please open an issue on GitHub by using the provided button
in the top right menu.
```

**Supplementary materials:** This online version also contains materials that supplement what can be found in the print book. Specifically, at the end of each chapter, one will find an *Online Supplementary Materials* section with:

- *Just The Code*: a Jupyter notebook and slideshow with all the code from the chapter
- *Self-Assessement Quizzes:* expanded, interactive self-assessment quizzes for each section
- *Auto-Quizzes:* Jupyter notebooks featuring random quizzes with automatically generated answers
- *Solutions to Warm-Up Worksheets:* solutions to all odd-numbered warm-up exercises
- *Additional Sections:* additional content, typically at a somewhat more advanced level than the published book (e.g., proofs of more advanced results not required in the main text)
 

**Image credit:** Sidebar logo made with [Midjourney](https://www.midjourney.com/)
