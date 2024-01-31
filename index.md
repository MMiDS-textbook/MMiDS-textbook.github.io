# <b>MATHEMATICAL METHODS in DATA SCIENCE (with Python)</b>

**Author:** [Sebastien Roch](https://people.math.wisc.edu/~roch/), Department of Mathematics, University of Wisconsin-Madison

This textbook on the **mathematics of data** has two intended audiences:

- *For students majoring in math or other quantitative fields like physics, economics, engineering, etc.*: it is meant as an invitation to data science and AI from a rigorous mathematical perspective.

- *For mathematically-inclined students in data science related fields (at the undergraduate or graduate level)*: it can serve as a mathematical companion to machine learning, AI, and statistics courses.

Content-wise it is a second course in **linear algebra**, **multivariable calculus**, and **probability theory** motivated by and illustrated on data science applications. As such, the reader is expected to be familiar with the basics of those areas, as well as to have been exposed to proofs -- but no knowledge of data science is assumed. Moreover, while the emphasis is on the mathematical concepts and methods, coding is used throughout. Basic familiarity with [Python](https://docs.python.org/3/tutorial/index.html) will suffice. The book provides an introduction to some specialized packages, especially [Numpy](https://numpy.org), [NetworkX](https://networkx.org), and [PyTorch](https://pytorch.org/).

The book is based on Jupyter notebooks that were developed for
[MATH 535: Mathematical Methods in Data Science](https://people.math.wisc.edu/~roch/mmids/), a one-semester advanced undergraduate and Master's level course
offered at [UW-Madison](https://math.wisc.edu/).

```{tableofcontents}
```

````{important}
 To run the code in these notes, you need to import the
 following librairies.

```python
# PYTHON 3
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mmids
seed = 535
rng = np.random.default_rng(seed)
```

The file `mmids.py` is [here](https://raw.githubusercontent.com/MMiDS-textbook/MMiDS-textbook.github.io/main/utils/mmids.py).

All datasets can be downloaded on the [GitHub page of the notes](https://github.com/MMiDS-textbook/MMiDS-textbook.github.io/tree/main/utils/datasets).

Jupyter notebooks containing just the code are provided at the end of each chapter.
Running them in [Google Colaboratory](https://colab.google) is recommended.  
````

```{note}
If you find typos, please open an issue on GitHub by using the provided button
in the top right menu.
```

**Image credit:** Sidebar logo made with [Midjourney](https://www.midjourney.com/)
