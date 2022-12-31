# Induced Set Attention Block [![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2FRishit-dagli%2Fisab)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FRishit-dagli%2Fisab)

![PyPI](https://img.shields.io/pypi/v/isab)
[![Run Tests](https://github.com/Rishit-dagli/isab/actions/workflows/tests.yml/badge.svg)](https://github.com/Rishit-dagli/isab/actions/workflows/tests.yml)
[![Upload Python Package](https://github.com/Rishit-dagli/isab/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Rishit-dagli/isab/actions/workflows/python-publish.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rishit-dagli/isab/blob/main/example/isab_example.ipynb)

![GitHub License](https://img.shields.io/github/license/Rishit-dagli/isab)
[![GitHub stars](https://img.shields.io/github/stars/Rishit-dagli/isab?style=social)](https://github.com/Rishit-dagli/isab/stargazers)
[![GitHub followers](https://img.shields.io/github/followers/Rishit-dagli?label=Follow&style=social)](https://github.com/Rishit-dagli)
[![Twitter Follow](https://img.shields.io/twitter/follow/rishit_dagli?style=social)](https://twitter.com/intent/follow?screen_name=rishit_dagli)

Set Transformer from the paper "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks" is based on Isab, an attention scheme inspired by inducing point methods from sparse Gaussian process literature and making it permutation invariant. It proposes to reduce attention from O(n²) to O(mn), where m is the number of inducing points (learned latents).


## Installation

Run the following to install:

```sh
pip install isab
```

## Developing isab

To install `isab`, along with tools you need to develop and test, run the following in your virtualenv:

```sh
git clone https://github.com/Rishit-dagli/isab.git
# or clone your own fork

cd isab
pip install -e .[dev]
```

To run rank and shape tests run any of the following:

```py
python -m isab.test_isab
pytest isab --verbose
```

## Usage

```python
import tensorflow as tf
from isab import Isab


attn = Isab(
    dim = 512,
    heads = 8,
    num_latents = 128
)

seq = tf.random.normal((1, 16384, 512)) # (batch, seq, dim)
mask = tf.ones((1, 16384), dtype = tf.bool) # (batch, seq)

out, latents = attn(seq, mask = mask) # (1, 16384, 512), (1, 128, 512)
```

You can also choose not to set the number of latents, and pass in the latents yourself:

```python
import tensorflow as tf
from isab import Isab


attn = Isab(
    dim = 512,
    heads = 8
)

seq = tf.random.normal((1, 16384, 512)) # (batch, seq, dim)
latents = tf.Variable(tf.random.normal((128, 512))) # some memory, passed through multiple Isabs

out, new_latents = attn(seq, latents) # (1, 16384, 512), (1, 128, 512)
```

## Want to Contribute 🙋‍♂️?

Awesome! If you want to contribute to this project, you're always welcome! See [Contributing Guidelines](CONTRIBUTING.md). You can also take a look at [open issues](https://github.com/Rishit-dagli/isab/issues) for getting more information about current or upcoming tasks.

## Want to discuss? 💬

Have any questions, doubts or want to present your opinions, views? You're always welcome. You can [start discussions](https://github.com/Rishit-dagli/isab/discussions).

## Citation

```bibtex
@misc{https://doi.org/10.48550/arxiv.1810.00825,
  doi = {10.48550/ARXIV.1810.00825},
  
  url = {https://arxiv.org/abs/1810.00825},
  
  author = {Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam R. and Choi, Seungjin and Teh, Yee Whye},
  
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
  
  publisher = {arXiv},
  
  year = {2018},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

@misc{https://doi.org/10.48550/arxiv.2212.11972,
  doi = {10.48550/ARXIV.2212.11972},
  
  url = {https://arxiv.org/abs/2212.11972},
  
  author = {Jabri, Allan and Fleet, David and Chen, Ting},
  
  keywords = {Machine Learning (cs.LG), Computer Vision and Pattern Recognition (cs.CV), Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Scalable Adaptive Computation for Iterative Generation},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## License

```
Copyright 2020 Rishit Dagli

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```