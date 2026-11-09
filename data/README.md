# Demo Inputs

`sample_input.py` creates random CPU features for `demo.py`. Supply the batch size, channel count, height, and width at runtime; no fixed input dimensions are included. An optional seed makes the random input repeatable.

The returned tensor uses `torch.float32` with shape `[batch_size, channels, height, width]`. These arguments control the demo input, not the paper's training configuration or learned weights.

See the [main README](../README.md) for installation and module usage.

## Research Datasets

Research experiments with the full DWB-Net model use **RUOD** and **Trash-ICRA19**. This repository provides links and citations for these benchmarks; it does not include dataset files or dataset loaders. The synthetic demo requires neither dataset.

| Dataset | Original sources | Reference |
| --- | --- | --- |
| RUOD | [Project repository](https://github.com/dlut-dimt/RUOD) · [Alternative download source](https://github.com/xiaoDetection/RUOD) | [Fu et al., Neurocomputing, 2023](https://doi.org/10.1016/j.neucom.2022.10.039) |
| Trash-ICRA19 | [Lab project page](https://irvlab.cs.umn.edu/resources/trash-icra19) · [Dataset record and download](https://doi.org/10.13020/x0qn-y082) | [Dataset citation](https://experts.umn.edu/en/datasets/trash-icra19-a-bounding-box-labeled-dataset-of-underwater-trash/) · [Fulton et al., ICRA, 2019](https://doi.org/10.1109/ICRA.2019.8793975) |

Obtain the datasets from their original providers, follow their respective terms, and cite the original work when using them. The license for DWB-Modules does not apply to these third-party datasets.

<a id="dataset-citations"></a>
<details>
<summary><strong>Dataset citations (BibTeX)</strong></summary>

**RUOD**

```bibtex
@article{fu2023rethinking,
  title = {Rethinking general underwater object detection: Datasets, challenges, and solutions},
  author = {Fu, Chenping and Liu, Risheng and Fan, Xin and Chen, Puyang and Fu, Hao and Yuan, Wanqi and Zhu, Ming and Luo, Zhongxuan},
  journal = {Neurocomputing},
  volume = {517},
  pages = {243--256},
  year = {2023},
  doi = {10.1016/j.neucom.2022.10.039}
}
```

**Trash-ICRA19 dataset**

```bibtex
@misc{fulton2020trashicra19,
  title = {{[Trash-ICRA19]} A Bounding Box Labeled Dataset of Underwater Trash},
  author = {Fulton, Michael S. and Hong, Jungseok and Sattar, Junaed},
  publisher = {Data Repository for the University of Minnesota},
  year = {2020},
  doi = {10.13020/x0qn-y082},
  url = {https://doi.org/10.13020/x0qn-y082}
}
```

**Corresponding ICRA paper**

```bibtex
@inproceedings{fulton2019robotic,
  title = {Robotic Detection of Marine Litter Using Deep Visual Detection Models},
  author = {Fulton, Michael and Hong, Jungseok and Islam, Md Jahidul and Sattar, Junaed},
  booktitle = {2019 International Conference on Robotics and Automation (ICRA)},
  pages = {5752--5758},
  year = {2019},
  doi = {10.1109/ICRA.2019.8793975}
}
```

</details>
