# DWB-Modules

**Selected feature modules from DWB-Net for underwater object detection.** This release does not include the complete detector, training or evaluation pipelines, or pretrained weights.

[Highlights](#highlights) · [Modules](#repository-structure) · [Installation](#installation) · [Quick Start](#quick-start) · [Datasets](#datasets) · [Paper](#paper) · [Citation](#citation) · [License](#license)

DWB-Net connects degradation-aware wavelet evidence modeling with reliability-guided Bayesian query refinement. This repository provides standalone PyTorch implementations of its wavelet feature modules.

## Highlights

The paper develops two connected contributions:

- **Degradation-conditioned asymmetric wavelet modeling.** A shared local degradation descriptor guides directional high-frequency selection and low-frequency residual compensation. This design adapts fine-detail and coarse-structure processing to local underwater degradation.
- **Reliability-guided Bayesian query refinement.** Attention aggregates spatial detail reliability to modulate query-specific wavelet observation precision. The resulting update connects evidence reliability with query refinement and learned localization-uncertainty estimates.

Experiments with the full DWB-Net model show improved detection performance on RUOD and Trash-ICRA19. Detailed results and evaluation protocols will be released alongside the paper.

## Repository Structure

```text
DWB-Modules/
├── models/
│   ├── wavelet.py
│   ├── degradation.py
│   └── dwm.py
├── data/
│   ├── sample_input.py
│   └── README.md
├── demo.py
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

| Component | Implementation |
| --- | --- |
| Haar subband decomposition | [models/wavelet.py](models/wavelet.py) |
| Learned degradation cues | [models/degradation.py](models/degradation.py) |
| Degradation-aware modulation and feature fusion | [models/dwm.py](models/dwm.py) |
| Synthetic demo inputs | [data/sample_input.py](data/sample_input.py) |
| Runnable example | [demo.py](demo.py) |

## Installation

Use Python 3.10 or later and PyTorch 2.3 or later. CPU execution has been checked with Python 3.10.18 and PyTorch 2.7.1+cu128.

From the repository root:

```bash
python -m pip install -r requirements.txt
```

## Quick Start

Choose the input dimensions when running the CPU example. Replace the placeholders below with positive integers:

```text
python demo.py --batch-size <batch_size> --channels <channels> --height <height> --width <width>
```

The example uses synthetic feature tensors and reports input and output shapes. It requires no dataset or pretrained weights. Run `python demo.py --help` for the available options.

To apply the feature module to a BCHW feature tensor from your own code:

```python
import torch
from models.dwm import DWM

module = DWM(channels=64).eval()

def enhance_features(x):
    with torch.inference_mode():
        return module(x)
```

<details>
<summary><strong>Usage notes</strong></summary>

- Keep the `models/` and `data/` directories alongside `demo.py`.
- `DWM` preserves the input shape and dtype, with FP32 internal computation. Keep its parameters and BatchNorm buffers in FP32.
- The output projection is initialized to zero, so the module initially preserves ordinary finite FP32 inputs. Learned enhancement requires trained weights.
- The demo illustrates the interface; it does not evaluate detection performance.

</details>

## Datasets

Experiments with the full DWB-Net model use two public underwater object detection datasets:

| Dataset | Original source |
| --- | --- |
| RUOD | [Project and download links](https://github.com/dlut-dimt/RUOD) |
| Trash-ICRA19 | [Project page](https://irvlab.cs.umn.edu/resources/trash-icra19) · [Dataset record and download](https://doi.org/10.13020/x0qn-y082) |

The datasets are provided by their original authors and are not redistributed here. Please follow their original terms and cite the corresponding works. [Dataset citations](data/README.md#dataset-citations) are provided separately. The included demo uses synthetic features and requires no dataset downloads.

## Release Scope

This repository contains the standalone feature modules and a minimal example. The complete detector, query-level refinement, training and evaluation pipelines, datasets, and pretrained checkpoints are outside this release.

## Paper

The DWB-Net manuscript is currently in the submission process. This section will be updated after submission. The paper link, detailed results, and citation information will be added when the manuscript becomes publicly available.

## Citation

If you use this code in research, please cite this repository using its title, author information, repository URL, and the version or commit you used. Once the paper is publicly available, please also cite the paper. Its BibTeX entry will be added here.

## License

This source-available release is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Use is limited to the purposes permitted by that license. Commercial uses outside those permissions require separate authorization from the copyright holders.

Redistributed copies must retain the license terms or their URL and the following notice:

Required Notice: DWB-Modules provides selected feature modules from DWB-Net.

The research citation request above is separate from the license terms. Third-party datasets remain subject to their own terms.

## Contact

Use the repository's Issues page for implementation questions and bug reports.

