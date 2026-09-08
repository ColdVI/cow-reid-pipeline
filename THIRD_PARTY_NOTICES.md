# Third-party notices

The MIT license in this repository covers only the original pipeline code.
Dependencies and model weights retain their own licenses.

## Ultralytics YOLO

The optional `yolo` detector backend imports the `ultralytics` package and can
use `yolo11n-seg.pt`. Ultralytics states that its code and trained models are
available under AGPL-3.0 by default, with an Enterprise license for proprietary,
private, or commercial use. Review the current terms before use:

https://www.ultralytics.com/license

The pretrained YOLO weight is intentionally not included in the distributable
ZIP. The fully offline `motion` detector fallback does not use Ultralytics.

## PyTorch and torchvision

The Re-ID inference and metric fine-tuning path use PyTorch and torchvision.
Their packages, pretrained weights, and bundled components retain their own
license notices.

## OpenCows2020 MetricLearningIdentification

The optional cattle-specific Re-ID initialization is downloaded from:

https://github.com/CWOA/MetricLearningIdentification

Paper: https://arxiv.org/abs/2006.09205

The upstream repository states an MIT license. The checkpoint is not bundled in
this ZIP. `cow-reid download-cattle-weights` accepts it only when SHA-256 is:

```text
c126fcc5aa446e2f93134c9a013e47cad232f4147114c9e409ce21844598b34a
```

Review upstream license and dataset terms before redistribution or commercial use.
