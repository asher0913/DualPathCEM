# FaceScrub protocol

FaceScrub is not included in this repository. The public release contains no
face image, identity list, or derivative image crop.

The formal experiments used the CEM-compatible FaceScrub split after OpenCV
bilinear resizing to 48 x 48 pixels. The released dataset manifest records:

| Field | Value |
|---|---:|
| Non-empty identities in the released split | 526 |
| Training images | 39,969 |
| Validation images | 4,170 |
| Relative-path SHA-256 | `c05bd3b1aa8eaeaa2941a38bd47e2848047a598320888aeb024d5f88752c843d` |
| Source revision | `49f4e9bd6921596d3c10e5e6ed55a232aa1d63d4` |

The training and evaluation loaders resize images to 64 x 64 at runtime,
matching the released model configuration. The frozen Slot-CEM classifier has
530 outputs, as in the original FaceScrub configuration. The prepared archive
contains images for 526 identities; the remaining four outputs are unused.
This convention is retained to load the released checkpoints exactly.
Directory names must follow the ImageFolder convention:

```text
data/facescrub/
  train/<identity>/*.jpg
  val/<identity>/*.jpg
```

The repository cannot grant rights to the underlying images. Researchers must
obtain FaceScrub independently and comply with the dataset's terms and any
applicable privacy requirements.
