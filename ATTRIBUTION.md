# Attribution

## Datasets
All three are published under **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). They are downloaded by
`data/download.py` into `data/raw/` and are not committed to this repository. Changes we make: images are resized
and cut into patches for training, and weak "pseudo masks" are derived from image-level labels
(`holdpoint/vision/pseudo_mask.py`). None of the authors endorse this project.

- **Concrete Crack Segmentation Dataset**: Özgenel, Ç. F. (2019). Mendeley Data, V1.
  https://doi.org/10.17632/jwsn7tfbrp.1. License: CC BY 4.0. 458 photos of concrete with binary crack masks.
  Used to train and evaluate the segmentation model and to check the pseudo masks against real masks.
- **SDNET2018**: Maguire, M., Dorafshan, S., Thomas, R. J. (2018). *SDNET2018: A concrete crack image dataset for
  machine learning applications.* Utah State University. https://doi.org/10.15142/T3TD19. License: CC BY 4.0.
  Over 56,000 cracked and non-cracked patches of bridge decks, walls and pavements. Used for the crack / no-crack
  classifier and, through pseudo masks, for segmentation pre-training.
  Related paper: Dorafshan, S., Thomas, R. J., Maguire, M. (2018). *Data in Brief* 21, 1664-1668.
- **Concrete Crack Images for Classification**: Özgenel, Ç. F. (2019). Mendeley Data, V2.
  https://doi.org/10.17632/5y9wdsg2zt.2. License: CC BY 4.0. 40,000 patches (227 x 227 px), half with cracks.
  Second source for the crack / no-crack classifier.

## Software
- segmentation_models_pytorch (MIT)
- PyTorch, torchvision (BSD-style)
- OpenCV (Apache-2.0), scikit-image (BSD-3-Clause), NumPy (BSD-3-Clause)
- libarchive-c (CC0) for unpacking the RAR archives
- Qwen2.5-VL-7B-Instruct (Apache-2.0), served through an OpenAI-compatible endpoint (vLLM)
