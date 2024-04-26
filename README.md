# Skyline Extraction

This repo contains two high-level implementations for skyline extraction.

The two approaches are:
1. OneFormer based (https://arxiv.org/abs/2211.06220)
2. Frajberg CNN based (https://link.springer.com/chapter/10.1007/978-3-319-68612-7_2)

Both models were trained on images from the dataset: https://www.kaggle.com/datasets/mieszkokaminski/geopose3k-processed-original-new

# OneFormer
A pre-trained model is on HuggingFace. This model is automatically loaded by when using the API provided.

The first 900 images of the dataset were used to fine-tune OneFormer.

## Examples
A full usage example is available under `examples/Example-OneFormer.ipynb`

## Reference
### Class: `SkylineExtraction.OneFormer.SkylinePipeline`

Parameters:
- `device: torch.Device` - the device that the model will run on
- (Optional) `checkpoint: str` - a path to the checkpoint, default value of `mieszkok/shi-labs_oneformer_ade20k_swin_large_geopose3k_original_images900_epochs5`


# Frajberg CNN
Two pre-trained models are included:
1. Mod-GeoPose3k-321
2. Mod-GeoPose3k-642

A subset of 3,093 images from the dataset was used for training and validation using a 4:1 split.

## Examples
A full usage example is available under `examples/Example-FrajbergCNN.ipynb`

## Reference
### Class: `SkylineExtraction.Frajberg.SkylinePipeline`

Parameters:
- `model: nn.Module` - an instance of the `SkylineCNN` class
- `patch_extraction: BasePatchExtractor` - an instance of one of the three patch extractors: `PixelWisePatchExtractor`, `ResizePatchExtractor` or `CannyPatchExtractor`
- `device: torch.Device` - the device that the model will run on
- `skyline_search_params: FindSkylineParams` - an instance of `FindSkylineParams` providing a configuration for the skyline search
- `eps: int` - the epsilon value for creating the graph 

Methods:
- `run_inference` - runs the inference stage of the pipeline
  - Parameters:
    - `return_probability_map: bool` - defaults to False, if true instead of a discrete binary classification map, each pixel will hold a probability - that is, the probability whether it belongs to the skyline or not.
- `find_skyline_path` - runs the skyline search stage of the pipeline 
  - Parameters:
    - `eps: int` - an overload value for the constructor `eps`
- `convert_skyline_path_to_image` - converts a found skyline path into an image - that is, a binary image with the same dimensions as the original input with a skyline drawn on.
- `downfill_skyline` - converts the skyline path image into a binary mask
  - Parameters:
    - `fill_sides: bool` - defaults to True - `fill_sides` decides whether to fill the gaps on the side of the mask if there are any, if the skyline path does not stretch the full length of the image. 

### Class: `SkylineExtraction.Frajberg.PixelWisePatchExtractor`

Parameters:
- `img: NDArray` - the image which to perform patch extraction on
- 
### Class: `SkylineExtraction.Frajberg.ResizePatchExtractor`

Parameters:
- `img: NDArray` - the image which to perform patch extraction on
- `resize_dim: Size` - the size to which an image should be resized
- `return_image_resized: bool` - true by default, if false the CNN output will not be resized to the shape of the original input image

### Class: `SkylineExtraction.Frajberg.CannyPatchExtractor`

Parameters:
- `img: NDArray` - the image which to perform patch extraction on
- `t1: int` - the first threshold for Canny
- `t2: int` - the second threshold for Canny
