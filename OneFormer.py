from dataclasses import dataclass
from typing import List
import torch
from transformers import AutoModelForUniversalSegmentation, AutoProcessor
import numpy as np
from numpy.typing import NDArray, ArrayLike
from cv2.typing import MatLike

PRE_TRAINED_CHECKPOINT = "mieszkok/shi-labs_oneformer_ade20k_swin_large_geopose3k_original_images900_epochs5"

@dataclass
class SegmentsInfo:
    # dictionary with keys: id:int, label_id:int, was_fused:bool, score: float
    id: int
    label_id: int
    was_fused: bool
    score: float

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

@dataclass
class PanopticSegmentationOutput:
    segmentation: torch.Tensor
    segments_info: List[SegmentsInfo]

class SkylinePipeline:
    def __init__(self,
                 imgs: List[MatLike],
                 device: torch.device,
                 checkpoint=PRE_TRAINED_CHECKPOINT,
                 ) -> None:
        self.imgs = imgs
        self.device = device
        self.checkpoint = checkpoint
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForUniversalSegmentation.from_pretrained(checkpoint).to(device)

    @staticmethod
    def post_process_panoptic_segmentation(segmentation: torch.Tensor, segments_info: List[SegmentsInfo]) -> NDArray:
        # convert each segment_id in the segmentation to the label_id
        # of the corresponding segment_info
        segmentation_np: NDArray = segmentation.cpu().numpy()
        # to avoid mapping already mapped segments in the loop, we negate
        # the existing segment_ids, such that they will all be unique 
        segmentation_np = segmentation_np * -1
        for segment_info in segments_info:
            segmentation_np[segmentation_np == -segment_info.id] = segment_info.label_id

        return segmentation_np


    def run_inferences(self):
        self.model.eval()
        # set is_training attribute of base OneFormerModel to None after training
        # this disables the text encoder and hence enables to do forward passes
        # without passing text_inputs
        self.model.model.is_training = False
        self.model.to(self.device)

        for i in range(len(self.imgs)):
            img = self.imgs[i]

            panoptic_inputs = self.processor(images=img, task_inputs=["panoptic"], return_tensors="pt")
            panoptic_inputs = {k: v.to(self.device) for k, v in panoptic_inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**panoptic_inputs)
            panoptic_segmentation = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=[
                (img.shape[0], img.shape[1])])[0]
            mask = SkylinePipeline.post_process_panoptic_segmentation(
                panoptic_segmentation['segmentation'],
                [SegmentsInfo.from_dict(d) for d in panoptic_segmentation['segments_info']],
            )
            yield mask
        return 