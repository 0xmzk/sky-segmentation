import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from cv2.typing import Size

 
def pre_process(image_dir, num_positive_patches, num_negative_patches, patch_size, resize_dim: Size | None = None, seed=12):
    np.random.seed(seed)
    image_path = os.path.join(image_dir, "original.png")
    mask_path = os.path.join(image_dir, "mask.png")
    

    # read in mask and image
    image = cv2.imread(image_path)
    if resize_dim is not None:
        image = cv2.resize(image, resize_dim, interpolation=cv2.INTER_NEAREST)
    
    # if image width is not at least 100 we cannot use this image - this should not be the case though
    if image.shape[1] - 28 < 100:
        raise ValueError(f"{image_dir} has (width - 28) is less than 100")

    mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
    if resize_dim is not None:
        mask = cv2.resize(mask, resize_dim, interpolation=cv2.INTER_NEAREST)
    
    # np.unique(mask) -> [0 255] 
    # 0: sky
    # 255: foreground
    
    # find boundary between sky and foreground 
    edge_distance = 14
    # For each col, find y coord of first 255 value (this is the boundary)
    boundary_y_coords = np.argmax(mask == 255, axis=0)
    # Because length of boundary_y_coords is the x coord
    # we remove values that are too close to the left and right edge
    boundary_y_coords = boundary_y_coords[edge_distance:-edge_distance]
    # Remove any y coord values that are too close to the top and bottom edge
    boundary_y_coords = boundary_y_coords[(boundary_y_coords > edge_distance) & (boundary_y_coords < mask.shape[1] - edge_distance)]
    # Since we shifted everything down by edge_distance previously the indicies are currently
    # "true_indicies - edge_distance" and to get the true x coords we can simply add edge_distance
    # to the ones that were kept after filtering.
    boundary_y_coords_indicies = np.arange(len(boundary_y_coords))  + edge_distance
    # Combine the two to get an (x,y) stack
    boundary_xy_coords = np.stack([boundary_y_coords_indicies, boundary_y_coords]).T
    if len(boundary_xy_coords) < num_positive_patches:
        print(f"{image_dir} - skyline too close to edges, skipping ... ")
        return -1 
    # = START SAMPLING POSITIVE = 
    sample_xy_positive = boundary_xy_coords
    np.random.shuffle(sample_xy_positive)
    sample_xy_positive = sample_xy_positive[:num_positive_patches]

    sample_patches_positive = np.zeros([num_positive_patches, 3, patch_size, patch_size])
    for i in range(len(sample_patches_positive)):
        xy_point = sample_xy_positive[i]
        x,y = xy_point[0], xy_point[1]
        patch = np.asarray(image[y - 14:y + 15, x - 14:x + 15], dtype=np.float64)
        patch /= 255
        
        sample_patches_positive[i] = patch.T
    
    # = END SAMPLING POSITIVE = 

    # = START SAMPLING NEVATIVE = 
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(image_gray, 1, 1)

    # np.unique(edge_map) -> [0 255]
    # 0: no edge
    # 255: edge

    edge_xy = np.argwhere(edge_map == 255)
    # Filter out all edge responses that are too close to the edges of the image
    edge_xy = edge_xy[(edge_xy[:, 0] > edge_distance) & (edge_xy[:, 0] < mask.shape[0] - edge_distance) & (edge_xy[:, 1] > edge_distance) & (edge_xy[:, 1] < mask.shape[1] - edge_distance)]
    np.random.shuffle(edge_xy)
    # Filter out any edge responses that are a part of positive patches
    # Convert arrays to sets of tuples
    set1 = set(map(tuple, boundary_xy_coords))
    set2 = set(map(tuple, edge_xy))
    # Subtract set1 from set2 to find unique elements in arr2
    edge_xy_filtered = set2 - set1
    # Convert the set back to a numpy array
    edge_xy = np.array(list(edge_xy_filtered))
    
    sample_patches_negative = np.zeros((num_negative_patches, 3, patch_size, patch_size))
    for i in range(len(sample_patches_negative)):
        xy_point = edge_xy[i]
        x,y = xy_point[0], xy_point[1]
        patch = np.asarray(image[y - 14:y + 15, x - 14:x + 15], dtype=np.float64)
        patch /= 255

        sample_patches_negative[i] = patch.T

    # = END SAMPLING NEGATIVE = 

    sample_labels_positive = np.ones((num_positive_patches, 1), dtype=np.uint8)
    samples_labels_negative = np.zeros((num_negative_patches, 1), dtype=np.uint8)
    patches = np.concatenate((sample_patches_positive, sample_patches_negative), dtype=np.float64)
    labels = np.concatenate((sample_labels_positive, samples_labels_negative), dtype=np.uint8)
    
    return patches, labels

 
def pre_process_legacy(image_dir, positive_patches, negative_patches, patch_size, resize_dim: Size | None = None):
    np.random.seed(12)
    image_path = os.path.join(image_dir, "original.png")
    mask_path = os.path.join(image_dir, "mask.png")
    
    image = cv2.imread(image_path)
    if resize_dim is not None:
        image = cv2.resize(image, resize_dim, interpolation=cv2.INTER_NEAREST)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(image_gray, 1, 1)
    
    mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
    if resize_dim is not None:
        mask = cv2.resize(mask, resize_dim, interpolation=cv2.INTER_NEAREST)
    annotation_edge_map = cv2.Canny(mask, 5, 10)
    
    # == PROCESS POSITIVE PATCHES ==
    # get edge responses coordinates
    gt_edge_coordinates = np.argwhere(annotation_edge_map == 255)
    # ensure all x,y coordinates are at least 14 pixels away from the edge
    gt_edge_coordinates = gt_edge_coordinates[(gt_edge_coordinates[:, 0] > 14) & (gt_edge_coordinates[:, 0] < image_gray.shape[0] - 14) & (gt_edge_coordinates[:, 1] > 14) & (gt_edge_coordinates[:, 1] < image_gray.shape[1] - 14)]
    np.random.shuffle(gt_edge_coordinates)

    sample_positive = gt_edge_coordinates[:positive_patches]
    ones = np.ones((sample_positive.shape[0], 1), dtype=np.int64)
    sample_positive = np.concatenate((sample_positive, ones), axis=1, dtype=np.int64)

    # == END PROCESS POSITIVE PATCHES ==
    # == PROCESS NEGATIVE PATCHES ==
    # get positive patches
    negative = np.logical_and(edge_map == 255, annotation_edge_map == 0)
    negative_coords = np.argwhere(negative)
    # ensure all x,y coordinates are at least 14 pixels away from the edge
    negative_coords = negative_coords[(negative_coords[:, 0] > 14) & (negative_coords[:, 0] < image_gray.shape[0] - 14) & (negative_coords[:, 1] > 14) & (negative_coords[:, 1] < image_gray.shape[1] - 14)]
    np.random.shuffle(negative_coords)

    sample_negative = negative_coords[:negative_patches]
    zeros = np.zeros((sample_negative.shape[0], 1), dtype=np.int64)
    sample_negative = np.concatenate((sample_negative, zeros), axis=1, dtype=np.int64)
    # == END PROCESS NEGATIVE PATCHES ==
    
    # combine positive and negative patches
    data = np.concatenate((sample_positive, sample_negative))

    if data.shape[0] != 300:
        print(f"Error: {image_dir} does not have 300 patches, data.shape: {data.shape} - Skipping...")
        print(f"Positive patches: {sample_positive.shape[0]}, Negative patches: {sample_negative.shape[0]}")
        return None
    
    # data.shape: (300, 3), where 3 is (y, x, label)
    # iterate over data and extract patches
    patches = np.zeros((data.shape[0], 3, patch_size, patch_size), dtype=np.float64)
    labels = np.zeros((data.shape[0], 1), dtype=np.int64)
    for i in range(data.shape[0]):
        y, x, label = data[i]
        patch = np.asarray(image[y - 14:y + 15, x - 14:x + 15], dtype=np.float64)
        # normalise
        patch /= 255
        
        patches[i] = patch.T
        labels[i] = label

    return patches, labels

class GeoPoseDatasetFromDiskByImageDirs(Dataset):
    def __init__(self, dataset_path, image_dirs) -> None:
        super().__init__()
        self.image_dirs = [os.path.join(dataset_path, image_dir) for image_dir in image_dirs] 

    def __len__(self):
        return len(self.image_dirs)
    
    def __getitem__(self, idx):
        image_dir = self.image_dirs[idx]
        patches = np.load(os.path.join(image_dir, "patches.npy"))
        labels = np.load(os.path.join(image_dir, "labels.npy"))
        labels = torch.Tensor(labels)
        patches = torch.Tensor(patches)
        return patches, labels, image_dir


class GeoposeDatasetFromDisk(Dataset):
    """
    Dataset will load patches from disk, where each image directory contains the following files:
    - patches.npy
    - labels.npy

    patches.npy contains the patches extracted from the image
    labels.npy contains the labels for the patches

    Each have a shape of (300, 3, 28, 28) and (300, 1) respectively
    """
    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path
        self.image_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path)]
        
    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        image_dir = self.image_dirs[idx]
        patches = np.load(os.path.join(image_dir, "patches.npy"))
        labels = np.load(os.path.join(image_dir, "labels.npy"))
        labels = torch.Tensor(labels)
        patches = torch.Tensor(patches)
        return patches, labels, image_dir
    

class GeoposeLoader(DataLoader):
    def __init__(self, dataset, shuffle=False):
        self.dataset: GeoposeDatasetFromDisk = dataset
        self.shuffle = shuffle

    def __iter__(self):
        self.idx = 0
        self.dataset_len = len(self.dataset)
        self.indices = np.arange(self.dataset_len)
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.idx >= self.dataset_len:
            raise StopIteration
        idx = self.indices[self.idx]
        patches, labels, image_dir = self.dataset[idx]
        self.idx += 1
        return patches, labels.squeeze(), image_dir
    
    def __len__(self):
        return self.dataset_len