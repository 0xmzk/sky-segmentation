from dataclasses import dataclass, field
from typing import Any, Callable, Generator
import cv2
import numpy as np
from skimage.morphology import skeletonize
from cv2.typing import Size, MatLike
import torch
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from numpy._typing import _ShapeLike

import networkx as nx
from torch import nn
from torch.nn import functional as F

# ====== MODEL DEFINITION ======


class SkylineCNN(nn.Module):
    def __init__(self):
        super(SkylineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=6, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(50, 500, kernel_size=4, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv4 = nn.Conv2d(500, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Layer 1 Conv
        x = self.conv1(x)
        # Layer 2 Pool (max)
        x = self.pool1(x)
        # Layer 3 Conv
        x = self.conv2(x)
        # Layer 4 Pool (max)
        x = self.pool2(x)
        # Layer 5 Conv
        x = self.conv3(x)
        # Layer 6 ReLU
        x = self.relu(x)
        # Layer 7 Conv
        x = self.conv4(x)
        # Layer 8 Softmax
        x = F.softmax(x, dim=1)
        return x

# ======= SKYLINE SEARCH CLASSES =======


class CostFunctions:
    @staticmethod
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

@dataclass
class FindSkylineParams:
    start_attach_range: int = 20
    max_attach_range_increments: int = 1
    attach_range_increment_fn: Callable[[int], int] = field(default=lambda attach_range: attach_range * 2)

class SkylineSearch:
    @staticmethod
    def construct_graph(binary_image: np.ndarray, cost_fn=CostFunctions.manhattan_distance, eps=1) -> nx.DiGraph:
        """
        Construct a graph from a binary image.

        Funciton iterates over every coordinate in the binary image and connects it to 
        coordinates in the next columns within the epsilon range. 

        For a given coordinate (x, y), and epsilon, eps, the function will connect (x, y) to
        all coordinates in the next eps columns. i.e. for eps=2, (x, y) will be connected to
        all coordinates in columns x+1 and x+2.

        Parameters
        ----------
        binary_image : np.ndarray of shape (H, W) - must be a binary image
            The binary image from which to construct the graph.
        cost_f : function, optional
            The cost function to use when calculating the edge weights between nodes
            in the graph. The default is manhattan_distance.
        eps : int, optional
            eps, epsilon. 
            The epsilon range to consider when connecting nodes in the graph. The default is 1.
        """

        if eps < 1:
            raise ValueError("Epsilon must be greater than 0")

        # init graph
        G = nx.DiGraph()

        # morph binary_image into coordinate list
        y_coords, x_coords = np.where(binary_image == 1)
        coords = np.column_stack((x_coords, y_coords))

        # sort by ascending x
        coords = coords[np.argsort(coords[:, 0])]

        # go over each column by x
        for x in coords[:, 0]:
            # get all coordinaes (x,y) in current column
            coords_in_current_col = coords[coords[:, 0] == x]
            # go over all the coordinates in the current column
            for x1, y1 in coords_in_current_col:
                # add the current (x,y) as a node to the graph
                coord_hashable = tuple((x1, y1))
                G.add_node(coord_hashable)
                # keep a track of the nodes that are added within the epsilon range
                connected_nodes_counter = 0
                # go over the epsilon range
                for e in range(1, eps + 1):
                    # stop iterating if we are at the end of the image
                    if x + e > len(coords[:, 0]):
                        break
                    # get the (x,y) coordinates in the x+epsilon column
                    coords_in_next_col = coords[coords[:, 0] == x + e]
                    # if there are no coordinates then go to the next column
                    if len(coords_in_next_col) < 1:
                        continue
                    # go over any coordinates in that column
                    for next_coord in coords_in_next_col:
                            # calculate cost between (x1,y1) and next_coords
                            # and add an edge connecting the nodes 
                            next_coord_hashable = tuple(next_coord)
                            cost = cost_fn((x1,y1), next_coord)
                            G.add_edge(coord_hashable,
                                    next_coord_hashable, weight=cost)
                            connected_nodes_counter += 1
                # if no nodes were added within the epsilon range
                # we will find the next non empty column to connect with
                if connected_nodes_counter < 1:
                    # get the next non empty columns greater than current x
                    next_non_empty_cols = coords[coords[:, 0] > x]
                    # if non were found then we are at the end of the image
                    if len(next_non_empty_cols) < 1:
                        break  
                    # else we get the first next non empty column
                    # and all the (x,y) coordinates in it
                    next_non_empty_col = coords[coords[:, 0] == next_non_empty_cols[0][0]]
                    # iterate over those coordinates
                    for next_coord in next_non_empty_col:
                        # again, for each coordinate we calcualte cose
                        # and then add an edge connecting the two nodes
                        next_coord_hashable = tuple(next_coord)
                        cost = cost_fn((x1,y1), next_coord)
                        G.add_edge(coord_hashable,
                                    next_coord_hashable, weight=cost)

        return G

    @staticmethod
    def add_virtual_nodes(G: nx.DiGraph, attach_range=10) -> tuple[nx.DiGraph, tuple, tuple]:
        """
        Add virtual nodes to the graph to represent the start and end points of the skyline search

        Idea from (Lie et al., 2004)
        "(2) Two virtual vertices, s and t, are added in the front and rear end, respectively, of the graph to represent the 0th and the (N + 1)th stages." 
        """

        nodes = [k for k, _ in G.nodes(data=True)]
        nodes_by_x = [x for x, _ in nodes if x > 0] # x > 0 to not include nodes, s and t.
        min_x = nodes_by_x[:attach_range]
        max_x = nodes_by_x[-attach_range:]
        min_nodes = [n for n in nodes if n[0] in min_x]
        max_nodes = [n for n in nodes if n[0] in max_x]

        s = (-1, -1)
        t = (-2, -2)

        for node in min_nodes:
            G.add_edge(s, node, weight=0)
        for node in max_nodes:
            G.add_edge(node, t, weight=0)

        return G, s, t
    
    @staticmethod
    def find_skyline(G: nx.DiGraph,
                     params: FindSkylineParams,
                     include_virtual_nodes_in_output=False,
                     ) -> NDArray[np.int64]:
        
        attach_range = params.start_attach_range
        max_attach_range_increments = params.max_attach_range_increments
        attach_range_increment_fn = params.attach_range_increment_fn
        
        G, s, t = SkylineSearch.add_virtual_nodes(G, attach_range=attach_range)
        attach_range_increment_counter = 0 
        path = None
        while attach_range_increment_counter <= max_attach_range_increments:
            try:
                path = nx.shortest_path(G, source=s, target=t, weight='weight')
                break
            except nx.NetworkXNoPath:
                attach_range_increment_counter += 1
                attach_range = attach_range_increment_fn(attach_range)
                G, s, t = SkylineSearch.add_virtual_nodes(G, attach_range=attach_range)
        if (path is None):
            raise nx.NetworkXNoPath()
        if len(path[1:-1]) < 2:
            raise nx.NetworkXNoPath()
        path = np.array(path, dtype=np.int64)
        if include_virtual_nodes_in_output:
            return path
        return path[1:-1]


    @staticmethod
    def find_skyline_legacy(G: nx.DiGraph,
                     s: tuple[int, int],
                     t: tuple[int, int],
                     include_virtual_nodes=False,
                     ) -> NDArray[np.int64]:
        """
        Given a graph of possible skyline points, and virtual nodes s and t, find the shortest path

        This will return a numpy array of shape (n, 2) where n is the number of points in the shortest path
        """
        path = nx.shortest_path(G, source=s, target=t, weight='weight')
        path = np.array(path, dtype=np.int64)
        if not include_virtual_nodes:
            path = path[1:-1]
        return path


# ======= IMAGE PROCESSING FUNCTIONS =======


class SkylinePostprocessor:
    @staticmethod
    def downfill_skyline(skyline_img: NDArray[np.uint8], fill_sides=False):
        """
        Downfill the skyline image to ensure that the skyline is continuous

        This function mutates the skyline image in place, if you want to keep the original image 
        you should pass a copy i..e skyline.copy()
        """
        skyline_points = np.where(skyline_img == 255)
        if len(skyline_points) < 1:
            raise RuntimeError("There are no skyline points in this binary image")
        for x in skyline_points[1]:
            y = np.where(skyline_img[:, x] == 255)[0][0]
            skyline_img[y:,x] = 255

        fill_sides = True
        if fill_sides:
            y,x = skyline_points
            skyline_x_min = np.argmin(x)
            skyline_x_max = np.argmax(x)
            leftmost_x, leftmost_y = x[skyline_x_min], y[skyline_x_min] 
            rightmost_x, rightmost_y = x[skyline_x_max], y[skyline_x_max]
            skyline_img[leftmost_y:, :leftmost_x ] = 255
            skyline_img[rightmost_y:, rightmost_x: ] = 255
            
        return skyline_img

    @staticmethod
    def skyline_search(skeleton: NDArray[np.uint8], eps=5) -> NDArray[np.int64]:
        """
        Given a skeletonised version of the Frajberg Patch CNN output, find the shortest path 
        from the left to the right of the image.

        This search assumes that the skyline is continuous across the image. 
        """
        G = SkylineSearch.construct_graph(skeleton, eps=eps)
        G, s, t = SkylineSearch.add_virtual_nodes(G)
        shortest_path = SkylineSearch.find_skyline_legacy(G, s, t)
        return shortest_path
    
    @staticmethod
    def convert_shortest_path_to_image(shortest_path: NDArray, dim: _ShapeLike) -> NDArray[np.uint8]:
        """
        Given a shortest path, convert it to an image. This does not modify the shortest_path provided.
        """
        img = np.zeros(dim, dtype=np.uint8)
        for point in range(shortest_path.shape[0] - 1):
            p1 = shortest_path[point]
            p2 = shortest_path[point + 1]
            cv2.line(img, p1, p2, 255, 1)
        return img

# ======= PATCH EXTRACTOR CLASSES =======


class BasePatchExtractor(ABC):
    def __init__(self, img: MatLike, total_patches: int, batch_size=300) -> None:
        self.img = img
        self.batch_size = batch_size
        self.original_img_dim = img.shape
        self.output_dim = (img.shape[0] - 28, img.shape[1] - 28)
        self.total_patches = total_patches

    def __len__(self):
        return self.total_patches

    @abstractmethod
    def process_output(self, outputs: list[torch.Tensor], return_probability_map=False) -> NDArray[np.uint8]:
        pass

    @abstractmethod
    def get_extractor(self) -> Generator[NDArray[np.float32], Any, None]:
        pass


class PatchExtractorPostprocess:
    @staticmethod
    def merge_and_reshape_outputs(cnn_outputs: list[torch.Tensor]) -> torch.Tensor:
        # outputs is a list of tensorts of size (batch_size, 2, 1, 1)
        # we merge the tensors to a single tensor of size (n, 2, 1, 1)
        merged = torch.cat(cnn_outputs, dim=0).cpu()
        # we reshape the tensor to (n, 2)
        merged = merged.view(-1, 2)
        return merged

    @staticmethod
    def post_process(
        extractor: BasePatchExtractor,
        outputs: list[torch.Tensor],
        return_probability_map=False
    ) -> NDArray[np.uint8] | NDArray[np.float64]:
        """
        Post-process outputs for a single image from the model

        Parameters:

        extractor: PatchExtractor
            The PatchExtractor instance that was used to extract the patches

        outputs: list[torch.Tensor]
            The outputs from the model - a list of tensors of size (batch_size, 2, 1, 1)

        return_probability_map: bool
            If True, the output will be the probability of each pixel belonging to the skyline
            If False, the output will be the class with the highest probability

        return: np.ndarray
            The output image
        """
        merged = PatchExtractorPostprocess.merge_and_reshape_outputs(outputs)

        # since the output is a probability distribution, we find the probability of
        # the pixel belonging to the skyline, which is the second element of the tensor
        if return_probability_map:
            merged = merged[:, 1]
        # else we just return the class with the highest probability
        else:
            merged = torch.argmax(merged, dim=1)

        # ensure outputs from the model match dimensions of the input image
        if merged.shape[0] != extractor.total_patches:
            raise ValueError(
                "The number of outputs from the model does not match the number of patches extracted from the image")

        # reshape the output to the original image dimensions
        merged = merged.reshape(extractor.output_dim)

        # convert to uint8
        if not return_probability_map:
            converted: NDArray[np.uint8] = merged.numpy().astype(np.uint8)
            return converted

        return merged.numpy().astype(np.float64)


class PixelWisePatchExtractor(BasePatchExtractor):
    def __init__(self, img, batch_size=300) -> None:
        """
        Given an image of shape (H, W, 3), this class will extract patches of size (29, 29, 3).
        The patches will be extracted pixel-wise, i.e. each pixel in the image will be the center of a patch.
        The image reconstructed from the patches will be of shape (H-28, W-28).

        img : np.ndarray
            The image from which patches will be extracted

        batch_size : int
            The number of patches to be extracted per batch 
        """
        total_patches = (img.shape[0] - 28) * (img.shape[1] - 28)
        super().__init__(
            img=img,
            batch_size=batch_size,
            total_patches=total_patches
        )

    def get_extractor(self):
        return self.__extract_patches_generator()

    def __extract_patches_generator(self):
        patches_per_batch = self.batch_size
        # Initialize an empty list to store the current batch's patches
        current_batch = np.zeros(
            (self.batch_size, 3, 29, 29), dtype=np.float32)

        img = np.array(self.img, dtype=np.int64)

        batch_idx = 0
        for i in range(14, img.shape[0] - 14):
            for j in range(14, img.shape[1] - 14):
                patch = img[i - 14:i + 15, j - 14:j + 15]
                patch = patch / 255
                patch = patch.T
                current_batch[batch_idx] = patch
                batch_idx += 1
                # If the current batch is full, yield it
                if batch_idx == patches_per_batch:
                    yield current_batch
                    batch_idx = 0

        # If there are any remaining patches, yield them
        if batch_idx > 0:
            yield current_batch[:batch_idx]

    def process_output(self, outputs: list[torch.Tensor], return_probability_map=False):
        return PatchExtractorPostprocess.post_process(self, outputs, return_probability_map)


class ResizePatchExtractor(PixelWisePatchExtractor):
    def __init__(self, img, resize_dim: Size, return_image_resized=True, batch_size=300) -> None:
        """
        Given an image of shape (H, W, 3), this class will extract patches of size (29, 29, 3).

        The patches will be extracted pixel-wise, i.e. each pixel in the image will be the center of a patch.
        The image reconstructed from the patches will be of shape (H-28, W-28).

        img: np.ndarray
            The image from which patches will be extracted

        resize_dim: tuple (int, int)
            The dimensions to which the image will be resized

        return_image_resized: bool
            If True, the output image will be resized (via cv2.INTER_LINEAR) to the original image dimensions 
            after processing. i.e. (H, W)
            If False, the output image will be of shape resize_dim

        batch_size: int
            The number of patches to be extracted per batch
        """
        self.resize_dim = resize_dim
        self.return_image_resized = return_image_resized
        original_img_dim = img.shape
        _img = img
        # check if dimensions already match
        if not img.shape[0] == resize_dim[0] and not img.shape[1] == resize_dim[1]:
            _img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_NEAREST)
        super().__init__(
            img=_img,
            batch_size=batch_size
        )
        self.original_img_dim = original_img_dim

    def process_output(self, outputs: list[torch.Tensor], return_probability_map=False):
        output = PatchExtractorPostprocess.post_process(
            self, outputs, return_probability_map)
        if self.return_image_resized:
            output = cv2.resize(output, (
                self.original_img_dim[1], self.original_img_dim[0]
            ), interpolation=cv2.INTER_CUBIC)
        return output


class CannyPatchExtractor(BasePatchExtractor):
    def __init__(self, img, batch_size=300, t1=20, t2=200) -> None:
        """
        Given an image of shape (H, W, 3), this class will extract patches of size (29, 29, 3).

        The patches will be extracted from the edges of the image using the Canny edge detector.
        The image reconstructed from the patches will be of shape (H, W).

        img: np.ndarray
            The image from which patches will be extracted

        batch_size: int
            The number of patches to be extracted per batch

        t1: int
            The first threshold for the Canny edge detector

        t2: int 
            The second threshold for the Canny edge detector
        """
        edge_map: np.ndarray = cv2.Canny(img, t1, t2)
        edge_xy = np.stack(np.where(edge_map > 0)).T
        # remove (x,y) entries that are too close to the image border (14 pixels)
        edge_xy = edge_xy[
            (edge_xy[:, 0] > 14) & (edge_xy[:, 0] < img.shape[0] - 14) &
            (edge_xy[:, 1] > 14) & (edge_xy[:, 1] < img.shape[1] - 14)
        ]
        total_patches = edge_xy.shape[0]
        super().__init__(img, total_patches, batch_size)
        self.edge_xy = edge_xy

    def get_extractor(self):
        return self.__extract_patches_generator()

    def __extract_patches_generator(self):
        current_batch = np.zeros(
            (self.batch_size, 3, 29, 29), dtype=np.float32)

        img = np.array(self.img, dtype=np.int64)

        batch_idx = 0
        for i in range(self.total_patches):
            y, x = self.edge_xy[i]
            patch = img[y - 14:y + 15, x - 14:x + 15]
            patch = patch / 255
            patch = patch.T
            current_batch[batch_idx] = patch
            batch_idx += 1

            if batch_idx == self.batch_size:
                yield current_batch
                # check if remaining patches are less than the batch size
                bs = min(self.batch_size, self.total_patches - i - 1)
                # if it is the case, then reduce the size of the next batch
                if bs < self.batch_size:
                    current_batch = np.zeros((bs, 3, 29, 29), dtype=np.float32)
                batch_idx = 0

        if batch_idx > 0:
            yield current_batch[:batch_idx]

    def process_output(self, outputs: list[torch.Tensor], return_probability_map=False):
        merged = PatchExtractorPostprocess.merge_and_reshape_outputs(outputs)

        # since the output is a probability distribution, we find the probability of
        # the pixel belonging to the skyline, which is the second element of the tensor
        if return_probability_map:
            merged = merged[:, 1]
        # else we just return the class with the highest probability
        else:
            merged = torch.argmax(merged, dim=1)

        # ensure outputs from the model match dimensions of the input image
        if merged.shape[0] != self.total_patches:
            raise ValueError(
                "The number of outputs from the model does not match the number of patches extracted from the image")

        # reshape the output to the original image dimensions
        # because canny is not ordered pixel-wise, we need to map the outputs back to the original image
        # we do this by using the edge_xy array
        output = np.zeros(
            (self.original_img_dim[0], self.original_img_dim[1]), dtype=np.uint8 if not return_probability_map else np.float64)
        for i in range(self.total_patches):
            y, x = self.edge_xy[i]
            output[y, x] = merged[i].item()

        return output
    
class NeighbourhoodPatchExtractor(BasePatchExtractor):
    pass

# ======= END OF PATCH EXTRACTOR CLASSES =======

# ======= PIPELINE CLASSES =======


class SkylinePipeline:
    def __init__(self,
                 model: nn.Module,
                 patch_extractor: BasePatchExtractor,
                 device: torch.device,
                 eps: int,
                 skyline_search_params: FindSkylineParams,
                 ) -> None:
        self.model = model
        self.patch_extractor = patch_extractor
        self.device = device
        self.skyline_search_params = skyline_search_params
        self.eps = eps
        self.model.to(self.device)

    def run_inference(self, return_probability_map=False) -> NDArray[np.uint8]:
        """
            Will run the patches from the provided patch extraction in the constructor
            and output the output of the CNN, either a probability map or a binary image.
        """
        self.model.eval()
        outputs: list[torch.Tensor] = []
        batches = self.patch_extractor.get_extractor()
        with torch.no_grad():
            for batches in batches:
                batches = torch.Tensor(batches).to(self.device)
                output = self.model(batches)
                outputs.append(output)
        self.processed_output = self.patch_extractor.process_output(
            outputs, return_probability_map)
        return self.processed_output
    
    def downfill_skyline(self, fill_sides=True) -> NDArray[np.uint8]:
        """
        Uses SkylinePostprocessor.downfill_skyline to downfill the skyline image to create 
        a sky-foreground mask. 

        This function does not mutate self and returns the mask.

        Parameters:
            fill_sides - decides whether to fill in gaps on the side of the mask if there are any
        """
        if not hasattr(self, "processed_output"):
            raise AttributeError("The model has not been run yet")
        if not hasattr(self, "skyline_image"):
            raise AttributeError("The skyline has not been found yet")
        img = SkylinePostprocessor.downfill_skyline(
            self.skyline_image.copy(), fill_sides=fill_sides)
        return img

    def __canny_find_skyline_path(self, eps: int | None = None) -> NDArray[np.int64]:
        eps = self.eps if eps is None else eps

        # kernelsize as a function of pixel density in the image
        def kernel_size_from_pixel_density_exponential(density,
                                                       max_kernel_size=25,
                                                       min_kernel_size=3,
                                                       scaling_factor=5):
            # Exponential scaling
            kernel_size = min_kernel_size + \
                (max_kernel_size - min_kernel_size) * \
                np.exp(-scaling_factor * density)
            # Round to nearest integer
            kernel_size = int(np.round(kernel_size))
            return kernel_size

        k = kernel_size_from_pixel_density_exponential(
            np.sum(self.processed_output) / self.processed_output.size
            )
        dilated = cv2.dilate(
            self.processed_output, np.ones((k, k), np.uint8), iterations=1
        )

        self.skeleton = skeletonize(dilated)


        G = SkylineSearch.construct_graph(self.skeleton, eps=eps, cost_fn=CostFunctions.euclidean_distance)
        shortest_path = SkylineSearch.find_skyline(G, self.skyline_search_params)

        self.skyline_path = shortest_path
        return self.skyline_path

    def find_skyline_path(self, eps: int | None = None) -> NDArray[np.int64]:
        eps = self.eps if eps is None else eps

        if not hasattr(self, "processed_output"):
            raise AttributeError("The model has not been run yet")

        # Run different find_skyline implementations depending on extractor
        if isinstance(self.patch_extractor, CannyPatchExtractor):
            return self.__canny_find_skyline_path(eps)

        # Else we use the default implementation
        erode = cv2.erode(self.processed_output,
                          np.ones((3, 3), np.uint8), iterations=1)
        if np.sum(erode) == 0:
            raise RuntimeError(
                "Erosion removed all pixels from the output image")

        self.skeleton = skeletonize(erode)

        G = SkylineSearch.construct_graph(self.skeleton, eps=eps)
        shortest_path = SkylineSearch.find_skyline(G, self.skyline_search_params)

        self.skyline_path = shortest_path
        return self.skyline_path

    def convert_skyline_path_to_image(self) -> NDArray[np.uint8]:
        """
        Function will convert a found skyline_path into an image and return it.
        
        self.skyline_image will also contain the result
        """
        if not hasattr(self, "skyline_path"):
            raise AttributeError("The skyline path has not been found yet")
        skyline = SkylinePostprocessor.convert_shortest_path_to_image(
            self.skyline_path, self.processed_output.shape
        )
        self.skyline_image = skyline
        return skyline