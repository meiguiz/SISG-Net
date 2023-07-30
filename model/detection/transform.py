import math

import numpy as np
import torch
import torchvision

from torch import nn, Tensor
from typing import List, Tuple, Dict, Optional

from .image_list import ImageList
from .roi_heads import paste_masks_in_image
import torch.nn.functional as F
import cv2
@torch.jit.unused
def _get_shape_onnx(image):
    # type: (Tensor) -> Tensor
    from torch.onnx import operators
    return operators.shape_as_tensor(image)[-2:]


@torch.jit.unused
def _fake_cast_onnx(v):
    # type: (Tensor) -> float
    # ONNX requires a tensor but here we fake its type for JIT.
    return v


def _resize_image_and_masks(image: Tensor, self_min_size: float, self_max_size: float,
                            target: Optional[Dict[str, Tensor]] = None,
                            fixed_size: Optional[Tuple[int, int]] = None,
                            ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    if torchvision._is_tracing():
        im_shape = _get_shape_onnx(image)
    else:
        im_shape = torch.tensor(image.shape[-2:])

    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None
    if fixed_size is not None:
        size = [fixed_size[1], fixed_size[0]]
    else:
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(self_min_size / min_size, self_max_size / max_size)

        if torchvision._is_tracing():
            scale_factor = _fake_cast_onnx(scale)
        else:
            scale_factor = scale.item()
        recompute_scale_factor = True

    image = torch.nn.functional.interpolate(image[None], size=size, scale_factor=scale_factor, mode='bilinear',
                                            recompute_scale_factor=recompute_scale_factor, align_corners=False)[0]

    if target is None:
        return image, target

    if "masks" in target:
        mask = target["masks"]
        mask = torch.nn.functional.interpolate(mask[:, None].float(), size=size, scale_factor=scale_factor,
                                               recompute_scale_factor=recompute_scale_factor)[:, 0].byte()
        target["masks"] = mask
    return image, target


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size, image_mean, image_std, size_divisible=32, fixed_size=None):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisible = size_divisible
        self.fixed_size = fixed_size

    def forward(self,
                images,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image):
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        # type: (List[int]) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self,
               image: Tensor,
               target: Optional[Dict[str, Tensor]] = None,
               ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])
        image, target = _resize_image_and_masks(image, size, float(self.max_size), target, self.fixed_size)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image, target

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)

        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self,
                    result,               # type: List[Dict[str, Tensor]]
                    image_shapes,         # type: List[Tuple[int, int]]
                    original_image_sizes  # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        if self.training:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            fmasks_1 = result[0]["masks"]
            fmasks_2 = result[1]["masks"]
            boxes_1 = result[0]["boxes"]
            boxes_1 = resize_boxes(boxes_1, image_shapes, original_image_sizes)
            boxes_2 = result[1]["boxes"]
            boxes_2 = resize_boxes(boxes_2, image_shapes, original_image_sizes)
            masks_1 = paste_masks_in_image(fmasks_1, boxes_1, original_image_sizes)
            masks_2 = paste_masks_in_image(fmasks_2, boxes_2, original_image_sizes)

            feature_map_num_1 = masks_1.shape[0]
            feature_map_num_2 = masks_2.shape[0]

            feature_1 = torch.zeros(1, 1, 384, 512).to(device)
            feature_2 = torch.zeros(1, 1, 384, 512).to(device)
            for index in range(feature_map_num_1):  # 通过遍历的方式，将64个通道的tensor拿出

                feature = masks_1[index]
                # feature_3 = feature.detach().cpu().numpy()
                # feature_3 = np.asarray(feature_3*255,dtype=np.uint8)
                # feature_3 = feature_3.squeeze()
                # feature_2 = cv2.resize(feature_3, (28, 28), interpolation=cv2.INTER_NEAREST)
                # #feature = np.asarray(feature * 255, dtype=np.uint8)
                # feature_2 = cv2.applyColorMap(feature_2, cv2.COLORMAP_WINTER)  # 变成伪彩图
                # cv2.imwrite('/home/meiguiz/下载/AFFGA-Net-main (1)/demo/feature_map/channel_{}.png'.format(str(index)), feature_2)

                feature = feature.unsqueeze(0)

                feature_1 = torch.add(feature_1, feature)
            feature_1 = F.interpolate(feature_1, size=(24, 32), mode='bilinear', align_corners=True)
            # feature_3 = feature_1.detach().cpu().numpy()
            # feature_3 = np.asarray(feature_3 * 255, dtype=np.uint8)
            # feature_3 = feature_3.squeeze()
            # feature_2 = cv2.applyColorMap(feature_3, cv2.COLORMAP_WINTER)
            # cv2.imwrite('/home/meiguiz/下载/AFFGA-Net-main (1)/demo/feature_map/channel.png',
            #             feature_2)
            for index in range(feature_map_num_2):  # 通过遍历的方式，将64个通道的tensor拿出

                feature = masks_2[index]
                # feature_2 = cv2.resize(feature, (28, 28), interpolation=cv2.INTER_NEAREST)
                # #
                # feature_2 = cv2.applyColorMap(feature, cv2.COLORMAP_HSV)  # 变成伪彩图
                # cv2.imwrite('/home/meiguiz/下载/AFFGA-Net-main (1)/demo/feature_map/channel_{}.png'.format(str(index)), feature)

                feature = feature.unsqueeze(0)

                feature_2 = torch.add(feature_2, feature)
            feature_2 = F.interpolate(feature_2, size=(24, 32), mode='bilinear', align_corners=True)
            feature_0 = torch.cat((feature_1,feature_2),dim=0)
            return result, feature_0
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                feature_map_num = masks.shape[0]
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                feature_1 = torch.zeros(1, 1, 384, 512).to(device)
                for index in range(feature_map_num):  # 通过遍历的方式，将64个通道的tensor拿出

                    feature = masks[index]
                    # feature_2 = cv2.resize(feature, (28, 28), interpolation=cv2.INTER_NEAREST)
                    # #
                    # feature_2 = cv2.applyColorMap(feature, cv2.COLORMAP_HSV)  # 变成伪彩图
                    # cv2.imwrite('/home/meiguiz/下载/AFFGA-Net-main (1)/demo/feature_map/channel_{}.png'.format(str(index)), feature)

                    feature = feature.unsqueeze(0)

                    feature_1 = torch.add(feature_1, feature)
                feature_1 = F.interpolate(feature_1, size=(24, 32), mode='bilinear', align_corners=True)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result, feature_1

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                         self.max_size)
        format_string += '\n)'
        return format_string


def resize_keypoints(keypoints, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=keypoints.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=keypoints.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    if torch._C._get_tracing_state():
        resized_data_0 = resized_data[:, :, 0] * ratio_w
        resized_data_1 = resized_data[:, :, 1] * ratio_h
        resized_data = torch.stack((resized_data_0, resized_data_1, resized_data[:, :, 2]), dim=2)
    else:
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
    return resized_data


def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin
    xmax = xmax
    ymin = ymin
    ymax = ymax
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
