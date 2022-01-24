import cv2
import numpy.typing as _npt
import os

__all__ = [
    'shrink_image',
    'shrink_image_at_path',
]

def _get_crop_dims(actual_dim, desired_dim):
    to_crop = (actual_dim - desired_dim) / 2

    # No need to crop if we're already at the right dimension.
    if to_crop == 0:
        return 0, actual_dim

    int_crop = int(to_crop)
    if int_crop == to_crop:
        # We're dealing with a whole number, which means we can crop the same
        # amount from each side.
        crop_high = crop_low = int_crop
    else:
        # We're dealing with a fraction. Round down for the high and round
        # up for the low.
        crop_high = int_crop
        crop_low = int_crop + 1

    return crop_low, actual_dim - crop_high

def shrink_image(
    img: _npt.ArrayLike,
    height: int = None,
    width: int = None,
) -> _npt.ArrayLike :
    """
    Shrink an image to the specified height and width. If both height and width
    are provided, the image is scaled down to the dimension which needs less
    scaling and then the other dimension is cropped to size, centering the
    chosen image in the middle with respect to the remaining dimension.

    Args:
        img (numpy array):
            the input image to be scaled
        height (int):
            the target height for the image (optional)
        width (int):
            the target width for the image  (optional)

    Returns:
        the resized image
    """
    if not any((height, width)):
        raise ValueError('Either height or width must be provided.')

    src_height, src_width, _ = img.shape

    # Determine by what amount each dimension needs to be scaled.
    try:
        height_scale = height / src_height
    except TypeError:
        # No scaled height given, but we know we have a width. Make sure the
        # latter is actually related to shrinkage.
        if width > src_width:
            raise ValueError(
                f'Target width must be less than source width ({src_width}).'
            )

        # Set both scales to the width scale.
        height_scale = width_scale = width / src_width
    else:
        # We were given a target height from which we were able to get a height
        # scale. Is it valid though?
        if height_scale > 1:
            raise ValueError(
                'Target height must be less than source height '
                f'({src_height}).'
            )

        # Has the target width been provided?
        try:
            width_scale = width / src_width
        except TypeError:
            # Nope, no targat width given, so we default to the height scale.
            width_scale = height_scale
            width = int(src_width * width_scale)

    # Determine the final dimensions.
    if width_scale < height_scale:
        # The width requires more scaling than the height to get to the final
        # value. This means we must scale to the desired height (the opposite
        # would mean losing parts of the top and bottom of the image). In the
        # process of scaling to the height, we will be scaling the width an
        # insufficient amount, which means we will need to crop it later on to
        # get to the desired width.
        dx = int(src_width * height_scale)
        dy = height
    else:
        # The opposite is true.
        dx = width
        dy = int(src_height * width_scale)

    # Scale the image.
    img = cv2.resize(img, (dx, dy), interpolation=cv2.INTER_AREA)

    # Prepare for the crop. Only one of these dimensions will be changed.
    height_crop_low = 0
    height_crop_high = img.shape[0]
    width_crop_low = 0
    width_crop_high = img.shape[1]

    if dy == height:
        # We scaled to the height, so we need to crop to the width.
        width_crop_low, width_crop_high = _get_crop_dims(
            width_crop_high,
            width,
        )
    else:
        # We scaled to the width, so we need to crop to the height.
        height_crop_low, height_crop_high = _get_crop_dims(
            height_crop_high,
            height,
        )

    # Crop and center to the desired aspect ratio.
    return img[
        height_crop_low:height_crop_high,
        width_crop_low:width_crop_high,
        :
    ]

def shrink_image_at_path(
    path: str,
    height: int = None,
    width: int = None,
) -> _npt.ArrayLike :
    """
    Wrapper for `shrink_image()`, using a path rather than a numpy array as
    input.

    Args:
        path (string):
            the path to the input image to be scaled
        height (int):
            the target height for the image (optional)
        width (int):
            the target width for the image  (optional)

    Returns:
        the resized image
    """
    if not os.path.isfile(path):
        raise(f'No file exists at {path}.')

    return shrink_image(cv2.imread(path), height, width)
