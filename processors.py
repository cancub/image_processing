import cv2
import functools
import numpy.typing as _npt
import os
import string
import typing
import xml.etree.ElementTree as _ET

__all__ = [
    'shrink_image',
    'shrink_image_at_path',
    'img_to_rects_svg',
]

BLUR_ID = 'blur'

# This isn't a great ID to use for XML/HTML, since it could accidentally be
# used somewhere else. Thankfully, we're increasing the entropy by appending
# one more character for each template.
BASE_ID = 'R'

# The possible set of characters to use in identifying specific templates.
AVAILABLE_ID_CHARS = list(string.hexdigits)


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


def _configure_svg_blur(svg, blur):
    source_graphic_name = 'SourceGraphic'
    filter_stage_names = ['blurSquares', 'opaqueBlur']

    # Locate the definitions block which has almost assuredly been added by the
    # template-generating procedure.
    defs = svg.find('defs')

    # Just in case though, make the definition block if it doesn't exist.
    if defs is None:
        defs = _ET.SubElement(svg, 'defs')

    # Add a blur filter.
    blur_filter = _ET.SubElement(
        defs,
        'filter',
        {
            'id': BLUR_ID,
            'x': '0',
            'y': '0',
            'height': '100%',
            'width': '100%',
            'primitiveUnits': 'userSpaceOnUse',
        },
    )

    _ET.SubElement(
        blur_filter,
        'feGaussianBlur',
        {
            'x': '0',
            'y': '0',
            'width': '100%',
            'height': '100%',
            'stdDeviation': str(blur),
            'in': source_graphic_name,
            'result': filter_stage_names[0],
        },
    )

    comp_transfer = _ET.SubElement(
        blur_filter,
        'feComponentTransfer',
        {'in': filter_stage_names[0], 'result': filter_stage_names[1]},
    )
    _ET.SubElement(
        comp_transfer,
        'feFuncA',
        {'type': 'linear', 'intercept': '1'},
    )

    _ET.SubElement(
        blur_filter,
        'feBlend',
        {
            'mode': 'normal',
            'in': filter_stage_names[1],
            'in2': source_graphic_name,
        },
    )

    # Return the group for the <rect>s, setting it to use the above blur.
    return _ET.SubElement(svg, 'g', {'filter': f'url(#{BLUR_ID})'})


@functools.lru_cache
def _index_to_id(index):
    """
    We need to keep the length of the IDs to a minimum. That said, there are
    only so many characters that we can use. In this case we need to work with
    looping and wrapping. to get a clean solution.

    NOTE:
    This function will be called recursively without any changes to underlying
    data, so we speed it up via lru_cache().
    """
    loop_count = 0
    total_chars = len(AVAILABLE_ID_CHARS)

    while True:
        try:
            # Duplicate the character as many time as we have looped.
            char = AVAILABLE_ID_CHARS[index]
        except IndexError:
            # We're (still) past the available set of characters with this
            # index. So wrap the index and increment the count.
            index -= total_chars
            loop_count += 1
        else:
            # We've finally found an ID that falls within the avaibale range.
            break

    if loop_count > 0:
        # We didn't manage to land within the acceptable range of IDs on the
        # first attempt. This means we must add another character so as not to
        # not collide with the ID associated with the non-modulus index. For
        # example, if
        #   1 := "a"
        # and
        #   300 - total_chars := "a"
        # then we need to add a character to the 300 ID to make it distinct
        # from the 1 ID. We can maximize the space saved by recursively
        # calling this same function, using the number of times we looped as
        # the argument.
        # NOTE:
        # Decrement by one so that we are able to use all N characters, rather
        # than starting at index=loop_count=1 and skipping the first character.
        char = _index_to_id(loop_count-1) + char

    return char


def _pixel_to_use_element(img, x_pos, y_pos):
    # Colour order is a bit wonky in image files loaded by OpenCV.
    b,g,r = img[y_pos, x_pos, :]

    attrib = {
        'href': f'#{BASE_ID}{_index_to_id(y_pos)}',
        'fill': f'rgb({r},{g},{b})',
    }

    if x_pos > 0:
        attrib['x'] = str(x_pos)

    return _ET.Element('use', attrib)


def img_to_rects_svg(
    img: _npt.ArrayLike,
    scale: typing.Union[float, int] = 1,
    blur: typing.Union[float, int] = None,
) -> _ET.Element:
    """
    Convert an image (reprsented by a numpy array) to an SVG of <rect>
    elements, optionally performing some scaling and blurring of the image.

    Args:
        img (numpy array):
            the image to be converted to an SVG
        scale (float or int, optional):
            the scaling to be applied to the image prior to conversion
        blur (float or int, optinal):
            the standard deviation of the Gaussian blur to apply to the SVG
            after the conversion is complete

    Returns:
        the SVG as an ElementTree.Element
    """
    if scale != 1:
        # NOTE:
        # This is just a straight scaling, so we can avoid all the extra
        # processing of shrink_image() by skipping to the chase.
        img = cv2.resize(
            img,
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    height, width, _ = img.shape

    # Build the overall <svg> container element.
    svg = _ET.Element(
        'svg',
        {
            'viewbox': f'0 0 {width} {height}',
            'width': str(width),
            'height': str(height),
            'xmlns':'http://www.w3.org/2000/svg',
        },
    )

    # The ID for the OG <rect> template that the <use> templates will also
    # reference.
    root_id = f'{BASE_ID}{_index_to_id(0)}'

    # Regardless of whether or not we're blurring the image, we will need a
    # definitions block to store the various templates.
    defs = _ET.SubElement(svg, 'defs')

    # Add the original template to the definitions.
    _ET.SubElement(
        defs,
        'rect',
        {
            'id': root_id,
            'width': '1',
            'height': '1',
        }
    )

    # Get rid of the x attributes in <use> elemements by adding templates for
    # each of the individual rows.
    # NOTE:
    # Work with templates for pixels in rows rather templates for pixels in
    # columns, because there are more columns than rows, ergo we have more
    # savings.
    # Also, we can skip the first row, since it's already handled by the
    # original <rect>.
    for y in range(1, height):
        _ET.SubElement(
            defs,
            'use',
            {
                'y': str(y),
                'href': f'#{root_id}',
                'id': f'{BASE_ID}{_index_to_id(y)}',
            }
        )

    if blur is None:
        # No blur means that the rectangles will be inserted directly under the
        # <svg> element.
        rects_container = svg
    else:
        # The rectangles will need to be placed into a <g> group element to
        # apply the blur to them.
        rects_container = _configure_svg_blur(svg, blur)

    # Convert the pixels of the image and their colours to <rect> elements.
    for y in range(height):
        for x in range(width):
            rects_container.append(_pixel_to_use_element(img, x, y))

    return svg
