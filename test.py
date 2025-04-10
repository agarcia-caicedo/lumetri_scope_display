"""Module to convert an image to show up in a lumetri scope.

Done out of curiosity and for fun. This code isn't clean or efficient, but it
should be documented enough to get you started if you want to tinker with it <3
"""
import colorsys
import os
from pathlib import Path
import pathlib
import numpy as np
import random
import time
from functools import partial
from multiprocessing import Pool
import math

from PIL import Image, ImageOps

import dither


def image_to_waveform_rgb(
    input_path,
    output_path,
    output_width=1920,
    output_height=1080,
    style=None,
    dither_input=True,
    color_threshold=160,
    **kwargs
):
    """Convert an image to show up in a Waveform RGB display.

    Args:
        input_path (str): Image to process.
        output_path (str): Path where the output image should be saved to.
        output_width (int): Horizontal resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        output_height (int): Vertical resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        style (str or None): Apply different "styles" to the output image. This
            takes advantage of the fact that the vertical order of pixels
            doesn't matter and can therefore be rearranged. Options are:
            None, "random", "sorted", "rgb_sorted".
        dither_input (bool): Whether the input image should be dithered.
        color_threshold (int): The color intensity at which a pixel is
            considered to contain that color. For example a pixel with rgb of
            (20, 150, 180) will become (0, 0, 255) with a threshold of 160.

    Returns:
        str: Path of the converted image.

    Examples:
        ::

            # Process a single image to Waveform
            output_image = image_to_waveform_rgb(
                input_path=r"some\input\image.jpg",
                output_path=r"your\output\waveform_image.png",
                output_width=1920,
                output_height=1080,
                style="rgb_sorted",
                dither_input=True,
                color_threshold=160,
            )
    """
    input_image = Image.open(input_path)

    # Fit the given image into the output resolution.
    image_to_process = ImageOps.contain(
        input_image, (output_width, output_height)
    )
    input_image.close()
    del input_image

    # Dither the image before processing it.
    if dither_input:
        image_to_process = dither.convert(image_to_process)

    # Determine offsets to center the resized image.
    resized_width = image_to_process.width
    resized_height = image_to_process.height
    horizontal_offset = int((output_width - resized_width) / 2)
    vertical_offset = int((output_height - resized_height) / 2)

    # Create an empty output image to fill in the loop.
    output_image = Image.new(
        mode="RGB", size=(output_width, output_height), color=0
    )

    for pixel_horizontal in range(resized_width):
        # Creating a column list makes it easier to apply styles later on.
        current_column = []

        offset_horizontal_index = pixel_horizontal + horizontal_offset
        for pixel_vertical in range(resized_height):
            r, g, b = image_to_process.getpixel((pixel_horizontal, pixel_vertical))
            intensity = int(255.0 * (resized_height - pixel_vertical) / resized_height)
            red = intensity if r > color_threshold else 0
            green = intensity if g > color_threshold else 0
            blue = intensity if b > color_threshold else 0
            current_column.append((red, green, blue))

        # The vertical position of pixels doesn't matter (luminance determines
        # vertical position in lumetri scope). Therefore we can apply different
        # "styles" to the image, by reordering pixels vertically.
        if style and style == "random":
            # Shuffle the pixels randomly
            random.shuffle(current_column)
        elif style and style == "sorted":
            # Sort the pixels as-is in ascending RGB brightness.
            current_column.sort()
        elif style and style == "rgb_sorted":
            # Since the RGB values of a pixel are treated individually in the
            # Waveform display anyways they don't need to remain together.
            # This sorting style takes advantage of this and sorts the RGB
            # channels individually.
            sorted_column = [
                sorted(list(column)) for column in zip(*current_column)
            ]
            current_column = [
                (r, g, b) for r, g, b in zip(*sorted_column)
            ]

        # Store the pixels into the output image.
        for offset_vertical_index, color in enumerate(current_column, vertical_offset):
            output_image.putpixel(
                (offset_horizontal_index, offset_vertical_index), color
            )

    image_to_process.close()
    del image_to_process

    # Store the image in a lossless format.
    output_image.save(os.path.splitext(output_path)[0] + ".png", "PNG")
    output_image.close()
    del output_image

    return output_path



def process_image(
    image,
    input_folder,
    output_folder,
    output_width=1920,
    output_height=1080,
    replace_existing=False,
    **kwargs,
):
    """Process a single image with the given conversion_function.

    Args:
        image (str): Name of the image to process.
        conversion_function (func): The function to use for the conversion.
            Either "image_to_waveform_rgb" or "image_to_vectorscope_hls".
        input_folder (str): Folder with images to process. Note: The script
            will attempt to process ALL files inside this folder.
        output_folder (str): Folder where the output images should be saved to.
        output_width (int): Horizontal resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        output_height (int): Vertical resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        replace_existing (bool): Whether to overwrite existing files.
        kwargs (dict): Keyword arguments to be passed on to the given
            conversion_function.

    Returns:
        str: Path of the converted image.
    """
    output_path = os.path.join(output_folder, image)
    if os.path.isfile(output_path) and not replace_existing:
        print("Skipping {}: Already exists.".format(output_path))
        return output_path

    print("Processing:", image)
    input_path = os.path.join(input_folder, image)
    output_image = image_to_waveform_rgb(
        input_path=input_path,
        output_path=output_path,
        output_width=output_width,
        output_height=output_height,
        **kwargs,
    )

    return output_image


def process_folder(
    input_folder,
    output_folder,
    threads=8,
    replace_existing=False,
    output_width=1920,
    output_height=1080,
    style="rgb_sorted",
    dither_input=True,
    color_threshold=160,
):
    """Process all images in a folder.

    Args:
        conversion_function (func): The function to use for the conversion.
            Either "image_to_waveform_rgb" or "image_to_vectorscope_hls".
        input_folder (str): Folder with images to process. Note: The script
            will attempt to process ALL files inside this folder.
        output_folder (str): Folder where the output images should be saved to.
        output_width (int): Horizontal resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        output_height (int): Vertical resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        replace_existing (bool): Whether to overwrite existing files.
        threads (int): How many threads should be used while processing.
        kwargs (dict): Keyword arguments to be passed on to the given
            conversion_function.

    Examples:
        ::

            # Process an entire folder to Waveform
            process_folder(
                conversion_function=image_to_waveform_rgb,
                input_folder=r"some\input\folder",
                output_folder=r"your\output\folder",
                threads=8,
                replace_existing=False,
                output_width=1920,
                output_height=1080,
                style="rgb_sorted",
                dither_input=True,
                color_threshold=160,
            )

            # Process an entire folder to Vectorscope
            process_folder(
                conversion_function=image_to_vectorscope_hls,
                input_folder=r"some\input\folder",
                output_folder=r"your\output\folder",
                threads=8,
                replace_existing=False,
                output_width=1920,
                output_height=1080,
                style="sorted",
                scale_to_fit=True,
                output_resolution=512,
                shades=8,
                outline_with_remaining_pixels=True,
                maintain_chunk_positions=True,
                chunk_fill_color=None,
            )
    """
    start_time = time.time()

    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    _, _, filenames = list(os.walk(input_folder))[0]
    if __name__ == '__main__':
        with Pool(threads) as pool:
         pool.map(
                partial(
                    process_image,
                    conversion_function=image_to_waveform_rgb,
                    input_folder=input_folder,
                    output_folder=output_folder,
                    output_width=output_width,
                    output_height=output_height,
                    replace_existing=replace_existing,
                    style=style,
                    dither_input=dither_input,
                    color_threshold=color_threshold
                ),
                filenames,
            )

    message = "Took: {} seconds to process {} images with {} threads."
    print(message.format(time.time() - start_time, len(filenames), threads))


process_folder(r"C:\Users\annoy\Videos\Captures\choke",r"C:\Users\annoy\Videos\Captures\choke2",8,False,1920,1080,"rgb_sorted",True,160)