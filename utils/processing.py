"""
Helpful functions for processing data.
"""
import os
import re
import imageio


def extract_number(filename):
    match = re.search(r"image_(\d+).png", filename)
    if match:
        return int(match.group(1))
    return 0


def get_sorted_images(directory):
    filenames = [
        f
        for f in os.listdir(directory)
        if f.startswith("image_") and f.endswith(".png")
    ]
    sorted_filenames = sorted(filenames, key=extract_number)
    return sorted_filenames


def create_mp4_from_pngs(png_dir, mp4_path, fps=30):
    """
    Create mp4 video from pngs in a directory. Creates
    directory if it doesn't exist.
    """
    images = []
    filenames = get_sorted_images(png_dir)
    for filename in filenames:
        file_path = os.path.join(png_dir, filename)
        images.append(imageio.imread(file_path))

    if not os.path.exists(os.path.dirname(mp4_path)):
        os.makedirs(os.path.dirname(mp4_path))
    imageio.mimsave(mp4_path, images, fps=fps)
