import skimage
from pathlib import Path


#####
#
# General function to load the data
#
#####


def load(folder_path, return_paths=False):
    folder = Path(folder_path)

    # numbered as 0000.tif, 0001.tif so sorting should give correct time sequence
    tif_list = sorted(folder.glob("*.tif"))

    # check if there is actually some data loaded
    assert len(tif_list) > 0, f"No .tif files found in folder: {folder}"

    img_list = [skimage.io.imread(str(p)) for p in tif_list]

    if return_paths:
        return img_list, tif_list
    return img_list