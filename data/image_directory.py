import os 

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename : str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def scan_directory(dir : str):
    image_files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                fpath = os.path.join(root, fname)
                fpath = os.path.abspath(fpath)
                image_files.append(fpath)
    
    # sort the file names
    sorted_files = sorted(image_files)
    return sorted_files
