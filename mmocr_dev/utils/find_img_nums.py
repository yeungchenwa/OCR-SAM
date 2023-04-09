import os


def find_num_imgs(root_dir):
    """Given a root directory, find the number of images in the directory

    Args:
        root_dir (str): root directory 
        The directory should be in the following format:
            root_dir
                |--folder1
                    |--image1.jpg
                    |--image2.jpg
                    |--...
                |--folder2
                    ...
    """
    num_imgs = 0
    for folder in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, folder)):
            continue
        for img in os.listdir(os.path.join(root_dir, folder)):
            if img.endswith(".jpg"):
                num_imgs += 1
    return num_imgs


if __name__ == "__main__":
    print(
        find_num_imgs(
            "/media/jiangqing/jqnas/projects/TextCLIP/data/LAION400M/data/part-00003"
        ))
