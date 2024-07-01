import glob
import re

import os.path as osp

from dataset.base_image_dataset import BaseImageDataset


class CelebreID(BaseImageDataset):
    """
    Reference:
    Celebrities-ReID: A Benchmark for Clothes Variation in Long-Term Person Re-Identification
    Beyond Scalar Neuron: Adopting Vector-Neuron Capsules for Long-Term Person Re-Identification
    URL: https://github.com/Huang-3/Celeb-reID

    Dataset statistics:
    split | Training |     Testing     | total
    ---------------------------------------------
    subsets| Training | query | gallery | total
    ---------------------------------------------
    #ID    |   632    |  420  |   420   | 1,052
    ---------------------------------------------
    #Image |  20,208  | 2,972 |  11,006 | 34,186

    The resolution of each image is 128*256.

    The meaning of name of each image:
    For example "x_y_z.jpg", "x" represents ID, "y" represents y-th image, "z" is meaningless...
    """

    def __init__(self, dataset_root='data', dataset_filename='Celeb-reID', verbose=True, **kwargs):
        super(CelebreID, self).__init__()
        self.dataset_dir = osp.join(dataset_root, dataset_filename)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self.check_before_run(required_files=[self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir])

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Celeb-reID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        camid = 0
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            assert 1 <= pid <= 632
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
            camid += 1  # assume camids are different for all images, not used
        return dataset
