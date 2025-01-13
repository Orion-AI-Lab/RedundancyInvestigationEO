from pathlib import Path
from tqdm import tqdm
import numpy as np
import rasterio
import pickle
from rasterio import features
import geopandas as gpd
import xmltodict
from albumentations.augmentations.crops.functional import crop_bbox_by_coords
from shapely.geometry import Polygon, box

from utils.paths import Paths

class Tilerizer:

    def __init__(self, tif_path, annot_path, tiles_path=None, masks_path=None, task='detection'):
        self.tif_path = tif_path
        self.annot_path = annot_path
        self.task = task
        self.metadata, self.tif = self._load_tif()
        self.labels = self._load_annots()
        self._create_folders()

    def _load_tif(self):
        with rasterio.open(self.tif_path) as src:
            metadata = src.profile
            band = src.read()
        return metadata, band

    def _load_annots(self):
        if self.task == 'detection':
            with open(self.annot_path, 'r') as annotation_file:
                annotation = xmltodict.parse(annotation_file.read())
            labels = []
            if isinstance(annotation['annotation']['object'], list):
                for bbox in annotation['annotation']['object']:
                    xmin = bbox['bndbox']['xmin']
                    ymin = bbox['bndbox']['ymin']
                    xmax = bbox['bndbox']['xmax']
                    ymax = bbox['bndbox']['ymax']
                    labels.append([float(xmin), float(ymin), float(xmax), float(ymax)])
            else:
                xmin = annotation['annotation']['object']['bndbox']['xmin']
                ymin = annotation['annotation']['object']['bndbox']['ymin']
                xmax = annotation['annotation']['object']['bndbox']['xmax']
                ymax = annotation['annotation']['object']['bndbox']['ymax']
                labels.append([float(xmin),float(ymin),float(xmax),float(ymax)])
        else:
            raise Exception('Task {} not implemented yet.'.format(self.task))
        return labels                     

    def _create_folders(self):
        self.tiles_folder = self.tif_path.parent / 'tiles' / self.tif_path.stem
        self.tiles_folder.mkdir(parents=True, exist_ok=True)
        self.labels_folder = self.tif_path.parent / 'labels' / self.tif_path.stem
        self.labels_folder.mkdir(parents=True, exist_ok=True)

    def _crop_labels(self, crop_coords):
        if self.task == 'detection':
            x_min, y_min, x_max, y_max = crop_coords
            bbox_polys = np.array([box(*bbox) for bbox in self.labels])
            crop_poly = box(*crop_coords)
            intersect_bboxes = np.array([bbox.intersection(crop_poly).area/bbox.area for bbox in bbox_polys])
            idx_full_bboxes = np.where(intersect_bboxes == 1.)
            idx_partial_boxes = np.where((intersect_bboxes > 0.) & (intersect_bboxes < 1.))[0]
            # Crop boxes with partial intersection
            subboxes = np.array([bbox.intersection(crop_poly) for bbox in bbox_polys[idx_partial_boxes]])
            # Check if all boxes are in the crop
            for bbox in subboxes:
                assert bbox.intersection(crop_poly).area/bbox.area == 1., 'A box has not been cropped well!'
            valid_boxes = np.hstack((bbox_polys[idx_full_bboxes], subboxes))
            # From polygones to boxes
            valid_boxes = [list(box_poly.bounds) for box_poly in valid_boxes]
            norm_valid_boxes = [[valid_box[0]-x_min, valid_box[1]-y_min, valid_box[2]-x_min, valid_box[3]-y_min]
                                for valid_box in valid_boxes]
            return norm_valid_boxes
        else:
            raise Exception('Task {} not implemented yet.'.format(self.task))
        
    def create_tiles(self, tile_size=256):            
        num_rows = self.tif.shape[1]
        num_cols = self.tif.shape[2]
        samples = []
        print('Full area size: ', self.tif.shape[1:])
        print('Desired tile size: ', tile_size)

        print('Saving tiles')

        for row in tqdm(range(0, num_rows, tile_size)):
            for col in tqdm(range(0, num_cols, tile_size)):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)
                tile = self.tif[:, window.row_off:window.row_off+window.height,
                                window.col_off:window.col_off+window.width]
                tile_profile = self.metadata.copy()
                tile_profile.update({
                    'height': tile.shape[1],
                    'width': tile.shape[2],
                    'transform': rasterio.windows.transform(window, self.metadata['transform'])
                })

                tile_name = 'tile_{}_{}.tif'.format(row, col)
                with rasterio.open(self.tiles_folder / tile_name, 'w', **tile_profile) as dst:
                    dst.write(tile)

                crop_coords = [window.col_off, window.row_off, window.col_off+window.width, window.row_off+window.height]
                labels = self._crop_labels(crop_coords)
                label_file_name = 'labels_{}_{}.pkl'.format(row, col)
                with open(self.labels_folder / label_file_name, 'wb') as f:
                    pickle.dump(labels, f)

                sample = {'image': self.tiles_folder / tile_name,
                          'labels': self.labels_folder / label_file_name}
                samples.append(sample)

        paths_file_name = self.tif_path.stem + '_paths.pkl'
        with open(self.tif_path.parent / paths_file_name, 'wb') as f:
            pickle.dump(samples, f)
        return samples

if __name__ == '__main__':
    # Please replace folder_path with the NeonTree official folder
    folder_path = 'path/to/dataset'
    # This is an example for testing
    tile_name = '2018_BART_4_322000_4882000_image_crop'
    tif_name = tile_name + '.tif'
    xml_name = tile_name + '.xml'
    neontree_path = Path(folde_path)
    tif_path = neontree_path / 'training' / 'RGB' / tif_name
    annot_path = neontree_path / 'annotations' / xml_name
    tilerizer = Tilerizer(tif_path, annot_path)
    samples = tilerizer.create_tiles()
