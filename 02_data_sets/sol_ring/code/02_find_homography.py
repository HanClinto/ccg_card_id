import os
import json
import sys
from pathlib import Path

# ClintUtils is a sibling repo — add its parent to sys.path so it's importable
_project_root = Path(__file__).resolve().parents[3]
_clintutils_parent = _project_root.parent
if str(_clintutils_parent) not in sys.path:
    sys.path.insert(0, str(_clintutils_parent))

from tqdm import tqdm

import cv2 as cv
import numpy as np

import ClintUtils.files as cffiles
import ClintUtils.align_img as align_img

# Resolve data dir from project environment
sys.path.insert(0, str(_project_root))
from ccg_card_id.project_settings import get_data_root

_solring_dir = get_data_root() / "datasets" / "solring"

DATA_DIR = str(_solring_dir / '02_keyframes')
REFERENCE_DIR = str(_solring_dir / '03_reference')
OUTPUT_DIR = str(_solring_dir / '04_data')
OUTPUT_DIR_GOOD = os.path.join(OUTPUT_DIR, 'good')
OUTPUT_DIR_BAD = os.path.join(OUTPUT_DIR, 'bad')
OUTPUT_DIR_ALIGNED = os.path.join(OUTPUT_DIR, 'aligned')

OUTPUT_DIR_RESIZED = os.path.join(OUTPUT_DIR, 'resized')
OUTPUT_DIR_RESIZED_224 = os.path.join(OUTPUT_DIR_RESIZED, '224')
OUTPUT_DIR_RESIZED_512 = os.path.join(OUTPUT_DIR_RESIZED, '512')

OUTPUT_DIR_MASKS = os.path.join(OUTPUT_DIR, 'masks')
OUTPUT_DIR_MASKS_224 = os.path.join(OUTPUT_DIR_MASKS, '224')
OUTPUT_DIR_MASKS_512 = os.path.join(OUTPUT_DIR_MASKS, '512')

os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_GOOD, exist_ok=True)
os.makedirs(OUTPUT_DIR_BAD, exist_ok=True)
os.makedirs(OUTPUT_DIR_ALIGNED, exist_ok=True)
os.makedirs(OUTPUT_DIR_RESIZED, exist_ok=True)
os.makedirs(OUTPUT_DIR_RESIZED_224, exist_ok=True)
os.makedirs(OUTPUT_DIR_RESIZED_512, exist_ok=True)
os.makedirs(OUTPUT_DIR_MASKS, exist_ok=True)
os.makedirs(OUTPUT_DIR_MASKS_224, exist_ok=True)
os.makedirs(OUTPUT_DIR_MASKS_512, exist_ok=True)

# Enumerate all the files in the data directory
files = os.listdir(DATA_DIR)

# Simple JSON-backed key-value store (replaces pickledb)
class _JsonDB:
    def __init__(self, path):
        self._path = path
        self._data = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    def exists(self, key): return key in self._data
    def set(self, key, val): self._data[key] = val
    def get(self, key): return self._data.get(key)
    def rem(self, key): self._data.pop(key, None)
    def getall(self): return list(self._data.keys())
    def dump(self): Path(self._path).write_text(json.dumps(self._data, indent=2))

# Load our homography index
homography_db = _JsonDB(OUTPUT_DIR + '/homography.json')

RECHECK_HOMOGRAPHY = False # Do we re-check files we have already processed?

# For each file
for img_file in tqdm(files):
    # If the file isn't an image, skip it
    if not (img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png')):
        continue

    # Get the file name
    print(f'File: "{img_file}"')
    sfid = img_file.split('_')[0]
    print(f' SFID: "{sfid}"')

    # Check if we have already processed this file
    if not RECHECK_HOMOGRAPHY and homography_db.exists(img_file):
        continue

    # Get the reference image for this file
    ref_img_url = f"https://cards.scryfall.io/png/front/{sfid[0]}/{sfid[1]}/{sfid}.png"
    ref_img_path = os.path.join(REFERENCE_DIR, sfid + '.png')
    cffiles.download_if_not_exists(ref_img_path, ref_img_url)

    ref_img = cv.imread(ref_img_path)
    img = cv.imread(DATA_DIR + '/' + img_file)

    bad_img = False

    try:
        alignment = align_img.align_images(img, ref_img, verify_match=True)
        print(f'Got alignment for {img_file}')
    except Exception as e:
        print(f'Failed to find homography for {img_file}')
        print(e)
        bad_img = True

    if alignment['error']:
        print(f'Error in alighment for {img_file}')
        print(alignment['error_message'])
        bad_img = True

    if bad_img:
        print(f'Bad image: {img_file}')
        # Move the file to bad images folder
        os.rename(DATA_DIR + '/' + img_file, os.path.join(OUTPUT_DIR_BAD, img_file))

        # If the image is in our database, remove it
        if homography_db.exists(img_file):
            homography_db.rem(img_file)

        continue
    
    # Save the aligned image from output_image
    cv.imwrite(OUTPUT_DIR_ALIGNED + '/' + img_file, alignment['output_image'])

    # Calculate area of overlap

    # Find the smallest X coordinate of the scene corners
    min_x = min(alignment['scene_corners'][0][0][0], alignment['scene_corners'][1][0][0], alignment['scene_corners'][2][0][0], alignment['scene_corners'][3][0][0])
    # Find the largest X coordinate of the scene corners
    max_x = max(alignment['scene_corners'][0][0][0], alignment['scene_corners'][1][0][0], alignment['scene_corners'][2][0][0], alignment['scene_corners'][3][0][0])
    # Find the smallest Y coordinate of the scene corners
    min_y = min(alignment['scene_corners'][0][0][1], alignment['scene_corners'][1][0][1], alignment['scene_corners'][2][0][1], alignment['scene_corners'][3][0][1])
    # Find the largest Y coordinate of the scene corners
    max_y = max(alignment['scene_corners'][0][0][1], alignment['scene_corners'][1][0][1], alignment['scene_corners'][2][0][1], alignment['scene_corners'][3][0][1])

    # Calculate the area of overlap (just using bounding boxes, nothing precise)
    img_size = alignment['scene_img_size']

    bbox_area = (max_x - min_x) * (max_y - min_y)
    # Calculate the area of the scene image
    scene_area = img_size[0] * img_size[1]
    
    bbox_area_pct = bbox_area / scene_area

    alignment['matching_area_pct'] = bbox_area_pct

    alignment['sfid'] = sfid

    # Sort the corners
    min_corner_sum = -1
    min_corner_index = 0
    corners = []
    for idx, scene_corner in enumerate(alignment["scene_corners"]):
        scene_corner = scene_corner[0]
        corner_sum = scene_corner[0] + scene_corner[1]
        if min_corner_sum == -1 or corner_sum < min_corner_sum:
            min_corner_sum = corner_sum
            min_corner_index = idx

        # Normalize the corners to be in the range 0-1
        scene_corner[0] = scene_corner[0] / img_size[1]
        scene_corner[1] = scene_corner[1] / img_size[0]

        corners.append(scene_corner)

    corners_sorted = []
    for corner_idx in range(4):
        corner_idx = (corner_idx + min_corner_index) % 4
        corners_sorted.append(corners[corner_idx])

    alignment['corners'] = corners
    alignment['corners_sorted'] = corners_sorted
    alignment['min_corner_index'] = min_corner_index

    # Put the file in the good images folder
    good_img_path = os.path.join(OUTPUT_DIR_GOOD, img_file)
    # NOTE: Use cv.imwrite if we want to duplicate the image in both places.
    # cv.imwrite(good_img_path, img)
    # NOTE: Use os.rename if we want to move the file instead of copying it.
    os.rename(os.path.join(DATA_DIR, img_file), good_img_path)
    alignment['img_path'] = good_img_path
    alignment['filename'] = img_file

    img_file_png = os.path.splitext(img_file)[0] + '.png'

    # Resize the image into our target sizes
    for size in [224, 512]:
        resized_img = cv.resize(img, (size, size))
        resized_filename = os.path.join(OUTPUT_DIR_RESIZED,str(size),img_file_png)
        alignment[f'resized_path_{size}'] = resized_filename
        cv.imwrite(resized_filename, resized_img)

        # Create a mask for the image
        mask = np.zeros(resized_img.shape, dtype=np.uint8)
        # Draw circles on the mask for each of the corner points
        circle_size = int(size / 50)

        for corner in alignment['corners']:
            cv.circle(mask, (int(corner[0] * size), int(corner[1] * size)), circle_size, (255, 255, 255), -1)

        # Write the file as a PNG
        mask_filename = os.path.join(OUTPUT_DIR_MASKS,str(size),img_file_png)
        alignment[f'mask_path_{size}'] = mask_filename
        cv.imwrite(mask_filename, mask)

    # Remove output_image from alignment so that we don't serialize it.
    if 'output_image' in alignment:
        del alignment['output_image']

    # Convert 'obj_corners' to a list of lists
    alignment['obj_corners'] = alignment['obj_corners'].tolist()

    # Save the alignment to the database
    homography_db.set(sfid, alignment)

homography_db.dump()

allkeys = homography_db.getall()

print(f'Found {len(allkeys)} images')

with open(OUTPUT_DIR + '/dataset.tsv', 'w') as f:
    f.write('img_path\tresized_512\tmask_512\tresized_224\tmask_224\tsfid\tphash_similarity\tmatching_area_pct\tnum_good_matches\tnum_keypoints_obj\tnum_keypoints_scene\tpct_match_obj\tpct_match_scene\tcorner0_x\tcorner0_y\tcorner1_x\tcorner1_y\tcorner2_x\tcorner2_y\tcorner3_x\tcorner3_y\n')
    for key in allkeys:
        datum = homography_db.get(key)
        f.write(f'{datum["img_path"]}\t{datum["resized_path_512"]}\t{datum["resized_path_512"]}\t{datum["mask_path_512"]}\t{datum["mask_path_224"]}\t{datum["sfid"]}\t{datum["phash_similarity"]}\t{datum["matching_area_pct"]}\t{datum["num_good_matches"]}\t{datum["num_keypoints_obj"]}\t{datum["num_keypoints_scene"]}\t{datum["pct_match_obj"]}\t{datum["pct_match_scene"]}\t{datum["min_corner_index"]}\t{datum["corners"][0][0]}\t{datum["corners"][0][1]}\t{datum["corners"][1][0]}\t{datum["corners"][1][1]}\t{datum["corners"][2][0]}\t{datum["corners"][2][1]}\t{datum["corners"][3][0]}\t{datum["corners"][3][1]}\n')
