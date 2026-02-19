# Sync Scryfall card images
# Given a bulk data JSON from Scryfall exists in ./cache/all_cards.json
# Download images for all cards listed in that file into ./cache/images/png/front/{card_id}[0]/{card_id}[1]/{card_id}.jpg
# Cards in the bulk data JSON have an "image_uris" field with URLs for various image types and sizes
# Examples:
#  https://cards.scryfall.io/png/front/0/2/02d6d693-f1f3-4317-bcc0-c21fa8490d38.png?1651492800
#  https://cards.scryfall.io/png/back/0/2/02d6d693-f1f3-4317-bcc0-c21fa8490d38.png?1651492800
# Note that the image URL includes a timestamp query parameter, which should be compared against the local file mtime to determine if the image needs to be re-downloaded
# If the local image file does not exist, or if the timestamp in the URL is newer than the local file mtime, download the image and save it to the appropriate path
# If a card has multiple faces, then download images for every face and save in the appropriate locations
# For a single-faced card, then image_uris is directly on the card object
# For a multi-faced card, then image_uris is on each face object within the object's card_faces array

import os
import requests
# import orjson
import json
from datetime import datetime
import datetime as dt
from tqdm import tqdm
import re
from PIL import Image
import imagehash
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

image_types = ["png", "large", "normal", "small", "art_crop", "border_crop"]
image_quality = "png"

english_only = True
bulk_data_types = ["default_cards" ,"all_cards"]
bulk_data_type = "default_cards"

# To get the root of the repository, replace everything after "ccg_card_id" in the path
CCG_CARD_ID_ROOT = re.sub("ccg_card_id/.*", "ccg_card_id", os.path.dirname(__file__))
SCRYFALL_CACHE_DIR = os.path.join(CCG_CARD_ID_ROOT, "01_data_sources", "scryfall", "cache")
BULK_DATA_PATH = os.path.join(SCRYFALL_CACHE_DIR, f"{bulk_data_type}.json")
IMAGES_DIR = os.path.join(SCRYFALL_CACHE_DIR, "images", image_quality)

DATASET_REF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "03_reference")
DATASET_TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "04_data", "aligned")

hash_methods = ["phash"] #, "whash_db4"] #, "ahash", "dhash", "crhash", "whash", "whash_db4"] #,  #["brief"] #, "phash"]
hash_sizes = [32, 64, 128, 256] #[8, 16, 32, 64, 128, 256]

reference_hashes = {}
for method in hash_methods:
    reference_hashes[method] = {}
    for size in hash_sizes:
        reference_hashes[method][size] = {}

def hash_distance(hash1, hash2, method):
    if method == "brief":
        """Compute the Hamming distance between two byte strings."""
        if len(hash1) != len(hash2):
            raise ValueError(f"Hashes must be of the same length. Hash1 length: {len(hash1)}, Hash2 length: {len(hash2)}")
        distance = 0
        for b1, b2 in zip(hash1, hash2):
            xor = b1 ^ b2
            distance += bin(xor).count("1")
        return distance
    else:
        return hash1 - hash2

def hash_file_imagehash(img, method, hash_size):
    # If img is a string, then open it as a file path
    if isinstance(img, str):
        img = Image.open(img)

    if method == "phash":
        phash = imagehash.phash(img, hash_size=hash_size)
    elif method == "ahash":
        phash = imagehash.average_hash(img, hash_size=hash_size)
    elif method == "dhash":
        phash = imagehash.dhash(img, hash_size=hash_size)
    elif method == "crhash":
        phash = imagehash.crop_resistant_hash(img)
    elif method == "whash":
        phash = imagehash.whash(img, hash_size=hash_size)
    elif method == "whash_db4":
        phash = imagehash.whash(img, hash_size=hash_size, mode="db4")
    else:
        raise ValueError(f"Unknown imagehash method: {method}")
    # Flatten the phash and get the raw bytes
    return phash # phash.hash

def hash_file_brief(path):
    import cv2
    import numpy as np

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Initiate FAST detector
    star = cv2.xfeatures2d.StarDetector_create()
    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # find the keypoints with STAR
    kp = star.detect(img,None)
    # compute the descriptors with BRIEF
    kp, des = brief.compute(img, kp)
    if des is None:
        return None
    # Flatten the descriptor array and convert to bytes
    des_flat = des.flatten()
    return des_flat.tobytes()

def hash_file(path, method, hash_size):
    if method == "brief":
        return hash_file_brief(path)
    else:
        return hash_file_imagehash(path, method, hash_size)
    
def load_bulk_data():
    """
    Loads the bulk data JSON from local cache.
    """
    then = datetime.now()
    print(f"Loading bulk data from {BULK_DATA_PATH}...")
    # Load bulk data JSON
    with open(BULK_DATA_PATH, "r", encoding="utf-8") as f:
        # bulk_data = orjson.loads(f.read())
        bulk_data = json.load(f)
    print(f"Loaded {len(bulk_data)} cards in {(datetime.now() - then).total_seconds():.2f} seconds.")
    return bulk_data

def local_path(card, face):
    card_id = card["id"]
    if "image_uris" not in face:
        face = card.get("card_faces", [card])[0]  # Use first face if image_uris not present
    image_uris = face.get("image_uris", {})
    image_url = image_uris.get(image_quality)
    if not image_url:
        return None  # No image available for this quality

    # Determine local image path
    subdir1 = card_id[0]
    subdir2 = card_id[1]
    # Given an image URL that looks like: 
    #  https://cards.scryfall.io/small/front/0/2/02d6d693-f1f3-4317-bcc0-c21fa8490d38.jpg?1651492800
    #  https://cards.scryfall.io/small/back/0/2/02d6d693-f1f3-4317-bcc0-c21fa8490d38.jpg?1651492800
    # Extract the string "front" or "back" from the image URL
    face_name = image_url.split("/")[4]
    local_dir = os.path.join(IMAGES_DIR, face_name, subdir1, subdir2)
    os.makedirs(local_dir, exist_ok=True)
    image_extension = image_url.split(".")[-1].split("?")[0]  # Extract extension before query params
    local_image_path = os.path.join(local_dir, f"{card_id}.{image_extension}")
    return local_image_path

def link_ref_images(bulk_data):
    for card in bulk_data:
        if card.get("image_status") in ("missing", "placeholder"):
            continue  # Skip cards without valid images

        if english_only and card.get("lang") != "en":
            continue  # Skip non-English cards

        if card.get("illustration_id") == "146aaae4-93f4-409a-be32-010e86d137da":
            # Ensure that this image is in our reference folder
            local_card_img_path = local_path(card, card)

            dest_img_path = os.path.join(DATASET_REF_DIR, os.path.basename(local_card_img_path))

            if not os.path.exists(dest_img_path):
                # Symlink from local_card_img_path to our dest_img_path
                os.link(local_card_img_path, dest_img_path)
            
            # Hash the reference image and store its hash
            for method in hash_methods:
                for hash_size in hash_sizes:
                    img_hash = hash_file(dest_img_path, method, hash_size)
                    reference_hashes[method][hash_size][card["id"]] = img_hash

def hash_all_images(bulk_data):
    """
    Hashes all images in bulkdata
    """
    for card in tqdm(bulk_data, desc="Hashing all images"):
        if card.get("image_status") in ("missing", "placeholder"):
            continue  # Skip cards without valid images

        if english_only and card.get("lang") != "en":
            continue  # Skip non-English cards

        # Ensure that this image is in our reference folder
        local_card_img_path = local_path(card, card)

        # Hash the reference image and store its hash
        img = Image.open(local_card_img_path)
        for method in hash_methods:
            for hash_size in hash_sizes:
                img_hash = hash_file(img, method, hash_size)
                reference_hashes[method][hash_size][card["id"]] = img_hash

def save_reference_hashes():
    ref_hashes_strs = {}
    for method in hash_methods:
        ref_hashes_strs[method] = {}
        for hash_size in hash_sizes:
            ref_hashes_strs[method][hash_size] = {}
            for ref_id, ref_hash in reference_hashes[method][hash_size].items():
                ref_hashes_strs[method][hash_size][ref_id] = str(ref_hash)

    for method in hash_methods:
        for hash_size in hash_sizes:
            REFERENCE_HASHES_PATH = os.path.join(SCRYFALL_CACHE_DIR, f"{bulk_data_type}_{method}_{hash_size}.json")
            with open(REFERENCE_HASHES_PATH, "w", encoding="utf-8") as f:
                json.dump(ref_hashes_strs[method][hash_size], f, indent=2)

def load_reference_hashes():
    global reference_hashes

    ref_hashes_strs = {}
    for method in tqdm(hash_methods, desc="Loading reference hashes from disk"):
        ref_hashes_strs[method] = {}
        for hash_size in hash_sizes:
            REFERENCE_HASHES_PATH = os.path.join(SCRYFALL_CACHE_DIR, f"{bulk_data_type}_{method}_{hash_size}.json")
            if not os.path.exists(REFERENCE_HASHES_PATH):
                return False
            with open(REFERENCE_HASHES_PATH, "r", encoding="utf-8") as f:
                ref_hashes_strs[method][hash_size] = json.load(f)

    # Use multiprocessing to parallelize the conversion
    import multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 8)  # Use up to 8 workers
    
    for method in hash_methods:
        reference_hashes[method] = {}
        for hash_size in hash_sizes:
            reference_hashes[method][hash_size] = {}
            
            # Convert items to list and split into batches
            items = list(ref_hashes_strs[method][hash_size].items())
            batch_size = max(100, len(items) // (num_workers * 10))
            batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
            
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(convert_hash_batch, batch): batch for batch in batches}
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                 desc=f"Converting {method}[{hash_size}]"):
                    results = future.result()
                    for ref_id, ref_hash in results:
                        reference_hashes[method][hash_size][ref_id] = ref_hash
    
    return True

def convert_hash_batch(items):
    """Convert a batch of (ref_id, ref_hash_str) tuples to hash objects."""
    return [(ref_id, imagehash.hex_to_hash(ref_hash_str)) for ref_id, ref_hash_str in items]

def get_scryfall_id_from_filename(filename):
    # Given a filename like:
    # "02d6d693-f1f3-4317-bcc0-c21fa8490d38_front.png"
    # "evolvingwilds_MID_nosleeve_cb471f90-46f2-4037-87fc-f523fc9d004f_003.png"
    # Extract the Scryfall ID (which is a UUID) from the filename
    match = re.search(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", filename)
    if match:
        return match.group(0)
    return None

if __name__ == "__main__":
    bulk_data = load_bulk_data()
    if load_reference_hashes():
        print("Loaded reference hashes from disk.")
    else:
        print("No cached reference hashes found, computing from images...")
        hash_all_images(bulk_data)
        save_reference_hashes()

    img_counts = {}
    correct_counts = {}
    correct_images = {}
    incorrect_images = {}

#    link_ref_images(bulk_data)
    # Save reference hashes to disk
    #REFERENCE_HASHES_PATH = os.path.join(DATASET_REF_DIR, "reference_hashes.json")
    #with open(REFERENCE_HASHES_PATH, "w", encoding="utf-8") as f:
    #    json.dump(reference_hashes, f, indent=2)

    for method in hash_methods:
        img_counts[method] = {}
        correct_counts[method] = {}
        correct_images[method] = {}
        incorrect_images[method] = {}
        for hash_size in hash_sizes:
            print(f"Computed {method} hashes for {len(reference_hashes[method])} reference images.")
            img_counts[method][hash_size] = 0
            correct_counts[method][hash_size] = 0
            correct_images[method][hash_size] = []
            incorrect_images[method][hash_size] = []

            # Iterate through all files in DATASET_TEST_DIR and compute their hashes
            for root, dirs, files in os.walk(DATASET_TEST_DIR):
                for file in tqdm(files, desc=f"Hashing test images with {method}[{hash_size}]"):
                    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue  # Skip non-image files
                    file_path = os.path.join(root, file)
                    img_hash = hash_file(file_path, method, hash_size)
                    # Compare against all reference hashes
                    dists = {}
                    for ref_id, ref_hash in reference_hashes[method][hash_size].items():
                        if img_hash is None or ref_hash is None:
                            continue
                        distance = hash_distance(img_hash, ref_hash, method)
                        dists[ref_id] = distance

                    # Find the minimum distance
                    if dists:
                        short_filename = os.path.basename(file_path)
                        img_counts[method][hash_size] += 1
                        best_match = min(dists, key=dists.get)
                        # print(f" File {short_filename} best matches reference {best_match} with {method} distance {dists[best_match]}")
                        # Get the correct ID for this test image from its filename
                        correct_id = file.split("_")[0]
                        correct_dist = dists.get(correct_id, None)
                        if correct_dist == dists[best_match]:
                            print(f"  Correctly matched {short_filename} to {correct_id} using {method}.")
                            correct_counts[method][hash_size] += 1
                            correct_images[method][hash_size].append((short_filename, best_match, dists[best_match]))
                        else:
                            print(f"  Incorrectly matched {short_filename} to {best_match} (correct: {correct_id}) using {method}.")
                            incorrect_images[method][hash_size].append((short_filename, best_match, dists[best_match], correct_id, correct_dist))
            if method == "crhash":
                continue # Skip other sizes for crhash because it's not size-dependent

    for method in hash_methods:
        for hash_size in hash_sizes:
            print(f"#### {method}[{hash_size}] matching accuracy: {correct_counts[method][hash_size]}/{img_counts[method][hash_size]} = {(correct_counts[method][hash_size]/img_counts[method][hash_size]*100) if img_counts[method][hash_size] > 0 else 0:.2f}%")

    # Find images that are correctly matched by all methods and sizes
    for method in hash_methods:
        for hash_size in hash_sizes:
            correct_set = set(img[0] for img in correct_images[method][hash_size])
            print(f"#### {method}[{hash_size}] correctly matched images: {len(correct_set)}")
            for img in correct_set:
                print(f"  Correct: {img}")
            incorrect_set = set(img[0] for img in incorrect_images[method][hash_size])
            print(f"#### {method}[{hash_size}] incorrectly matched images: {len(incorrect_set)}")
            for img in incorrect_set:
                print(f"  Incorrect: {img}")
    