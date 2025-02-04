import json
import numpy as np
import cv2
from shapely.wkt import loads
from shapely.affinity import scale
import matplotlib.pyplot as plt

def load_mask(json_path, img_size=(512, 512)):
    """
    Generate a segmentation mask from an xBD post-disaster JSON label file.

    Args:
        json_path (str): Path to the JSON label file.
        img_size (tuple): Target image size (height, width).

    Returns:
        np.ndarray: A 2D mask where each pixel corresponds to a damage class.
    """
    # Damage mapping for post-disaster masks
    damage_mapping = {
        "no-damage": 0,
        "minor-damage": 1,
        "major-damage": 2,
        "destroyed": 3
    }

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Initialize blank mask
    mask = np.zeros(img_size, dtype=np.uint8)

    # Get original image dimensions
    original_width = data["metadata"]["original_width"]
    original_height = data["metadata"]["original_height"]

    # Loop through polygons
    for feature in data["features"]["xy"]:
        properties = feature["properties"]

        # Get damage level from the "subtype" field
        damage_level = properties.get("subtype", "no-damage")  # Default to "no-damage"
        class_value = damage_mapping.get(damage_level, 0)  # Map to class index

        # Load WKT polygon
        polygon = loads(feature["wkt"])

        # Rescale coordinates from original size to target size
        scale_x = img_size[1] / original_width
        scale_y = img_size[0] / original_height
        polygon = scale(polygon, xfact=scale_x, yfact=scale_y, origin=(0, 0))

        # Convert polygon to OpenCV format
        if polygon.geom_type == "Polygon":
            pts = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask, [pts], class_value)

    print(f" Mask shape: {mask.shape}, Unique values: {np.unique(mask)}")
    return mask


# Test the function with a post-disaster JSON file
if __name__ == "__main__":
    label_path = "train/labels/palu-tsunami_00000166_post_disaster.json"
    mask = load_mask(label_path)

    # Visualize the mask
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="hot", vmin=0, vmax=3)
    plt.title("Post-Disaster Damage Segmentation Mask")
    plt.colorbar(ticks=[0, 1, 2, 3], label="Damage Class")
    plt.show()
