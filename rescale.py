import argparse
import os
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from skimage.io import imread, imsave
from skimage.transform import resize, rotate
import re
from PIL import Image

TARGET_PIXEL_SIZE = 0.5


def get_image_filename(prefix):
    """
    Find existing image file based on prefix
    Args:
        prefix: File path prefix
    Returns:
        Found image filename
    Raises:
        FileNotFoundError: When no image file is found
    """
    file_exists = False
    # Try common image suffixes
    for suffix in ['.jpg', '.png', '.tiff']:
        filename = prefix + suffix
        if os.path.exists(filename):
            file_exists = True
            break
    if not file_exists:
        raise FileNotFoundError('Image not found')
    return filename


def calculate_rescale_factor(original_pixel_size):
    """
    Calculate the rescaling factor to achieve target pixel size
    Args:
        original_pixel_size: Original pixel size in μm
    Returns:
        Rescaling factor
    """
    return original_pixel_size / TARGET_PIXEL_SIZE


def parse_polygon(polygon_str):
    """
    Parse polygon string and scale coordinates
    Args:
        polygon_str: Polygon string in format "POLYGON ((x1 y1, x2 y2, ..., xn yn))"
    Returns:
        shapely.geometry.Polygon object
    """
    # Extract coordinates using regular expressions
    match = re.search(r'POLYGON\s*\(\s*\(\s*(.*?)\s*\)\s*\)', polygon_str)
    if not match:
        raise ValueError("Invalid polygon format")
    
    coords = match.group(1)
    points = []
    for point in coords.split(','):
        point = point.strip()
        if point:
            # Remove parentheses and split x and y coordinates
            point = point.replace('(', '').replace(')', '')
            x, y = map(float, point.split())
            # Scale coordinates
            x_scaled = x * scale
            y_scaled = y * scale
            points.append((x_scaled, y_scaled))
    return Polygon(points)


def extract_256_image(image, polygon):
    """
    Extract a 256*256 cell image based on polygon center, pad with 0 if out of bounds
    Args:
        image: Original image array
        polygon: Polygon coordinates
    Returns:
        256*256 cell image array
    """
    # Get polygon center
    center_x, center_y = polygon.centroid.x, polygon.centroid.y
    
    # Calculate extraction region boundaries
    half_size = 128
    start_x = int(center_x - half_size)
    start_y = int(center_y - half_size)
    end_x = start_x + 256
    end_y = start_y + 256
    
    # Create a zero-filled 256*256 image
    extracted_image = np.zeros((256, 256, image.shape[2]), dtype=image.dtype)
    
    # Calculate valid region in the original image
    valid_start_x = max(0, start_x)
    valid_start_y = max(0, start_y)
    valid_end_x = min(image.shape[1], end_x)
    valid_end_y = min(image.shape[0], end_y)
    
    # Calculate corresponding region in the extracted image
    extract_start_x = valid_start_x - start_x
    extract_start_y = valid_start_y - start_y
    extract_end_x = extract_start_x + (valid_end_x - valid_start_x)
    extract_end_y = extract_start_y + (valid_end_y - valid_start_y)
    
    # Copy valid region to the extracted image
    extracted_image[extract_start_y:extract_end_y, extract_start_x:extract_end_x] = image[valid_start_y:valid_end_y, valid_start_x:valid_end_x]
    
    return extracted_image


def process_cells(image_path, cells_data_path, output_dir):
    """
    Process all cell images
    Args:
        image_path: Path to the original image
        cells_data_path: Path to the cell data CSV file
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read original image
    image = imread(image_path)
    
    # Read cell data
    cells_data = pd.read_csv(cells_data_path, skiprows=1, names=['geometry', 'id'])
    
    # Process each cell
    processed_count = 0
    error_count = 0
    
    for _, row in cells_data.iterrows():
        cell_id = row['id']
        
        try:
            # Parse polygon coordinates
            polygon = parse_polygon(row['geometry'])
            
            # Extract 256*256 cell image
            cell_image = extract_256_image(image, polygon)
            #cell_image = np.flipud(cell_image)
            #cell_image = np.fliplr(cell_image)
            #cell_image = rotate(cell_image, -90, resize=True, mode='reflect', preserve_range=True).astype(np.uint8)  # Rotate clockwise 90 degrees
            # Save image
            imsave(os.path.join(output_dir, f"{cell_id}_256.tif"), cell_image)
            processed_count += 1
            
            # Output progress every 100 cells processed
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} cells, current ID: {cell_id}...")
        except Exception as e:
            print(f"Error processing cell {cell_id}: {e}")
            error_count += 1
            # If too many errors, exit processing
            if error_count > 100:
                print("Too many errors, stopping processing")
                break
    
    print(f"Processing completed, processed {processed_count} cells, encountered {error_count} errors")


def test_image_transformation(input_path, output_path):
    """
    Test image flipping and rotation functionality
    """
    image = imread(input_path)
    transformed_image = np.fliplr(image)
    transformed_image = rotate(transformed_image, 90, resize=True, mode='reflect', preserve_range=True).astype(np.uint8)
    imsave(output_path, transformed_image)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--cells', type=str, help='Cell data CSV file path')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--test-image', type=str, help='Test image path')
    # Add new parameter pixel_size_raw
    parser.add_argument('--pixel-size-raw', type=float, help='Original pixel size', required=True)
    return parser.parse_args()


def main():
    """Main function, coordinate the entire processing flow"""
    args = get_args()

    global scale
    # Get pixel_size_raw from command line arguments
    pixel_size_raw = args.pixel_size_raw
    pixel_size = float(0.5)
    scale = pixel_size_raw / pixel_size

    # Set scale in global scope
    
    #scale = scale

    if args.test_image:
        test_image_transformation(args.test_image, "output.png")
        print("Test image saved as output.png")
    elif args.image and args.cells and args.output_dir:
        # Process cell images
        process_cells(args.image, args.cells, args.output_dir)
    else:
        print("Please provide necessary parameters, use --test-image for testing, or provide --image, --cells and --output-dir for cell image processing.")


if __name__ == "__main__":
    main()