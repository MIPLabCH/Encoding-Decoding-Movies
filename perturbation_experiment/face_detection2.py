import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def extract_foreground_grabcut(image_path, input_image, rectangle=None, iterations=3, output_dir=None, show_result=False):
    """
    Extract foreground from an image using GrabCut algorithm
    
    Args:
        image_path (str): Path to the input image
        rectangle (tuple): ROI as (x, y, width, height). If None, uses center portion
        iterations (int): Number of GrabCut iterations
        output_dir (str): Directory to save output files
        show_result (bool): Whether to display the result using matplotlib
    
    Returns:
        dict: {
            'success': bool,
            'original_image': numpy array,
            'segmented_image': numpy array,
            'mask': numpy array,
            'output_path': str or None,
            'error': str or None
        }
    """
    #try:
    # Load the input image
    image = cv2.imread(image_path)

    if image_path is None:
        image = input_image
    else:
        image = cv2.imread(image_path)
        if image is None:
            print("no image in the path")
            return {
                'success': False,
                'original_image': None,
                'segmented_image': None,
                'mask': None,
                'output_path': None,
                'error': f"Could not load image from {image_path}"
            }
    
    height, width = image.shape[:2]
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a mask image similar to the loaded image
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # Initialize background and foreground models
    # Arrays of 1 row and 65 columns, all elements are 0
    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)
    
    # Define the Region of Interest (ROI) rectangle
    if rectangle is None:
        # Use center 70% of the image as default ROI
        margin_x = int(width * 0.15)
        margin_y = int(height * 0.15)
        rect_width = width - 2 * margin_x
        rect_height = height - 2 * margin_y
        rectangle = (margin_x, margin_y, rect_width, rect_height)
    
    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rectangle, backgroundModel, foregroundModel, 
                iterations, cv2.GC_INIT_WITH_RECT)
    
    # Create final mask
    # Convert mask: 0,2 -> 0 (background), 1,3 -> 1 (foreground)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply mask to get segmented image
    segmented_image = image * mask2[:, :, np.newaxis]
    
    # Save the result
    output_path = None
    if output_dir or not output_dir:
        # Get base filename for output files
        if image_path is None:
            base_filename = "grabcut_" + str(np.random.randint(1000, 9999))  # Random name if no path provided
        else:
            base_filename = os.path.splitext(os.path.basename(image_path))[0]

        output_filename = f"{base_filename}_grabcut_extracted.jpg"
        if output_dir:
            output_path = os.path.join(output_dir, output_filename)
        else:
            output_path = output_filename
        cv2.imwrite(output_path, segmented_image)
    
    # Display result if requested
    if show_result:
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Segmented image
        plt.subplot(1, 2, 2)
        plt.title('Extracted Foreground')
        plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'success': True,
        'original_image': image,
        'segmented_image': segmented_image,
        'mask': mask2,
        'output_path': output_path,
        'error': None
    }
        
'''        
    except Exception as e:
        return {
            'success': False,
            'original_image': None,
            'segmented_image': None,
            'mask': None,
            'output_path': None,
            'error': str(e)
        }
'''    


def detect_and_extract_faces(image_path, input_image, output_dir=None, save_faces=True, save_annotated=True):
    """
    Detect and extract faces from an image using OpenCV and Haar Cascade
    
    Args:
        image_path (str): Path to the input image
        output_dir (str, optional): Directory to save output files. If None, uses current directory
        save_faces (bool): Whether to save individual face images
        save_annotated (bool): Whether to save the annotated image with rectangles
    
    Returns:
        dict: {
            'num_faces': int,
            'face_coordinates': list of (x, y, w, h) tuples,
            'annotated_image_path': str or None,
            'extracted_face_paths': list of str,
            'success': bool,
            'error': str or None
        }
    """
    
#    try:
        # Load the input image
    if image_path is None:
        image = input_image
    else:
        image = cv2.imread(image_path)
        if image is None:
            print("no image in the path")
            return {
                'num_faces': 0,
                'face_coordinates': [],
                'annotated_image_path': None,
                'extracted_face_paths': [],
                'success': False,
                'error': f"Could not load image from {image_path}"
            }
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert image to grayscale for better face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    
    extracted_face_paths = []
    face_coordinates = []
    
    # Get base filename for output files
    if image_path is None:
        base_filename = "extracted_faces" + str(np.random.randint(1000, 9999))  # Random name if no path provided
    else:
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        face_coordinates.append((x, y, w, h))
        
        if save_faces:
            # Extract the face region
            face_region = image[y:y + h, x:x + w]
            
            # Create face filename
            face_filename = f"{base_filename}_face_{i+1}_{w}x{h}.jpg"
            if output_dir:
                face_path = os.path.join(output_dir, face_filename)
            else:
                face_path = face_filename
            
            # Save the extracted face
            cv2.imwrite(face_path, face_region)
            extracted_face_paths.append(face_path)
        
        if save_annotated:
            # Draw rectangle around face on the original image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Save the annotated image
    annotated_image_path = None
    if save_annotated and len(faces) > 0:
        annotated_filename = f"{base_filename}_faces_detected.jpg"
        if output_dir:
            annotated_image_path = os.path.join(output_dir, annotated_filename)
        else:
            annotated_image_path = annotated_filename
        cv2.imwrite(annotated_image_path, image)
    
    return {
        'num_faces': len(faces),
        'face_coordinates': face_coordinates,
        'annotated_image_path': annotated_image_path,
        'extracted_face_paths': extracted_face_paths,
        'success': True,
        'error': None
    }

'''        
    except Exception as e:
        print("something went wrong..")
        return {
            'num_faces': 0,
            'face_coordinates': [],
            'annotated_image_path': None,
            'extracted_face_paths': [],
            'success': False,
            'error': str(e)
        }
'''


















def extract_with_custom_roi(image_path, x, y, width, height, iterations=5, output_dir=None):
    """
    Extract foreground with a custom ROI rectangle
    
    Args:
        image_path (str): Path to input image
        x, y, width, height (int): ROI coordinates
        iterations (int): Number of iterations
        output_dir (str): Output directory
    
    Returns:
        dict: Result dictionary
    """
    rectangle = (x, y, width, height)
    return extract_foreground_grabcut(image_path, rectangle, iterations, output_dir)

'''
def interactive_grabcut(image_path, output_dir=None):
    """
    Interactive GrabCut with mouse selection
    Note: This would require a GUI framework for full interactivity
    For now, it uses a default center rectangle
    """
    print("Loading image for interactive GrabCut...")
    print("Using default center rectangle. For custom selection, use extract_with_custom_roi()")
    
    result = extract_foreground_grabcut(image_path, output_dir=output_dir, show_result=True)
    
    if result['success']:
        print(f"✓ Foreground extracted successfully!")
        if result['output_path']:
            print(f"✓ Saved to: {result['output_path']}")
    else:
        print(f"✗ Error: {result['error']}")
    
    return result
'''

def batch_grabcut_extraction(image_paths, output_dir="grabcut_results"):
    """
    Process multiple images with GrabCut
    
    Args:
        image_paths (list): List of image file paths
        output_dir (str): Directory to save results
    
    Returns:
        dict: Results for each image
    """
    results = {}
    
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        result = extract_foreground_grabcut(image_path, output_dir=output_dir)
        results[image_path] = result
        
        if result['success']:
            print(f"  ✓ Foreground extracted successfully")
        else:
            print(f"  ✗ Error: {result['error']}")
    
    return results

def create_transparent_background(image_path, rectangle=None, output_dir=None):
    """
    Create image with transparent background (PNG format)
    
    Args:
        image_path (str): Path to input image
        rectangle (tuple): ROI rectangle
        output_dir (str): Output directory
    
    Returns:
        str: Path to output PNG file
    """
    result = extract_foreground_grabcut(image_path, rectangle)
    
    if not result['success']:
        return None
    
    # Convert to RGBA (add alpha channel)
    segmented = result['segmented_image']
    mask = result['mask']
    
    # Create RGBA image
    rgba_image = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGBA)
    
    # Set alpha channel based on mask
    rgba_image[:, :, 3] = mask * 255
    
    # Save as PNG
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_filename}_transparent.png"
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = output_filename
    
    cv2.imwrite(output_path, rgba_image)
    return output_path

# Example usage functions
def example_face_extraction(image_path):
    """
    Example: Extract a face/person from image
    Assumes the subject is in the center portion of the image
    """
    print("Extracting foreground (face/person) from image...")
    result = extract_foreground_grabcut(image_path, show_result=True)
    
    if result['success']:
        print("Success! Check the displayed result.")
        return result['output_path']
    else:
        print(f"Failed: {result['error']}")
        return None

def example_custom_region(image_path, x, y, w, h):
    """
    Example: Extract foreground from custom region
    """
    print(f"Extracting foreground from region ({x}, {y}, {w}, {h})...")
    result = extract_with_custom_roi(image_path, x, y, w, h, iterations=5)
    
    if result['success']:
        print("Success! Foreground extracted.")
        return result['output_path']
    else:
        print(f"Failed: {result['error']}")
        return None
    















