import cv2
import numpy as np
# SSIM requires scikit-image (skimage)
from skimage.metrics import structural_similarity as ssim

def robust_image_compare(image_path_A, image_path_B, ssim_tolerance=0.995):
    """
    Compares two images using a two-stage approach:
    1. Fast, strict pixel-for-pixel check (numpy.array_equal).
    2. Tolerant structural check (SSIM) if stage 1 fails.

    Returns: True if images are considered equal, False otherwise.
    """
    
    # --- Stage 0: Load and Validate Images ---
    try:
        imgA = cv2.imread(image_path_A)
        imgB = cv2.imread(image_path_B)

        if imgA is None or imgB is None:
            print("Error: Could not load one or both images.")
            return False
        
        if imgA.shape != imgB.shape:
            # If shapes are different, they are definitely unequal
            print("❌ Result: Images are NOT equal (Different dimensions).")
            return False
            
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return False
        
    # --- Stage 1: Quick Pixel-Perfect Check (Fastest) ---
    if np.array_equal(imgA, imgB):
        print("✅ Result: Images are pixel-perfect identical (Fast NumPy check).")
        return True

    # --- Stage 2: Tolerant Structural Check (SSIM) ---
    print("⚠️ Images are NOT strictly identical. Checking structural similarity...")
    
    # SSIM works best on grayscale for performance/simplicity in comparison
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM score
    score, _ = ssim(grayA, grayB, full=True)

    print(f"   - Calculated SSIM Score: {score:.4f}")
    
    # Check if the score is above the tolerance threshold
    if score >= ssim_tolerance:
        print(f"✅ Result: Images are considered EQUAL (Score > {ssim_tolerance})")
        return True
    else:
        print(f"❌ Result: Images are NOT EQUAL (Structural difference detected).")
        return False

# =================================================================
# --- Usage Example ---
# =================================================================
# NOTE: You need two image files named 'image1.png' and 'image2.png' 
# in the same directory for this code to run successfully.

# Scenario 1: Images are identical
# is_equal_scenario_1 = robust_image_compare('identical_A.png', 'identical_A.png')

# Scenario 2: Images are the same except for a tiny cursor (requires SSIM tolerance)
is_equal_scenario_2 = robust_image_compare(
    'prubsn\IL 3.png', 
    'prubsn\IL 4.png', 
    ssim_tolerance=0.995 # Set a high tolerance
)

# Scenario 3: Images have a major difference (e.g., a missing button)
# is_equal_scenario_3 = robust_image_compare('screenshot_A.png', 'screenshot_C_major_change.png')