from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import os
import base64
from datetime import datetime
from PIL import Image
import io
import math

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class ImageProcessorFromScratch:
    """All image processing functions implemented from scratch - NO BUILT-IN FUNCTIONS"""
    
    # ====================  Functions ====================
    @staticmethod
    def read_image(file):
        """Read uploaded image file"""
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def image_to_base64(img):
        """Convert image to base64 for HTML display"""
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        _, buffer = cv2.imencode('.png', img_bgr)
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

    @staticmethod
    def save_image(img, filename):
        """Save image to disk"""
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), img_bgr)

    @staticmethod
    def resize_to_match(img1, img2):
        """Resize img2 to match img1 dimensions"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if h1 != h2 or w1 != w2:
            scale_h = h1 / h2
            scale_w = w1 / w2
            new_h = int(h2 * scale_h)
            new_w = int(w2 * scale_w)
            
            if len(img2.shape) == 3:
                result = np.zeros((new_h, new_w, img2.shape[2]), dtype=img2.dtype)
                for i in range(new_h):
                    for j in range(new_w):
                        src_i = int(i / scale_h)
                        src_j = int(j / scale_w)
                        src_i = min(src_i, h2 - 1)
                        src_j = min(src_j, w2 - 1)
                        result[i, j] = img2[src_i, src_j]
            else:
                result = np.zeros((new_h, new_w), dtype=img2.dtype)
                for i in range(new_h):
                    for j in range(new_w):
                        src_i = int(i / scale_h)
                        src_j = int(j / scale_w)
                        src_i = min(src_i, h2 - 1)
                        src_j = min(src_j, w2 - 1)
                        result[i, j] = img2[src_i, src_j]
            return result
        return img2

    # ==================== Binary Image Creation ====================
    @staticmethod
    def create_binary_image(shape, pattern='checkerboard'):
        """Create binary image from scratch"""
        img = np.zeros(shape, dtype=np.uint8)
        if pattern == 'checkerboard':
            for i in range(shape[0]):
                for j in range(shape[1]):
                    img[i, j] = 255 if (i + j) % 2 == 0 else 0
        elif pattern == 'horizontal_stripes':
            stripe_height = shape[0] // 8
            for i in range(shape[0]):
                img[i, :] = 255 if (i // stripe_height) % 2 == 0 else 0
        elif pattern == 'vertical_stripes':
            stripe_width = shape[1] // 8
            for j in range(shape[1]):
                img[:, j] = 255 if (j // stripe_width) % 2 == 0 else 0
        return img

    # ====================  Image Type Conversions ====================
    @staticmethod
    def rgb_to_grayscale_avg(img):
        """Convert RGB to grayscale using simple averaging from scratch"""
        if len(img.shape) == 2:
            return img
        h, w = img.shape[:2]
        result = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                result[i, j] = (int(img[i, j, 0]) + int(img[i, j, 1]) + int(img[i, j, 2])) // 3
        return result

    @staticmethod
    def rgb_to_grayscale_weighted(img):
        """Convert RGB to grayscale using weighted luminance from scratch"""
        if len(img.shape) == 2:
            return img
        h, w = img.shape[:2]
        result = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                result[i, j] = int(0.299 * img[i, j, 0] + 0.587 * img[i, j, 1] + 0.114 * img[i, j, 2])
        return result

    @staticmethod
    def to_binary(img, threshold=128):
        """Convert to binary image from scratch"""
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        h, w = gray.shape
        result = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                result[i, j] = 255 if gray[i, j] >= threshold else 0
        return result

    @staticmethod
    def extract_red_channel(img):
        """Extract red channel from scratch"""
        if len(img.shape) != 3:
            return img
        h, w = img.shape[:2]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                result[i, j, 0] = img[i, j, 0]
        return result

    @staticmethod
    def extract_green_channel(img):
        """Extract green channel from scratch"""
        if len(img.shape) != 3:
            return img
        h, w = img.shape[:2]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                result[i, j, 1] = img[i, j, 1]
        return result

    @staticmethod
    def extract_blue_channel(img):
        """Extract blue channel from scratch"""
        if len(img.shape) != 3:
            return img
        h, w = img.shape[:2]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                result[i, j, 2] = img[i, j, 2]
        return result

    # ==================== Arithmetic Operations ====================
    @staticmethod
    def add_constant(img, value):
        """Add constant to image from scratch"""
        if len(img.shape) == 3:
            h, w, c = img.shape
            result = np.zeros((h, w, c), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        val = int(img[i, j, k]) + value
                        result[i, j, k] = 255 if val > 255 else (0 if val < 0 else val)
            return result
        else:
            h, w = img.shape
            result = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    val = int(img[i, j]) + value
                    result[i, j] = 255 if val > 255 else (0 if val < 0 else val)
            return result

    @staticmethod
    def subtract_constant(img, value):
        """Subtract constant from image from scratch"""
        return ImageProcessorFromScratch.add_constant(img, -value)

    @staticmethod
    def multiply_constant(img, factor):
        """Multiply image by constant from scratch"""
        if len(img.shape) == 3:
            h, w, c = img.shape
            result = np.zeros((h, w, c), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        val = int(img[i, j, k] * factor)
                        result[i, j, k] = 255 if val > 255 else (0 if val < 0 else val)
            return result
        else:
            h, w = img.shape
            result = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    val = int(img[i, j] * factor)
                    result[i, j] = 255 if val > 255 else (0 if val < 0 else val)
            return result

    @staticmethod
    def divide_constant(img, factor):
        """Divide image by constant from scratch"""
        if factor == 0:
            return img
        return ImageProcessorFromScratch.multiply_constant(img, 1.0 / factor)

    @staticmethod
    def complement(img):
        """Image complement (negative) from scratch"""
        if len(img.shape) == 3:
            h, w, c = img.shape
            result = np.zeros((h, w, c), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        result[i, j, k] = 255 - img[i, j, k]
            return result
        else:
            h, w = img.shape
            result = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    result[i, j] = 255 - img[i, j]
            return result

    @staticmethod
    def solarize(img, threshold=128, invert_dark=True):
        """Solarization effect from scratch"""
        if len(img.shape) == 3:
            h, w, c = img.shape
            result = np.zeros((h, w, c), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        if invert_dark:
                            if img[i, j, k] < threshold:
                                result[i, j, k] = 255 - img[i, j, k]
                            else:
                                result[i, j, k] = img[i, j, k]
                        else:
                            if img[i, j, k] > threshold:
                                result[i, j, k] = 255 - img[i, j, k]
                            else:
                                result[i, j, k] = img[i, j, k]
            return result
        else:
            h, w = img.shape
            result = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    if invert_dark:
                        if img[i, j] < threshold:
                            result[i, j] = 255 - img[i, j]
                        else:
                            result[i, j] = img[i, j]
                    else:
                        if img[i, j] > threshold:
                            result[i, j] = 255 - img[i, j]
                        else:
                            result[i, j] = img[i, j]
            return result

    @staticmethod
    def add_images(img1, img2):
        """Add two images pixel by pixel from scratch"""
        img2_resized = ImageProcessorFromScratch.resize_to_match(img1, img2)
        
        if len(img1.shape) == 3:
            h, w, c = img1.shape
            result = np.zeros((h, w, c), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        val = int(img1[i, j, k]) + int(img2_resized[i, j, k])
                        result[i, j, k] = 255 if val > 255 else val
            return result
        else:
            h, w = img1.shape
            result = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    val = int(img1[i, j]) + int(img2_resized[i, j])
                    result[i, j] = 255 if val > 255 else val
            return result

    @staticmethod
    def subtract_images(img1, img2):
        """Subtract img2 from img1 pixel by pixel from scratch"""
        img2_resized = ImageProcessorFromScratch.resize_to_match(img1, img2)
        
        if len(img1.shape) == 3:
            h, w, c = img1.shape
            result = np.zeros((h, w, c), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        val = int(img1[i, j, k]) - int(img2_resized[i, j, k])
                        result[i, j, k] = 0 if val < 0 else val
            return result
        else:
            h, w = img1.shape
            result = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    val = int(img1[i, j]) - int(img2_resized[i, j])
                    result[i, j] = 0 if val < 0 else val
            return result

    @staticmethod
    def absolute_difference(img1, img2):
        """Absolute difference between two images from scratch"""
        img2_resized = ImageProcessorFromScratch.resize_to_match(img1, img2)
        
        if len(img1.shape) == 3:
            h, w, c = img1.shape
            result = np.zeros((h, w, c), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        val = abs(int(img1[i, j, k]) - int(img2_resized[i, j, k]))
                        result[i, j, k] = val
            return result
        else:
            h, w = img1.shape
            result = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    result[i, j] = abs(int(img1[i, j]) - int(img2_resized[i, j]))
            return result

    @staticmethod
    def blend_images(img1, img2, alpha=0.5):
        """Blend two images with alpha weight from scratch"""
        img2_resized = ImageProcessorFromScratch.resize_to_match(img1, img2)
        
        if len(img1.shape) == 3:
            h, w, c = img1.shape
            result = np.zeros((h, w, c), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        val = alpha * img1[i, j, k] + (1 - alpha) * img2_resized[i, j, k]
                        result[i, j, k] = int(255 if val > 255 else (0 if val < 0 else val))
            return result
        else:
            h, w = img1.shape
            result = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    val = alpha * img1[i, j] + (1 - alpha) * img2_resized[i, j]
                    result[i, j] = int(255 if val > 255 else (0 if val < 0 else val))
            return result

    @staticmethod
    def background_subtraction(img1, img2, threshold=30):
        """Extract foreground by subtracting background from scratch"""
        diff = ImageProcessorFromScratch.absolute_difference(img1, img2)
        
        if len(diff.shape) == 3:
            h, w, c = diff.shape
            result = np.zeros((h, w, c), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    if diff[i, j, 0] > threshold or diff[i, j, 1] > threshold or diff[i, j, 2] > threshold:
                        result[i, j] = img1[i, j]
                    else:
                        result[i, j] = [0, 0, 0]
            return result
        else:
            h, w = diff.shape
            result = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    if diff[i, j] > threshold:
                        result[i, j] = img1[i, j]
                    else:
                        result[i, j] = 0
            return result

    @staticmethod
    def get_background(img1, img2, threshold=30):
        """Extract background from two images from scratch"""
        diff = ImageProcessorFromScratch.absolute_difference(img1, img2)
        
        if len(diff.shape) == 3:
            h, w, c = diff.shape
            result = np.zeros((h, w, c), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    if diff[i, j, 0] <= threshold and diff[i, j, 1] <= threshold and diff[i, j, 2] <= threshold:
                        result[i, j] = img1[i, j]
                    else:
                        result[i, j] = [0, 0, 0]
            return result
        else:
            h, w = diff.shape
            result = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    if diff[i, j] <= threshold:
                        result[i, j] = img1[i, j]
                    else:
                        result[i, j] = 0
            return result

    # ==================== Histogram ====================
    @staticmethod
    def compute_histogram(img):
        """Compute histogram from scratch"""
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        hist = [0] * 256
        h, w = gray.shape
        for i in range(h):
            for j in range(w):
                hist[gray[i, j]] += 1
        return hist

    @staticmethod
    def get_histogram_data(img):
        """Return histogram data for plotting"""
        hist = ImageProcessorFromScratch.compute_histogram(img)
        return hist

    # ==================== Contrast Enhancemnt  ====================
    @staticmethod
    def histogram_equalization(img):
        """Histogram equalization from scratch"""
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        
        h, w = gray.shape
        total_pixels = h * w
        
        # Compute histogram
        hist = [0] * 256
        for i in range(h):
            for j in range(w):
                hist[gray[i, j]] += 1
        
        # Compute CDF
        cdf = [0] * 256
        cdf[0] = hist[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + hist[i]
        
        # Find minimum non-zero CDF value
        cdf_min = 0
        for i in range(256):
            if cdf[i] > 0:
                cdf_min = cdf[i]
                break
        
        # Build lookup table
        lut = [0] * 256
        for i in range(256):
            if cdf[i] > 0:
                lut[i] = int(round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255))
                lut[i] = max(0, min(255, lut[i]))
        
        # Apply lookup table
        result = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                result[i, j] = lut[gray[i, j]]
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

    @staticmethod
    def histogram_stretching(img):
        """Histogram stretching (contrast stretching) from scratch"""
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Find min and max pixel values
        min_val = 255
        max_val = 0
        for i in range(h):
            for j in range(w):
                if gray[i, j] < min_val:
                    min_val = gray[i, j]
                if gray[i, j] > max_val:
                    max_val = gray[i, j]
        
        # Stretch to full range 0-255
        result = np.zeros((h, w), dtype=np.uint8)
        if max_val == min_val:
            return gray
        
        for i in range(h):
            for j in range(w):
                result[i, j] = int((gray[i, j] - min_val) / (max_val - min_val) * 255)
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

    # ==================== Filters ====================
    @staticmethod
    def mean_filter(img, kernel_size=3):
        """Mean filter from scratch - no built-in functions"""
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        
        h, w = gray.shape
        pad = kernel_size // 2
        result = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(pad, h - pad):
            for j in range(pad, w - pad):
                total = 0
                count = 0
                for ki in range(-pad, pad + 1):
                    for kj in range(-pad, pad + 1):
                        total += gray[i + ki, j + kj]
                        count += 1
                result[i, j] = total // count
        
        for i in range(h):
            for j in range(w):
                if i < pad or i >= h - pad or j < pad or j >= w - pad:
                    result[i, j] = gray[i, j]
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

    @staticmethod
    def median_filter(img, kernel_size=3):
        """Median filter from scratch - no built-in functions"""
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        
        h, w = gray.shape
        pad = kernel_size // 2
        result = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(pad, h - pad):
            for j in range(pad, w - pad):
                neighbors = []
                for ki in range(-pad, pad + 1):
                    for kj in range(-pad, pad + 1):
                        neighbors.append(gray[i + ki, j + kj])
                for x in range(len(neighbors)):
                    for y in range(x + 1, len(neighbors)):
                        if neighbors[x] > neighbors[y]:
                            neighbors[x], neighbors[y] = neighbors[y], neighbors[x]
                result[i, j] = neighbors[len(neighbors) // 2]
        
        for i in range(h):
            for j in range(w):
                if i < pad or i >= h - pad or j < pad or j >= w - pad:
                    result[i, j] = gray[i, j]
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

    @staticmethod
    def gaussian_filter_scratch(img, kernel_size=5, sigma=1.0):
        """Gaussian filter from scratch - no built-in functions"""
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        kernel_sum = 0
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center
                kernel[i, j] = (1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x**2 + y**2) / (2 * sigma**2))
                kernel_sum += kernel[i, j]
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] /= kernel_sum
        
        h, w = gray.shape
        pad = kernel_size // 2
        result = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(pad, h - pad):
            for j in range(pad, w - pad):
                total = 0
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        total += gray[i + ki - pad, j + kj - pad] * kernel[ki, kj]
                result[i, j] = int(total)
        
        for i in range(h):
            for j in range(w):
                if i < pad or i >= h - pad or j < pad or j >= w - pad:
                    result[i, j] = gray[i, j]
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

    # ==================== Edge Detection   ====================
    # 1. Convert image to grayscale
    # 2. Apply gradient kernels (convolution) -> Gx, Gy
    # 3. Compute gradient magnitude = sqrt(Gx^2 + Gy^2)
    # 4. Compute threshold = average of all gradient magnitude values
    # 5. Apply threshold: n > threshold -> 255 (edge), n <= threshold -> 0

    @staticmethod
    def sobel_edge_detection(img):
        """
        Sobel kernels:
        Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        Gradient magnitude = sqrt(Gx^2 + Gy^2)
        Threshold = average of all gradient values
        """
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Sobel kernels (matching Lecture )
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        
        gradient = np.zeros((h, w), dtype=np.float64)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                gx = 0
                gy = 0
                for ki in range(3):
                    for kj in range(3):
                        pixel = int(gray[i + ki - 1, j + kj - 1])
                        gx += pixel * sobel_x[ki][kj]
                        gy += pixel * sobel_y[ki][kj]
                gradient[i, j] = math.sqrt(gx**2 + gy**2)
        
        # Step 4: Compute threshold = average of all gradient magnitude values
        total = 0.0
        count = 0
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                total += gradient[i, j]
                count += 1
        threshold = total / count if count > 0 else 0
        
        # Step 5: Apply threshold -> binary edge map
        result = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                result[i, j] = 255 if gradient[i, j] > threshold else 0
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

    @staticmethod
    def prewitt_edge_detection(img):
        """
        Prewitt kernels:
        Kx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        Ky = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
        Gradient magnitude = sqrt(Gx^2 + Gy^2)
        Threshold = average of all gradient values
        """
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Prewitt kernels 
        prewitt_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        prewitt_y = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
        
        gradient = np.zeros((h, w), dtype=np.float64)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                gx = 0
                gy = 0
                for ki in range(3):
                    for kj in range(3):
                        pixel = int(gray[i + ki - 1, j + kj - 1])
                        gx += pixel * prewitt_x[ki][kj]
                        gy += pixel * prewitt_y[ki][kj]
                gradient[i, j] = math.sqrt(gx**2 + gy**2)
        
        # Threshold = average of all gradient values
        total = 0.0
        count = 0
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                total += gradient[i, j]
                count += 1
        threshold = total / count if count > 0 else 0
        
        # Apply threshold -> binary edge map
        result = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                result[i, j] = 255 if gradient[i, j] > threshold else 0
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

    @staticmethod
    def roberts_edge_detection(img):
        """
        Roberts kernels:
        Gx = [[+1, 0], [0, -1]]
        Gy = [[0, +1], [-1, 0]]
        Gradient magnitude = sqrt(Gx^2 + Gy^2)
        Threshold = average of all gradient values
        """
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Roberts kernels (2x2) 
        roberts_x = [[1, 0], [0, -1]]
        roberts_y = [[0, 1], [-1, 0]]
        
        gradient = np.zeros((h, w), dtype=np.float64)
        
        for i in range(h - 1):
            for j in range(w - 1):
                gx = (int(gray[i, j]) * roberts_x[0][0] + 
                      int(gray[i, j+1]) * roberts_x[0][1] +
                      int(gray[i+1, j]) * roberts_x[1][0] +
                      int(gray[i+1, j+1]) * roberts_x[1][1])
                gy = (int(gray[i, j]) * roberts_y[0][0] +
                      int(gray[i, j+1]) * roberts_y[0][1] +
                      int(gray[i+1, j]) * roberts_y[1][0] +
                      int(gray[i+1, j+1]) * roberts_y[1][1])
                gradient[i, j] = math.sqrt(gx**2 + gy**2)
        
        # Threshold = average of all gradient values
        total = 0.0
        count = 0
        for i in range(h - 1):
            for j in range(w - 1):
                total += gradient[i, j]
                count += 1
        threshold = total / count if count > 0 else 0
        
        # Apply threshold -> binary edge map
        result = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                result[i, j] = 255 if gradient[i, j] > threshold else 0
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

    @staticmethod
    def laplacian_edge_detection(img):
        """
        Laplacian kernel (8-neighbor, diagonal):
        [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
        """
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        
        h, w = gray.shape
        
        # Laplacian kernel 
        laplacian = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
        
        result_float = np.zeros((h, w), dtype=np.float64)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                lap = 0
                for ki in range(3):
                    for kj in range(3):
                        lap += int(gray[i + ki - 1, j + kj - 1]) * laplacian[ki][kj]
                result_float[i, j] = abs(lap)
        
        # Normalize to 0-255
        max_val = 0.0
        for i in range(h):
            for j in range(w):
                if result_float[i, j] > max_val:
                    max_val = result_float[i, j]
        
        result = np.zeros((h, w), dtype=np.uint8)
        if max_val > 0:
            for i in range(h):
                for j in range(w):
                    result[i, j] = int((result_float[i, j] / max_val) * 255)
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

    # ==================== Noise Addition ====================
    @staticmethod
    def add_salt_pepper_noise(img, noise_ratio=0.05):
        """Add salt and pepper noise from scratch"""
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
            is_rgb = True
        else:
            gray = img.copy()
            is_rgb = False
        
        h, w = gray.shape
        total_pixels = h * w
        num_noisy = int(noise_ratio * total_pixels)
        
        for _ in range(num_noisy):
            i = np.random.randint(0, h)
            j = np.random.randint(0, w)
            gray[i, j] = 255
        
        for _ in range(num_noisy):
            i = np.random.randint(0, h)
            j = np.random.randint(0, w)
            gray[i, j] = 0
        
        if is_rgb:
            result = np.zeros_like(img)
            for k in range(3):
                result[:, :, k] = gray
            return result
        return gray

    # ==================== Morphological Operations ====================
    @staticmethod
    def binary_dilation(img, kernel_size=3):
        """Binary dilation from scratch"""
        if len(img.shape) == 3:
            binary = ImageProcessorFromScratch.to_binary(img, 128)
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(binary)
        else:
            binary = ImageProcessorFromScratch.to_binary(img, 128)
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(binary) if len(binary.shape) == 3 else binary
        
        h, w = gray.shape
        pad = kernel_size // 2
        result = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                if gray[i, j] == 255:
                    for ki in range(-pad, pad + 1):
                        for kj in range(-pad, pad + 1):
                            ni = i + ki
                            nj = j + kj
                            if 0 <= ni < h and 0 <= nj < w:
                                result[ni, nj] = 255
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

    @staticmethod
    def binary_erosion(img, kernel_size=3):
        """Binary erosion from scratch"""
        if len(img.shape) == 3:
            binary = ImageProcessorFromScratch.to_binary(img, 128)
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(binary)
        else:
            binary = ImageProcessorFromScratch.to_binary(img, 128)
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(binary) if len(binary.shape) == 3 else binary
        
        h, w = gray.shape
        pad = kernel_size // 2
        result = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(pad, h - pad):
            for j in range(pad, w - pad):
                all_white = True
                for ki in range(-pad, pad + 1):
                    for kj in range(-pad, pad + 1):
                        if gray[i + ki, j + kj] != 255:
                            all_white = False
                            break
                    if not all_white:
                        break
                if all_white:
                    result[i, j] = 255
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

    @staticmethod
    def binary_opening(img, kernel_size=3):
        """Binary opening (erosion then dilation) from scratch"""
        eroded = ImageProcessorFromScratch.binary_erosion(img, kernel_size)
        opened = ImageProcessorFromScratch.binary_dilation(eroded, kernel_size)
        return opened

    @staticmethod
    def binary_closing(img, kernel_size=3):
        """Binary closing (dilation then erosion) from scratch"""
        dilated = ImageProcessorFromScratch.binary_dilation(img, kernel_size)
        closed = ImageProcessorFromScratch.binary_erosion(dilated, kernel_size)
        return closed

    # ==================== Thresholding & Segmentation ====================
    @staticmethod
    def otsu_threshold(img):
        """Otsu's thresholding method from scratch"""
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        
        hist = ImageProcessorFromScratch.compute_histogram(gray)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        best_threshold = 0
        best_variance = 0
        
        for t in range(256):
            w_b = sum(hist[:t])
            if w_b == 0:
                continue
            mean_b = sum(i * hist[i] for i in range(t)) / w_b
            
            w_f = total_pixels - w_b
            if w_f == 0:
                break
            mean_f = sum(i * hist[i] for i in range(t, 256)) / w_f
            
            variance = w_b * w_f * (mean_b - mean_f) ** 2
            
            if variance > best_variance:
                best_variance = variance
                best_threshold = t
        
        result = np.zeros_like(gray)
        h, w = gray.shape
        for i in range(h):
            for j in range(w):
                result[i, j] = 255 if gray[i, j] >= best_threshold else 0
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

    @staticmethod
    def adaptive_threshold_mean(img, block_size=11, C=2):
        """Adaptive thresholding using mean from scratch"""
        if len(img.shape) == 3:
            gray = ImageProcessorFromScratch.rgb_to_grayscale_weighted(img)
        else:
            gray = img
        
        h, w = gray.shape
        pad = block_size // 2
        result = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                i_start = max(0, i - pad)
                i_end = min(h, i + pad + 1)
                j_start = max(0, j - pad)
                j_end = min(w, j + pad + 1)
                
                local_sum = 0
                local_count = 0
                for ni in range(i_start, i_end):
                    for nj in range(j_start, j_end):
                        local_sum += gray[ni, nj]
                        local_count += 1
                
                local_mean = local_sum // local_count
                threshold = local_mean - C
                
                result[i, j] = 255 if gray[i, j] >= threshold else 0
        
        if len(img.shape) == 3:
            result_rgb = np.zeros_like(img)
            for k in range(3):
                result_rgb[:, :, k] = result
            return result_rgb
        return result

# Import cv2 only for image I/O (not for processing)
import cv2
processor = ImageProcessorFromScratch()


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    img = processor.read_image(file)
    processor.save_image(img, 'original.png')
    processor.save_image(img, 'current.png')
    
    img_base64 = processor.image_to_base64(img)
    return jsonify({'success': True, 'image': img_base64})


@app.route('/upload_second', methods=['POST'])
def upload_second():
    file = request.files['image']
    img = processor.read_image(file)
    processor.save_image(img, 'second.png')
    
    img_base64 = processor.image_to_base64(img)
    return jsonify({'success': True, 'image': img_base64})


@app.route('/apply', methods=['POST'])
def apply():
    data = request.json
    operation = data.get('operation')
    params = data.get('params', {})
    
    current_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current.png')
    img = cv2.imread(current_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    operations = {
        'grayscale_avg': lambda: processor.rgb_to_grayscale_avg(img),
        'grayscale_weighted': lambda: processor.rgb_to_grayscale_weighted(img),
        'binary': lambda: processor.to_binary(img, params.get('threshold', 128)),
        'red_channel': lambda: processor.extract_red_channel(img),
        'green_channel': lambda: processor.extract_green_channel(img),
        'blue_channel': lambda: processor.extract_blue_channel(img),
        
      
        'add_constant': lambda: processor.add_constant(img, params.get('value', 50)),
        'subtract_constant': lambda: processor.subtract_constant(img, params.get('value', 50)),
        'multiply_constant': lambda: processor.multiply_constant(img, params.get('factor', 1.5)),
        'divide_constant': lambda: processor.divide_constant(img, params.get('factor', 1.5)),
        'complement': lambda: processor.complement(img),
        'solarize': lambda: processor.solarize(img, params.get('threshold', 128), params.get('invert_dark', True)),
        
        'histogram_equalization': lambda: processor.histogram_equalization(img),
        'histogram_stretching': lambda: processor.histogram_stretching(img),
        
        'mean_filter': lambda: processor.mean_filter(img, params.get('kernel', 3)),
        'median_filter': lambda: processor.median_filter(img, params.get('kernel', 3)),
        'gaussian_filter': lambda: processor.gaussian_filter_scratch(img, params.get('kernel', 5), params.get('sigma', 1.0)),
        
        # Edge Detection 
        'sobel': lambda: processor.sobel_edge_detection(img),
        'prewitt': lambda: processor.prewitt_edge_detection(img),
        'roberts': lambda: processor.roberts_edge_detection(img),
        'laplacian': lambda: processor.laplacian_edge_detection(img),
        
        # Noise
        'salt_pepper': lambda: processor.add_salt_pepper_noise(img, params.get('ratio', 0.05)),
        'gaussian_noise': lambda: processor.add_gaussian_noise(img, sigma=params.get('sigma', 25)),
        
        # Morphology
        'dilation': lambda: processor.binary_dilation(img, params.get('kernel', 3)),
        'erosion': lambda: processor.binary_erosion(img, params.get('kernel', 3)),
        'opening': lambda: processor.binary_opening(img, params.get('kernel', 3)),
        'closing': lambda: processor.binary_closing(img, params.get('kernel', 3)),
        
        # Segmentation
        'otsu': lambda: processor.otsu_threshold(img),
        'adaptive_mean': lambda: processor.adaptive_threshold_mean(img, params.get('block_size', 11), params.get('C', 2)),
    }
    
    if operation in operations:
        result = operations[operation]()
        processor.save_image(result, 'current.png')
        result_base64 = processor.image_to_base64(result)
        return jsonify({'success': True, 'image': result_base64})
    
    return jsonify({'error': 'Unknown operation'}), 400


@app.route('/image_operations', methods=['POST'])
def image_operations():
    """Operations that require two images"""
    data = request.json
    operation = data.get('operation')
    params = data.get('params', {})
    
    current_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current.png')
    second_path = os.path.join(app.config['UPLOAD_FOLDER'], 'second.png')
    
    img1 = cv2.imread(current_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    if not os.path.exists(second_path):
        return jsonify({'error': 'Please upload a second image first'}), 400
    
    img2 = cv2.imread(second_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    operations = {
        'add_images': lambda: processor.add_images(img1, img2),
        'subtract_images': lambda: processor.subtract_images(img1, img2),
        'absolute_difference': lambda: processor.absolute_difference(img1, img2),
        'blend_images': lambda: processor.blend_images(img1, img2, params.get('alpha', 0.5)),
        'foreground': lambda: processor.background_subtraction(img1, img2, params.get('threshold', 30)),
        'background': lambda: processor.get_background(img1, img2, params.get('threshold', 30)),
        'watermark': lambda: processor.add_watermark(img1, params.get('text', 'WATERMARK'), params.get('position', 'bottom-right'), params.get('opacity', 0.7)),
    }
    
    if operation in operations:
        result = operations[operation]()
        processor.save_image(result, 'current.png')
        result_base64 = processor.image_to_base64(result)
        return jsonify({'success': True, 'image': result_base64})
    
    return jsonify({'error': 'Unknown operation'}), 400


@app.route('/get_histogram', methods=['POST'])
def get_histogram():
    """Return histogram data for the current image"""
    current_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current.png')
    img = cv2.imread(current_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    hist = processor.compute_histogram(img)
    return jsonify({'success': True, 'histogram': hist})


@app.route('/get_both_histograms', methods=['POST'])
def get_both_histograms():
    """Return histogram data for both original and current images"""
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original.png')
    current_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current.png')
    
    result = {'success': True}
    
    if os.path.exists(original_path):
        img_orig = cv2.imread(original_path)
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        result['before'] = processor.compute_histogram(img_orig)
    
    if os.path.exists(current_path):
        img_curr = cv2.imread(current_path)
        img_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2RGB)
        result['after'] = processor.compute_histogram(img_curr)
    
    return jsonify(result)


@app.route('/reset', methods=['POST'])
def reset():
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original.png')
    img = cv2.imread(original_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processor.save_image(img, 'current.png')
    img_base64 = processor.image_to_base64(img)
    return jsonify({'success': True, 'image': img_base64})


@app.route('/save', methods=['POST'])
def save():
    current_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current.png')
    return send_file(current_path, as_attachment=True, download_name=f'processed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
