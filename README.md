# ImageLab

> Image processing toolkit — all algorithms implemented from scratch.

---

## About

**ImageLab** is a full-stack image processing web application built as part of an **Image Processing university course**. It demonstrates a wide range of classical image processing techniques — all implemented **from scratch using NumPy**, without relying on any built-in processing functions from OpenCV or Pillow.

Users can upload images, apply transformations in real time, view before/after comparisons side by side, inspect pixel histograms, and save the processed results — all through an interactive dark-themed UI.

> **Note:** OpenCV is used **only for reading and writing** image files. Every processing algorithm — filtering, edge detection, morphology, thresholding — is implemented entirely with NumPy array operations and pure Python loops.

---

## Features & Operations

### Image Type Conversions
- Grayscale — Average method
- Grayscale — Weighted luminance (ITU-R BT.601)
- Binary (adjustable threshold)
- Red / Green / Blue channel extraction

### Arithmetic Operations
- Add / Subtract constant
- Multiply / Divide constant
- Complement (negative)
- Solarize effect

### Dual-Image Operations
- Image addition & subtraction
- Absolute difference
- Alpha blending
- Background subtraction

### Histogram Analysis
- Histogram equalization
- Histogram stretching
- Live before/after histogram overlay
- Dual-image histogram comparison

### Spatial Filtering
- Mean filter (box blur)
- Median filter
- Gaussian filter

### Edge Detection
- Sobel operator
- Prewitt operator
- Roberts operator
- Laplacian operator

### Morphological Operations
- Dilation & Erosion
- Opening & Closing

### Noise & Segmentation
- Salt & pepper noise
- Gaussian noise
- Otsu's thresholding
- Adaptive mean thresholding

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask (Python) |
| Algorithms | NumPy (all pixel-level logic) |
| Image I/O | OpenCV, Pillow |
| Histogram charts | Chart.js |
| Frontend | HTML / CSS / Vanilla JS |

---

## Project Structure

```
ImageLab/
├── app.py                # Flask app + all image processing logic
├── templates/
│   └── index.html        # Single-page frontend (dark UI, Chart.js histograms)
└── uploads/              # Runtime folder for uploaded & processed images (auto-created)
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Serve the main UI |
| `POST` | `/upload` | Upload the primary image |
| `POST` | `/upload_second` | Upload a second image for dual-image ops |
| `POST` | `/apply` | Apply a processing operation (JSON body) |
| `GET` | `/get_histogram` | Return histogram data for current image |
| `GET` | `/get_both_histograms` | Return histograms for both images |
| `POST` | `/reset` | Reset to original uploaded image |
| `GET` | `/save` | Download the processed image as PNG |

---

## UI Highlights

- **Dark theme** — deep navy/black palette with violet & green accents
- **Before / After split view** — side-by-side original vs processed image
- **Live histogram panel** — toggle Chart.js histogram overlay at any time
- **Dual-image mode** — upload a second image for blend, diff, and subtraction ops
- **Download result** — save the processed output as PNG
- **Max upload size** — 50 MB

---

## 👩‍💻 Author

**Jana Hany**  


## 📬 Contact

[GitHub](https://github.com/jana-h-any): @jana-h-any

[LinkedIn](www.linkedin.com/in/jana-hany) : [jana-hany]

[Email](janahanymostafa.h@gmail.com): [janahanymostafa.h@gmail.com]
