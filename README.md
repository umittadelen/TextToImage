## TextToImage
**TextToImage** is a free, open-source text-to-image generation tool designed for ease of use, allowing anyone to run advanced models on their computer with customizable parameters and progress tracking.

### Features
- **Progress Tracking**
- **Nudity Detection** (via **NudeDetector**) (not good enough)
- **Seed Control** for reproducibility
- **Adjustable CFG** (Classifier-Free Guidance) for creative flexibility

For setup instructions, see the **Installation Guide**.

---

# Installation Guide

### Prerequisites
- **A good GPU** (it will run slowly if you don't have)
- **CUDA**
- **VS Code** (recommended editor)

### Steps
1. **Clone the Repository**:
```bash
git clone https://github.com/umittadelen/TextToImage.git
cd TextToImage
```
2. **Change the IP**:
open the app.py and change the address at this line to your local IP
```bash
app.run(host='192.168.0.4', port=8080, debug=False)
```
3. **Install Dependencies**: (if the code don't automatically installs the libs)
```bash
pip install -r requirements.txt
```
