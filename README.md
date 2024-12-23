# TextToImage

**TextToImage** is a free, open-source text-to-image generation tool designed for ease of use, allowing anyone to run advanced models on their computer with customizable parameters and progress tracking.

## Features

- **Progress Tracking**
- **Seed Control** for reproducibility
- **Adjustable CFG** (Classifier-Free Guidance) for creative flexibility
- **Support for multiple schedulers**
- **ControlNet support**
- **Image-to-Image and Text-to-Image generation**

## Installation Guide

### Prerequisites

- **A good GPU** (it will run slowly if you don't have one)
- **CUDA**
- **VS Code** (recommended editor)

### Steps

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/umittadelen/TextToImage.git
    cd TextToImage
    ```

2. **Change the IP**:
    Open the [app.py](http://_vscodecontentref_/0) and change the address at this line to your local IP:
    ```python
    app.run(host='192.168.0.4', port=8080, debug=False)
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Application

1. **Start the Flask Server**:
    ```bash
    python app.py
    ```

2. **Access the Application**:
    Open your web browser and navigate to `http://<your-ip>:8080`.

### Generating Images

1. **Choose a Model**:
    Select a model from the dropdown list or add a new model URL.

2. **Set Parameters**:
    - **Model Type**: Choose the type of model (e.g., SDXL, SD1.5, FLUX).
    - **Scheduler**: Select the scheduler model.
    - **Prompt**: Enter the main prompt that describes the image you want to generate.
    - **Negative Prompt**: Enter negative prompts to specify what you want to avoid in the generated image.
    - **Width and Height**: Set the dimensions of the generated image.
    - **CFG Scale**: Adjust the scale value to control how closely the AI follows your prompts.
    - **Sampling Steps**: Set the number of sampling steps.
    - **Number of Images**: Specify the number of images to generate.
    - **Custom Seed**: Enter a custom seed for reproducibility.

3. **Generate Images**:
    Click the "Generate Images" button to start the image generation process.

### Additional Features

- **Stop Generation**: Click the "Stop Generation" button to stop the image generation process.
- **Clear Images**: Click the "Clear Images" button to clear all generated images.
- **Preview ControlNet**: Click the "Preview ControlNet" button to preview the ControlNet.
- **Get Metadata**: Click the "Get Metadata" button to extract metadata from an uploaded image.
- **Model Editor**: Click the "Model Editor" button to view and edit the JSON configuration of models.

## License

This project is licensed under the MIT License. See the [LICENSE.txt](http://_vscodecontentref_/1) file for details.

---

Thanks to [@John6666cat](https://github.com/John6666cat) for the models at [huggingface](https://huggingface.co/John6666).