# README

## Image Generation with Stable Diffusion

This Python script demonstrates how to generate an image from a text prompt using Stable Diffusion. The generated image is saved to a specified file path.

### Requirements

- Python 3.8 or higher
- Compatible GPU with CUDA support (optional but recommended for faster performance)
- Libraries:
  - `torch`
  - `diffusers`
  - `Pillow` (for saving images)

Install the required libraries using:
```bash
pip install torch diffusers pillow
```

### How to Use

1. Clone this repository or download the script.

2. Ensure you have the required libraries installed.

3. Run the script using:
   ```bash
   python script_name.py
   ```

4. Modify the `prompt_text` variable in the script to specify your desired text prompt for image generation.

5. The generated image will be saved in the same directory as the script with the file name specified by `output_file`.

### Key Functions

#### `generate_image_from_prompt(prompt: str, output_path: str, device: torch.device)`
Generates an image from a given text prompt and saves it to the specified file path.

**Arguments:**
- `prompt` (str): The text prompt for image generation.
- `output_path` (str): The path to save the generated image.
- `device` (torch.device): The device to run the pipeline (e.g., `cuda` or `cpu`).

**Returns:**
- `str`: The file path of the saved image.

### Example

```python
prompt_text = "a beautiful sunset over a mountain range"
output_file = "sunset.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate and save the image
generate_image_from_prompt(prompt_text, output_file, device)
```

### Notes

- Ensure your system meets the hardware and software requirements for running Stable Diffusion models.
- If using a GPU, make sure the necessary CUDA runtime is installed. For example:
  ```bash
  pip install nvidia-cuda-runtime-cu12
  ```
- Generated images may vary depending on the model used.

### Model Details

The script uses the `stabilityai/stable-diffusion-xl-base-1.0` model from the Hugging Face `diffusers` library.

### Troubleshooting

- If you encounter memory issues, consider running the script on a system with a GPU and sufficient VRAM.
- For slower systems, use CPU mode by ensuring `torch.device` is set to `cpu`.

### License

This project is licensed under the MIT License. See the LICENSE file for details.
