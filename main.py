import torch
from diffusers import DiffusionPipeline


def generate_image_from_prompt(prompt: str, output_path: str = "images/generated_image.png", device: torch.device = None):
    """
    Generates an image from a given text prompt using Stable Diffusion and saves it to the specified path.

    Args:
        prompt (str): The text prompt for the image generation.
        output_path (str): The file path to save the generated image. Default is "generated_image.png".
        device (torch.device): The device to run the pipeline on (e.g., "cuda" or "cpu").
                              If None, defaults to CUDA if available, otherwise CPU.

    Returns:
        str: The file path of the saved image.
    """

    # Load the Stable Diffusion XL base pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    # Move the pipeline to the selected device
    pipeline.to(device)

    # Generate the image from the text prompt
    generated_image = pipeline(prompt=prompt).images[0]

    # Save the generated image to the specified file path
    generated_image.save(output_path)
    print(f"Image successfully saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    # Determine the device (CUDA if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure your system has a compatible GPU or CUDA runtime if required
    # e.g., pip install nvidia-cuda-runtime-cu12

    # Define prompt and output file path
    prompt_text = "twins playing football"
    output_file = "images/twins_playing_football.png"

    # Generate and save the image
    generate_image_from_prompt(prompt_text, output_file, device)

