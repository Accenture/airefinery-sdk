import base64
import os
from io import BytesIO
from typing import Optional
from PIL import Image
import requests

# Constants for processing the images
valid_extensions = {".jpeg": "jpeg", ".jpg": "jpeg", ".png": "png"}
format_map = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG"}


def image_to_base64(image_input: str) -> Optional[str]:
    """
    Convert an image at the given path or URL to a base64-encoded string with MIME type.

    Args:
        image_input: Path to the image file or URL of the image.

    Returns:
        The base64-encoded string of the image content, including the MIME type.
    """
    if not image_input:
        return None

    try:
        # Determine if input is a URL or file path
        if image_input.startswith(("http://", "https://")):
            response = requests.get(image_input)
            if response.status_code != 200:
                raise ValueError(f"Failed to fetch image from URL: {image_input}")

            img = Image.open(BytesIO(response.content))
        else:
            with open(image_input, "rb") as image_file:
                img = Image.open(image_file)
                img.load()  # Fully load the image into memory to close the file

        # Validate the extension
        ext = os.path.splitext(image_input)[1].lower()
        if ext not in valid_extensions:
            raise ValueError(
                f"Invalid image format: {ext}. Supported formats are "
                f"{list(valid_extensions.keys())}"
            )
        # Convert image to base64
        buffer = BytesIO()
        img_format = valid_extensions[ext].upper()
        img.save(buffer, format=img_format)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        mime_type = f"image/{valid_extensions[ext]}"
        return f"data:{mime_type};base64,{img_base64}"

    except Exception as e:
        print(f"Error processing image: {e}")
        raise e


def save_base64_image(base64_string: str, output_path: str) -> bool:
    """
    Decodes a base64 image and saves it to the specified file path, with restrictions on file types.

    Parameters:
        base64_string (str): The base64-encoded string of the image.
        output_path (str): The path where the decoded image should be saved.
                           Only png, jpeg and jpg allowed for now.
                           Jpeg and jpg will both map to jpeg.

    Returns:
        bool: True if the image is saved successfully, False otherwise.
    """

    # Check if the file extension is valid
    ext = output_path.lower().rsplit(".", 1)[-1]
    if f".{ext}" not in valid_extensions:
        print(f"Invalid file extension: {ext}. Only .png, .jpeg, and .jpg are allowed.")
        return False

    try:
        # Decode the base64 string and open it as an image with PIL
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))

        # Ensure the format matches the desired extension
        image_format = format_map[ext]

        # Save the image with the correct format
        image.save(output_path, format=image_format)
        print(f"Image saved to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save image: {e}")
        return False
