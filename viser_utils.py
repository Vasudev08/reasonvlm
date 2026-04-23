import base64
import io
from PIL import Image, ImageDraw

def apply_viser_scaffolding(image_input, num_lines=3, line_color=(255, 0, 0, 128), line_width=2):
    """
    Applies VISER scaffolding (horizontal lines) to an image.
    
    Args:
        image_input: PIL Image or path to image.
        num_lines: Number of horizontal lines to add.
        line_color: Color of the lines (default is semi-transparent red).
        line_width: Width of the lines.
        
    Returns:
        PIL Image: Image with visual scaffolding.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.copy().convert("RGB")
    else:
        raise ValueError("Unsupported image input type. Provide path string or PIL Image.")

    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Calculate spacing
    # n lines divide the image into n+1 segments
    spacing = height / (num_lines + 1)
    
    for i in range(1, num_lines + 1):
        y = int(i * spacing)
        draw.line([(0, y), (width, y)], fill=line_color, width=line_width)
        
    return img

def get_viser_prompt(original_prompt):
    """
    Wraps the prompt with VISER instructions.
    """
    viser_instruction = "Scan the image sequentially based on horizontal lines exists in the image. "
    return viser_instruction + original_prompt

def encode_viser_image(image):
    """
    Encodes a PIL image to base64 for API requests.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
