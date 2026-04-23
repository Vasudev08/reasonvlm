import base64
import io
from PIL import Image, ImageDraw, ImageFont

def apply_scaffold_coordinates(image_input, grid_size=(10, 10), dot_color=(255, 0, 0), text_color=(255, 0, 0), dot_radius=2, font_size=None):
    """
    Applies SCAFFOLD coordinates (dot matrix with labels) to an image.
    
    Args:
        image_input: PIL Image or path to image.
        grid_size: Tuple (cols, rows) for the dot matrix.
        dot_color: Color of the dots.
        text_color: Color of the coordinate labels.
        dot_radius: Radius of the dots.
        font_size: Size of the coordinate text.
        
    Returns:
        PIL Image: Image with coordinate scaffolding.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.copy().convert("RGB")
    else:
        raise ValueError("Unsupported image input type. Provide path string or PIL Image.")

    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    cols, rows = grid_size
    
    # Calculate spacing
    x_spacing = width / (cols + 1)
    y_spacing = height / (rows + 1)
    
    # Try to load a font, fallback to default
    try:
        if font_size is None:
            font_size = max(10, int(min(width, height) / 50))
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            x = int(c * x_spacing)
            y = int(r * y_spacing)
            
            # Draw dot
            draw.ellipse([x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill=dot_color)
            
            # Draw label (x, y)
            # Normalizing coordinates to 0-100 or just using grid indices
            # Paper often uses normalized or simple grid indices. Let's use 0-100 normalized for better spatial grounding.
            norm_x = int((c / (cols + 1)) * 100)
            norm_y = int((r / (rows + 1)) * 100)
            label = f"({norm_x},{norm_y})"
            
            # Offset text slightly from dot
            draw.text((x + 5, y + 5), label, fill=text_color, font=font)
        
    return img

def get_scaffold_prompt(original_prompt):
    """
    Wraps the prompt with SCAFFOLD instructions.
    """
    scaffold_instruction = "The image is overlaid with a coordinate dot matrix. Use these coordinates to ground your visual reasoning. "
    return scaffold_instruction + original_prompt
