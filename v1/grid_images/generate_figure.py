from PIL import Image, ImageDraw, ImageFont
import os

images_path = "./images/hybrid-uniform.png"

# --- Inputs ---
image_combinations = {
    "gaussian_single_point": {"combination": "gaussian_single_point",'cell':"654" ,"path": "./images/ga-sp.png"},
    "gaussian_uniform": {"combination": "gaussian_uniform","cell":"746", "path":"./images/ga_uniform.png"},
    "gaussian_blend": {"combination": "gaussian_blend","cell":"758", "path":"./images/ga-blend.png"},
    "gaussian_simulated_binary": {"combination": "gaussian_simulated_binary", "cell":"654","path": "./images/ga-sbx.png"},
    "gaussian_multiparent": {"combination": "gaussian_multiparent","cell":"654", "path": "./images/ga-multi.png"},
    "polynomial_single_point": {"combination": "polynomial_single_point","cell":"727", "path": "./images/poly-sp.png"},
    "polynomial_uniform": {"combination": "polynomial_uniform","cell":"837", "path": "./images/poly-uniform.png"},
    "polynomial_blend": {"combination": "polynomial_blend","cell":"763", "path": "./images/poly-blend.png"},
    "polynomial_simulated_binary": {"combination": "polynomial_simulated_binary", "cell":"723","path": "./images/poly-sbx.png"},
    "polynomial_multiparent": {"combination": "polynomial_multiparent","cell":"723", "path": "./images/poly-multiparent.png"},
    "cauchy_single_point": {"combination": "cauchy_single_point", "cell":"794","path":  "./images/cauchy_sbx.png"},
    "cauchy_uniform": {"combination": "cauchy_uniform","cell":"741", "path": "./images/cauchy_uniform.png"},
    "cauchy_blend": {"combination": "cauchy_blend","cell":"758", "path": "./images/caucy_blend.png"},
    "cauchy_simulated_binary": {"combination": "cauchy_simulated_binary","cell":"794", "path": "./images/cauchy_sbx.png"},
    "cauchy_multiparent": {"combination": "cauchy_multiparent", "cell":"794","path": "./images/cauchy_sbx.png"},
    "adaptive_single_point": {"combination": "adaptive_single_point","cell":"654", "path": "./images/adaptive_multiparent.png"},
    "adaptive_uniform": {"combination": "adaptive_uniform","cell":"744", "path": "./images/adaptive_uniform.png"},
    "adaptive_blend": {"combination": "adaptive_blend", "cell":"758","path": "./images/adaptive_blend.png"},
    "adaptive_simulated_binary": {"combination": "adaptive_simulated_binary","cell":"654", "path": "./images/adaptive_multiparent.png"},
    "adaptive_multiparent": {"combination": "adaptive_multiparent","cell":"654", "path": "./images/adaptive_multiparent.png"},
    "hybrid_single_point": {"combination": "hybrid_single_point","cell":"773", "path": "./images/hybrid-sp.png"},
    "hybrid_uniform": {"combination": "hybrid_uniform","cell":"746", "path":  "./images/hybrid-uniform.png"},
    "hybrid_blend": {"combination": "hybrid_blend","cell":"738", "path":"./images/hybrid-blend.png"},
    "hybrid_simulated_binary": {"combination": "hybrid_simulated_binary","cell":"773", "path": "./images/hybrid-sbx.png"},
    "hybrid_multiparent": {"combination": "hybrid_multiparent","cell":"773", "path": "./images/hybrid-multiparent.png"},
}

mutation_types = ["gaussian", "polynomial", "cauchy", "adaptive", "hybrid"]
crossover_types = ["single_point", "uniform", "blend", "simulated_binary", "multiparent"]

# --- Parameters ---
cell_size = (300, 300)
label_font_size = 20
small_text_font_size = 14  # Smaller font size for text on images
output_path = "operator_heatmap.png"

# --- Load fonts ---
try:
    label_font = ImageFont.truetype("arial.ttf", label_font_size)
    small_font = ImageFont.truetype("arial.ttf", small_text_font_size)
except:
    label_font = ImageFont.load_default()
    small_font = ImageFont.load_default()

# --- Create new image canvas ---
width = cell_size[0] * len(crossover_types) + 100  # Reduced from 200 to 100 for row labels
height = cell_size[1] * len(mutation_types) + 50   # Reduced from 100 to 50 for column labels
canvas = Image.new("RGB", (width, height), color="white")
draw = ImageDraw.Draw(canvas)

# --- Draw Grid Images and Add Text ---
for row, mutation in enumerate(mutation_types):
    for col, crossover in enumerate(crossover_types):
        key = f"{mutation}_{crossover}"
        entry = image_combinations.get(key)
        cell="000"
        if not entry or not os.path.exists(entry["path"]):
            print(entry)
            continue
        if "cell" in entry:
            cell=entry["cell"]
        img = Image.open(entry["path"]).resize(cell_size)
        x = 100 + col * cell_size[0]  # Adjusted x starting point
        y = 50 + row * cell_size[1]   # Adjusted y starting point
        canvas.paste(img, (x, y))

        # Add text at the top-center of the image
        label_text = f"{cell}/1024 cells"
        bbox = small_font.getbbox(label_text)
        text_width = bbox[2] - bbox[0]
        text_x = x + (cell_size[0] - text_width) // 2  # Center horizontally
        text_y = y + 5  # Small offset from the top
        draw.text((text_x, text_y), label_text, fill="black", font=small_font)

# --- Add Row Labels ---
for row, mutation in enumerate(mutation_types):
    y = 50 + row * cell_size[1] + cell_size[1] // 2 - label_font_size // 2  # Adjusted y starting point
    draw.text((10, y), mutation, fill="black", font=label_font)

# --- Add Column Labels ---
for col, crossover in enumerate(crossover_types):
    x = 100 + col * cell_size[0] + cell_size[0] // 2 - label_font_size * len(crossover) // 4  # Adjusted x starting point
    draw.text((x, 10), crossover, fill="black", font=label_font)

# --- Save ---
canvas.save(output_path)
print(f"Grid image saved to {output_path}")