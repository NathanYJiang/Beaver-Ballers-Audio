from facenet_models import FacenetModel
import numpy as np
from PIL import Image, ImageDraw, ImageFont

image_path = '[Insert image path]'
image = Image.open(image_path)
image = image.convert('RGB')
image_array = np.array(image)
print(image_array.shape)
model = FacenetModel()
boxes, probabilities, landmarks = model.detect(image_array)
print(boxes, probabilities)

# For both step 4 and 7
def DrawBoxOnPicture(image, box, text, output_filename = "output_image.jpg"):
    draw = ImageDraw.Draw(image)
    font_size = 20
    font = ImageFont.truetype("Arial Unicode.ttf", font_size)
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
    text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
    draw.rectangle([text_bbox[:2], text_bbox[2:]], fill="red")
    draw.text((box[0], box[1]), text, fill="white", font=font)
    image.show()
    image.save(output_filename)
    
# For step 4
for box, prob in zip(boxes, probabilities):
    DrawBoxOnPicture(image, box, f"{prob:.2f}")

# For step 7
def DrawBoxesOnPicture(image, boxes, labels):
    for box, text in zip(boxes, labels):
        DrawBoxOnPicture(image, box, text)