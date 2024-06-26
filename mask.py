from sessions.silueta import SiluetaSession
from PIL import Image
from PIL.Image import Image as PILImage

def generate_mask(image_path: str) -> list[PILImage]:
    img = Image.open(image_path)
    predictor = SiluetaSession('silueta', None)
    masks = predictor.predict(img)
    return masks

i = 1
for mask in generate_mask('images/cat.jpg'):
    mask.save('mask' + str(i) + '.png', 'png')
    i += 1