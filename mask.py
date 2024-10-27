from sessions.silueta import SiluetaSession
from PIL import Image
from PIL.Image import Image as PILImage
from functools import lru_cache
@lru_cache(maxsize=128, typed=False)
def generate_mask(image_path: str) -> list[PILImage]:
    img = Image.open(image_path)
    predictor = SiluetaSession('silueta', None)
    masks = predictor.predict(img)
    return masks

if __name__ == "__main__":
    i = 1
    for mask in generate_mask('images/cat.jpg'):
        mask.save('mask' + str(i) + '.png', 'png')
        i += 1    
