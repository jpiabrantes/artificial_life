import numpy as np
from PIL import Image

# dirt = np.array((120., 72, 0)) / 255
# grass = np.array((85., 168, 74)) / 255
#
# norm_sugar = np.linspace(0, 1, 5)[:, None]
# s_rgb = norm_sugar * grass + (1 - norm_sugar) * dirt
# s_rgb = s_rgb.reshape(5, 1, 3)
# img = Image.fromarray((s_rgb*255).astype(np.uint8))
# img = img.resize((100, 5*100))
# img.save('test2.png')

DNA_COLORS = {1: (255, 255, 0), 2: (0, 255, 255), 3: (255, 0, 255), 4: (0, 0, 0), 5: (255, 255, 255)}
for i in range(1, 6):
    img = np.tile(np.array(DNA_COLORS[i]).astype(np.uint8), (50, 50, 1))
    scale = 50
    left, top, width, height = [int(round(f)) for f in ((0 + .05) * scale, (0 + .05) * scale, scale * 0.9, scale * 0.2)]
    img[top: top + height, left: left + width, :] = np.array((0, 0, 255), np.uint8)
    # damage_width = int(round((1 - health / 2) * width))
    #     img[top: top + height, left: left + damage_width, :] = np.array((255, 0, 0), np.uint8)
    img = Image.fromarray(img)
    img.save('%d.png' % i)
