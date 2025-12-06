from PIL import Image, ImageDraw
size = 1024
size_small = 856
size_rectangle = 474
width_rectangle = 27
radius = 146
margin = (size - size_small) / 2
margin_rectangle0 = 114
margin_rectangle1 = (size_small - size_rectangle - 2 * margin_rectangle0) / 2

fill = (53, 103, 197, 255)
outline = (255, 255, 255)
# fill = (255, 255, 255, 255)
# outline = (0, 0, 0)
# fill = (32, 32, 32, 255)
# outline = (255, 0, 255)

overlay = Image.new('RGBA', (size, size))
draw = ImageDraw.Draw(overlay, "RGBA")
draw.rounded_rectangle(((margin, margin), (margin + size_small, margin + size_small)), radius=radius, fill=fill)

rectangle_x0 = margin + margin_rectangle0
rectangle_x1 = rectangle_x0 + size_rectangle
rectangle_rect0 = ((rectangle_x0, rectangle_x0), (rectangle_x1, rectangle_x1))

rectangle_x0 = rectangle_x0 + margin_rectangle1
rectangle_x1 = rectangle_x0 + size_rectangle
rectangle_rect1 = ((rectangle_x0, rectangle_x0), (rectangle_x1, rectangle_x1))

rectangle_x0 = rectangle_x0 + margin_rectangle1
rectangle_x1 = rectangle_x0 + size_rectangle
rectangle_rect2 = ((rectangle_x0, rectangle_x0), (rectangle_x1, rectangle_x1))

overlay2 = Image.new('RGBA', (size, size))
draw2 = ImageDraw.Draw(overlay2, "RGBA")
draw2.rectangle(rectangle_rect2, outline=outline + (96,), width=width_rectangle)

overlay1 = Image.new('RGBA', (size, size))
draw1 = ImageDraw.Draw(overlay1, "RGBA")
draw1.rectangle(rectangle_rect1, outline=outline + (160,), width=width_rectangle)

overlay0 = Image.new('RGBA', (size, size))
draw0 = ImageDraw.Draw(overlay0, "RGBA")
draw0.rectangle(rectangle_rect0, outline=outline + (255,), width=width_rectangle)
overlay = Image.alpha_composite(overlay, overlay2)
overlay = Image.alpha_composite(overlay, overlay1)
overlay = Image.alpha_composite(overlay, overlay0)
overlay.show()
overlay.save('1024.png')
overlay512 = overlay.resize((512, 512), resample=Image.BICUBIC)
overlay512.save('512.png')
overlay256 = overlay.resize((256, 256), resample=Image.BICUBIC)
overlay256.save('256.png')
overlay128 = overlay.resize((128, 128), resample=Image.BICUBIC)
overlay128.save('128.png')
overlay64 = overlay.resize((64, 64), resample=Image.BICUBIC)
overlay64.save('64.png')
overlay32 = overlay.resize((32, 32), resample=Image.BICUBIC)
overlay32.save('32.png')
overlay16 = overlay.resize((16, 16), resample=Image.BICUBIC)
overlay16.save('16.png')

