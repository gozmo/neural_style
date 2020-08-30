def cut_image(image):
    w, h = image.size
    new_size = None
    s = None
    if w < h:
        s = w
    else:
        s = h
    return image.crop((0,0,s,s))

def resize_image(image):
    w, h = image.size
    
    new_w = 500
    new_h = 500
    return image.resize((new_w, new_h))

def parse(elem):
    parsers = [int, float, str]
    for parser in parsers:
        try:
            x = parser(elem)
            return x
        except:
            pass
    return elem
