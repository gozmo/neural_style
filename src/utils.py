from skopt.space import Categorical
from skopt.space import Integer 
from skopt.space import Real
def cut_image(image):
    w, h = image.size
    new_size = None
    s = None
    if w < h:
        s = w
    else:
        s = h
    return image.crop((0,0,s,s))

def resize_image(image, new_w, new_h):
    w, h = image.size
    
    return image.resize((new_w, new_h))

def parse(elem):
    parsers = [int, float, str]
    for parser in parsers:
        try:
            x = parser(elem)
            return x, parser
        except:
            pass
    return elem

def convert_to_serializable(value):
    if type(value) == np.int64:
        return int(value)
    elif type(value) == np.float64:
        return float(value)
    else:
        return value

def parse_search_string(search_string, parameter_name):
    if "-" in search_string:
        split = search_string.split("-")
        low_str = split[0]
        high_str = split[1]
        low, parser = parse(low_str)
        high, parser = parse(high_str)
        if parser == int:
            return Integer(low, high, name=parameter_name)
        if parser == float:
            return Real(low, high, name=parameter_name)

    elif "," in search_string:
        parameter_space = [parse(elem)[0] for elem in search_string.split(",")]
        return Categorical(parameter_space, name=parameter_name)
    else:
        raise Exception(f"Invalid search string: {search_string}")
