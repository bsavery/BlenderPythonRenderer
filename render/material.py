
def export_material(material):
    if material is None:
        return [0.0, 0.0, 0.0, 0.0]
    else:
        return list(material.diffuse_color)