import xml.etree.ElementTree as ET


def inkml2tag(inkml_path):
    tree = ET.parse(inkml_path)
    root = tree.getroot()
    prefix = "{http://www.w3.org/2003/InkML}"
    GT_tag = [GT for GT in root.findall(prefix + 'annotation') if GT.attrib == {'type': 'truth'}]
    return GT_tag[0].text


def inkml2ltx(origin_path, dest_path):
    with open(dest_path, mode='a') as dest_file:
        ltx = inkml2tag(origin_path) + '\n'
        dest_file.write(ltx)
