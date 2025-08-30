

from code.src.database.xmi_parser.base.namespace import NS

def get_elements_size(xml_content):
    """获取 XML 元素的数量
    就是统计拥有xmi:id的xmi标签的数量
    """
    # 通过正则表达式，匹配 xmi:id的数量
    import re
    pattern = r'xmi:id="[^"]+"'
    matches = re.findall(pattern, xml_content)
    return len(matches)
    