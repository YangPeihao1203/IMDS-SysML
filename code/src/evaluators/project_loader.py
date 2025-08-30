# 将llm生成的sysml推荐结果 拼接为完整的magic项目，这里处理两种，一种是json，一种是xmi格式
import os
import json
from code.src.database.xmi_parser.base.namespace import NS
import xml.etree.ElementTree as ET  # 导入解析XML所需的包
from enum import Enum
# from code.src.evaluators.syntax_evaluator.syntax_validator import XMISyntaxValidator
import re

class ContentType(Enum):
    JSON = "json"
    XMI = "xmi"

    def __str__(self):
        return self.value

class ProjectLoader:
    def __init__(self, project_path: str, content_type: ContentType = ContentType.XMI):
        self.project_path = project_path  # 项目文件地址，目前只接受 导出的xmi文件的地址
        self.project_data = None
        self.content_type = content_type  # 项目内容类型，默认为XMI
        
    def wrap_project(self,new_content):
        '''
        new_content: str, 新的内容，可能是json或者xmi格式。把llm生成的sysml推荐结果拼接为完整的sysml项目
        '''
        pass
    
    def load_local_project(self):
        """
        加载XMI项目文件
        """
        if not os.path.exists(self.project_path):
            raise FileNotFoundError(f"Project file not found: {self.project_path}")
        with open(self.project_path, 'r', encoding='utf-8') as file:
           self.project_data = file.read()
    
    def save_project(self, output_path: str):
        """
        保存项目到指定路径
        """
        if not self.project_data:
            raise ValueError("No project data to save.")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(self.project_data)
        print(f"Project saved to {output_path}")
        
        
    
 
class MagicDrawProjectLoader(ProjectLoader):
    """
    MagicDraw项目加载器，专门处理MagicDraw导出的XMI文件
    """

    def __init__(self, project_path: str,use_llm_structure: bool = False, llm=None,max_node_num=1000, NS: dict = NS):
        super().__init__(project_path)
        self.namespace = NS  # MagicDraw的命名空间
        self.project_xmi = None # 存储加载的XMI内容，是XML的模型树。最后填充完之后，文本更为project_data属性
        self.use_llm_structure = use_llm_structure  # 是否使用LLM决定要插入的元素在哪个位置目录
        self.llm=llm # LLM模型，用于辅助结构决策
        self.prompt="""
        You are a system modeling assistant helping with SysML XMI model construction. Based on the provided model structure, current user modeling focus, and a new model fragment, your task is to decide the most appropriate parent element (by ID) under which the new fragment should be inserted.
        You will be given:
        - {model_structure}: A hierarchical summary of the current SysML model, represented as a tree. Each node includes its `xmi:id`, element type, name, and any relevant attributes.
        - {current_position}: The `xmi:id` of the current element the user is working on, representing the active context.
        - {incremental_elem}: A new XML fragment (SysML-compliant) that needs to be inserted into the model.

        Your goal is to identify the `xmi:id` of the parent element in the existing model under which the new element should be inserted. Focus on structural and semantic consistency with the model hierarchy and user’s current context.

        Only output the `xmi:id` of the recommended parent element. Do not return any explanation or other content.

        Example output:
        `_18_0_1_ae502ce_1421696920523_400465_47687`
        """
        self.max_node_num=1000
        # 判断，如果use_llm_structure为True，则需要传入llm模型
        if self.use_llm_structure and not self.llm:
            raise ValueError("use_llm_structure is True, but no LLM model provided.")
        self.id_element_map = {}  # 用于存储ID到XML元素的映射
        self.load_project_xmi()
        
        

    def get_attr(self,elem,prefix_colon_key):
        if ":" not in prefix_colon_key:
            return elem.attrib.get(prefix_colon_key)
        prefix,key=prefix_colon_key.split(":")
        namespace=self.namespace.get(prefix)
        if namespace:
            return elem.attrib.get(f"{{{namespace}}}{key}")
        else:
            return elem.attrib.get(prefix_colon_key)
        
    def init_id_element_map(self):
        """
        初始化ID到XML元素的映射
        """
        if self.project_xmi is None:
            raise ValueError("Project XMI is not loaded.")
        for elem in self.project_xmi.iter():
            elem_id = self.get_attr(elem, 'xmi:id')
            if elem_id:
                self.id_element_map[elem_id] = elem

    def to_string(self, element, pretty=False, xml_declaration=True):
        """
        将XML Element转换为字符串格式

        参数:
        - element: 要序列化的XML元素
        - pretty: 是否美化缩进（默认False）
        - xml_declaration: 是否添加XML声明头（默认True）

        返回:
        - XML字符串
        """
        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        # 注册命名空间
        for prefix, uri in NS.items():
            ET.register_namespace(prefix, uri)

        # 直接序列化为字节串
        rough_bytes = ET.tostring(
            element,
            encoding='utf-8',
            method='xml',
            xml_declaration=xml_declaration
        )

        # 不需要美化，直接返回字符串
        if not pretty:
            return rough_bytes.decode('utf-8')

        # 美化缩进
        reparsed = minidom.parseString(rough_bytes)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # 如果不需要XML声明头，就移除第一行
        if not xml_declaration:
            lines = pretty_xml.split('\n')
            if lines[0].strip().startswith('<?xml'):
                lines = lines[1:]
            pretty_xml = '\n'.join(lines)
        else:
            pretty_xml = re.sub(r'^<\?xml[^>]*\?>', '<?xml version="1.0" encoding="UTF-8"?>', pretty_xml)

        # 去除空行
        pretty_xml = "\n".join([line for line in pretty_xml.split("\n") if line.strip() != ""])

        return pretty_xml


    def wrap_parent_content(self,content: str) -> str:
        return f'''
<xmi:XMI xmlns:StandardProfile="http://www.omg.org/spec/UML/20131001/StandardProfile"
         xmlns:uml="http://www.omg.org/spec/UML/20131001"
         xmlns:sysml="http://www.omg.org/spec/SysML/20181001/SysML"
         xmlns:Dependency_Matrix_Profile="http://www.magicdraw.com/schemas/Dependency_Matrix_Profile.xmi"
         xmlns:MD_Customization_for_SysML__additional_stereotypes="http://www.magicdraw.com/spec/Customization/180/SysML"
         xmlns:xmi="http://www.omg.org/spec/XMI/20131001"
         xmlns:MD_Customization_for_Requirements__additional_stereotypes="http://www.magicdraw.com/spec/Customization/180/Requirements"
         xmlns:Custom_Stereotypes="http://www.magicdraw.com/schemas/Custom_Stereotypes.xmi"
         xmlns:Stereotypes="http://www.magicdraw.com/schemas/Stereotypes.xmi">
    {content}
   </xmi:XMI>'''

    def load_xmi_string(self, xmi_string: str):
        try:
            res = ET.fromstring(xmi_string)
            return True, res
        except ET.XMLSyntaxError as e:
            print(f"XMI 解析失败: {e}")
            return False, None
    
    def get_model_and_stereotypes(self, content):
        # xmiSyntaxValidator = XMISyntaxValidator()
        wrapped_content = self.wrap_parent_content(content)
        is_valid, node_xml = self.load_xmi_string(wrapped_content)
        if not is_valid:
            return False,None, None
        model_content_list=[]
        stereo_content_list=[]

        for child in list(node_xml):
            full_tag = child.tag  # e.g., '{http://...}Block'
            if full_tag.startswith("{"):
                uri, localname = full_tag[1:].split("}")
            else:
                uri = None
                localname = full_tag

            if uri not in NS.values():
                print(f"未知的命名空间 URI: {uri}（元素: {localname}）")
                model_content_list.append(child)
            else:
                print(f"已知命名空间元素: {localname}，命名空间: {uri}")
                stereo_content_list.append(child)
        return True, model_content_list, stereo_content_list


    def load_project_xmi(self):
        """
        加载MagicDraw项目的XMI文件,将模型树更新到project_xmi属性中
        """
        self.load_local_project()
        try:
            self.project_xmi = ET.fromstring(self.project_data)
            self.init_id_element_map()  # 初始化ID到XML元素的映射
            
        except ET.ParseError as e:
            raise ValueError(f"加载XMI失败，解析错误: {e}")
    


        

    def _add_incremental_element(self,incremental_element,old_element):
        """
        incremental_element: 新增的模型元素
        old_element: 已存在的模型元素
        二者的id一样，本方法增量更新incremental_element的属性和子元素到old_element中
        如果新增的内容，old_element中不存在，则直接添加到old_element中；如果存在，则递归地调用本方法，处理同样的属性和子元素
        """
        # 1. 合并属性
        for key, value in incremental_element.attrib.items():
            if key not in old_element.attrib:
                old_element.set(key, value)
            # 若已存在，保留 old_element 中已有的值（可选策略）

        # 2. 为 old_element 的子元素构建 ID → element 映射
        old_children_id_map = {}
        for child in old_element:
            child_id = self.get_attr(child, 'xmi:id')
            if child_id:
                old_children_id_map[child_id] = child

        # 3. 遍历 incremental_element 的子元素，递归合并
        for inc_child in incremental_element:
            inc_child_id = self.get_attr(inc_child, 'xmi:id')
            if inc_child_id and inc_child_id in old_children_id_map:
                # 已存在，递归更新
                self._add_incremental_element(inc_child, old_children_id_map[inc_child_id])
            else:
                # 不存在，直接添加
                old_element.append(inc_child)
                self.id_element_map[inc_child_id] = inc_child  # 更新映射


    def add_incremental_elements(self, model_element_list, current_position=None):
        """
        将增量的模型元素添加到MagicDraw项目的XMI project_xmi 根元素中
        current_position: 可选参数，表示用户当前打开的模型的工作层级位置。用来让llm进行辅助
        """

        for incremental_elem in model_element_list:
            # 这里假设incremental_elem是一个Element对象
            # 将其添加到project_xmi的根元素下
            id_incremental = self.get_attr(incremental_elem, 'xmi:id')
            if id_incremental not in self.id_element_map:
                if not self.use_llm_structure: 
                    # 如果ID不存在于id_element_map中，直接添加到project_xmi
                    uml_Model = self.project_xmi.find('.//uml:Model', namespaces=self.namespace)
                    if uml_Model is None:
                        raise ValueError("未找到 uml:Model 元素，请检查XMI文件是否正确。")
                    uml_Model.append(incremental_elem)
                    # 更新id_element_map
                    self.id_element_map[id_incremental] = incremental_elem

            else:
                #如果id存在，则就要开始增量的更新了
                old_element = self.id_element_map[id_incremental]
                self._add_incremental_element(incremental_elem, old_element)




    def wrap_project(self, new_content,output_file=None,current_position=None,save=True):
        """
        将新的内容包装到MagicDraw项目中
        """
        if self.content_type == ContentType.JSON:
            pass 
        elif self.content_type == ContentType.XMI:
            if "<xmi:XMI" in new_content:
                self.project_data=new_content
                if save:
                    output_file = output_file or self.project_path.replace('.xml', "_wrapped.xml")
                    self.save_project(output_file)
                else:
                    return self.project_data
            
            success, model_list, stereo_list = self.get_model_and_stereotypes(new_content) 
            if not success:
                raise ValueError("XMI内容验证失败，无法包装项目。原因是 XML语法未通过验证。")
            
            for stereo in stereo_list:
                self.project_xmi.append(stereo)
            
            self.add_incremental_elements(model_list, current_position)
        self.project_data =self.to_string(self.project_xmi,pretty=True)  
        if save:
            output_file = output_file or self.project_path.replace('.xml', "_wrapped.xml")

            self.save_project(output_file) 
        else:
            print("Project data not saved, but ready for further processing.")
            return self.project_data


