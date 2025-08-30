import jsonschema
import xml.etree.ElementTree as ET
from code.src.database.xmi_parser.base.namespace import NS
from code.src.evaluators.project_loader import ProjectLoader,MagicDrawProjectLoader
import uuid
from code.src.evaluators.util import OSSUtil
import requests


class SyntaxValidator:
    def validate_syntax(self, content) -> bool:
        pass
        # """使用 jsonschema 验证语法，符合返回 True"""
        # try:
        #     jsonschema.validate(instance=json_generated, schema=self.json_schema)
        #     return True
        # except jsonschema.exceptions.ValidationError as e:
        #     print(f"语法验证失败: {e.message}")
        #     return False

class JsonSyntaxValidator(SyntaxValidator):
    def __init__(self):
        self.model_xmi=["packagedElement",]

    def validate(self, content) -> bool:
        """验证 JSON 数据是否符合给定的 JSON Schema"""
        try:
            jsonschema.validate(instance=content, schema=self.json_schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            print(f"JSON Schema 验证失败: {e.message}")
            return False


class XMISyntaxValidator(SyntaxValidator):

    def element2str(self, element):
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        import re

        # 注册命名空间
        for prefix, uri in NS.items():
            ET.register_namespace(prefix, uri)

        # Step 1: 转换为字符串
        rough_string = ET.tostring(element, encoding='utf-8')

        # Step 2: 美化缩进
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # Step 3: 去除 XML 声明
        pretty_xml = re.sub(r'^<\?xml[^>]+\?>\n?', '', pretty_xml)

        # Step 4: 去除 xmlns:* 命名空间声明（无论前面是否带空格）
        pretty_xml = re.sub(r'\s+xmlns(:\w+)?="[^"]+"', '', pretty_xml)

        # Step 5: 去除空行
        pretty_xml = "\n".join([line for line in pretty_xml.split("\n") if line.strip() != ""])

        return pretty_xml

    def wrap_xmi_content_simple(self, model_content: str,stereo_content:str) -> str:
        """将 XMI 内容包装在标准的 XMI 根元素中，后续直接保存，导入给 MagicDraw测试验证"""
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<xmi:XMI xmlns:StandardProfile="http://www.omg.org/spec/UML/20131001/StandardProfile"
         xmlns:uml="http://www.omg.org/spec/UML/20131001"
         xmlns:sysml="http://www.omg.org/spec/SysML/20181001/SysML"
         xmlns:Dependency_Matrix_Profile="http://www.magicdraw.com/schemas/Dependency_Matrix_Profile.xmi"
         xmlns:MD_Customization_for_SysML__additional_stereotypes="http://www.magicdraw.com/spec/Customization/180/SysML"
         xmlns:xmi="http://www.omg.org/spec/XMI/20131001"
         xmlns:MD_Customization_for_Requirements__additional_stereotypes="http://www.magicdraw.com/spec/Customization/180/Requirements"
         xmlns:Custom_Stereotypes="http://www.magicdraw.com/schemas/Custom_Stereotypes.xmi"
         xmlns:Stereotypes="http://www.magicdraw.com/schemas/Stereotypes.xmi">
   <xmi:documentation xmi:type="xmi:Documentation">
      <xmi:exporter>MagicDraw Clean XMI Exporter</xmi:exporter>
      <xmi:exporterVersion>2022x v7</xmi:exporterVersion>
   </xmi:documentation>
   
   <uml:Model xmi:type="uml:Model" xmi:id="eee_1045467100313_135436_1" name="Data">
      <ownedComment xmi:type="uml:Comment" xmi:id="_17_0_5_ae502ce_1403989437566_256279_11604"
                    body="Author:Sanford.&#xA;Created:6/28/14 5:03 PM.&#xA;Title:.&#xA;Comment:.&#xA;">
         <annotatedElement xmi:idref="eee_1045467100313_135436_1"/>
      </ownedComment>


        {model_content} 

      <profileApplication xmi:type="uml:ProfileApplication"
                          xmi:id="_17_0_5_ae502ce_1403989437566_955454_11605">
         <appliedProfile href="http://www.omg.org/spec/SysML/20181001/SysML.xmi#SysML"/>
      </profileApplication>
   </uml:Model>
   
        {stereo_content}
 
</xmi:XMI>'''

    # 语法
    def wrap_parent_content(self,content: str) -> str:
        if "<xmi:XMI" in content:
            # 如果内容已经包含 <xmi:XMI> 标签，则直接返回
            return content
        else:
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
        except ET.ParseError  as e:
            print(f"XMI 解析失败: {e}")
            return False, None

    def get_model_and_stereo_content(self, content):
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
                model_content_list.append(self.element2str(child))
            else:
                print(f"已知命名空间元素: {localname}，命名空间: {uri}")
                stereo_content_list.append(self.element2str(child))
        return True, model_content_list, stereo_content_list

    def validate(self, content,project_path="logs/test_project.xml",use_magic_valid=False) -> bool:
        import re
        if "```xml" in content:
            pattern = re.compile(r"```xml(.*?)```", re.DOTALL)
            match = pattern.search(content)
            if match:
                content = match.group(1).strip()
                print("提取了代码块中的 XML 内容")
                print(content)
        

        is_valid, model_content_list, stereo_content_list = self.get_model_and_stereo_content(content)
        if not is_valid:
            print("XMI 内容验证失败")
            return False
        if not use_magic_valid:
            return True
        # xmi_content = self.wrap_xmi_content_simple(
        #     model_content="\n".join(model_content_list),
        #     stereo_content="\n".join(stereo_content_list)
        # )
#        print(f"XMI 内容:\n{xmi_content}")
        # TODO 接下来做语法的验证，调用magicdraw 提供的基于java的二次开发验证工具 
        project_loader = MagicDrawProjectLoader(project_path)
        xmi_content =project_loader.wrap_project(content,save=False)
        #print(f"XMI 内容:\n{xmi_content}")
        filename = f"{uuid.uuid4().hex}.xmi"
        oos_util = OSSUtil()
        result=oos_util.upload_string(filename, xmi_content)
        if not result:
            raise Exception("上传 XMI 文件到 OSS 失败")

        # === 新增：请求验证服务器 ===
        try:
            url = "http://169.254.88.201:8088/load"
            params = {
                "ossfilename": filename,
                "localpath": "I:/magic_dir/"
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                return True
            else:
                print(f"服务器验证失败：{data}")
                return False
        except Exception as e:
            print(f"请求服务器验证失败: {e}")
            return False
        # return True
    



