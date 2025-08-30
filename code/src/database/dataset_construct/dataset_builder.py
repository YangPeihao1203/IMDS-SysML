import json
import string
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.dom.minidom

class DatasetBuilder:
    """
    A class to build a dataset for a specific task.
    """
    def __init__(self,client, input_file="", output_file="",write_frequency=10,dataset_type="JSON"):
        self.client = client
        self.input_file = input_file  # 指直接的json文件路径
        self.output_file = output_file # 指最终的json文件路径
        self.name_map={
            "BDD":"Block Definition Diagram",
            "IBD":"Internal Block Diagram",
            "PAR":"Parametric Diagram",
            "ACT":"Activity Diagram",
            "REQ":"Requirement Diagram",
            "USD":"Use Case Diagram",
            "STM":"State Machine Diagram",
            "SEQ":"Sequence Diagram",
            "PKG":"Package Diagram",
        }
        self.write_frequency = write_frequency# 每10条记录写入一次文件
        self.count = 0  # 记录处理的条目数
        self.dataset = []
        self.dataset_type=dataset_type
        self.init_template()
        

    def init_template(self):
        """
        Initialize the template for building prompts.
        This method can be overridden to set a custom template.
        """
        self.template = None
        if self.dataset_type == "JSON":
            self.template = """
                You are a systems engineer with expertise in requirements analysis and modeling.

                Below is a structured JSON representation of a SysML model element of type "{diagram_type}-{type}", including the main element (focus_elements) and its contextual elements (related_elements):

                focus_elements:
                {focus_element}
                ===============================

                related_elements:
                {related_elements}
                ===============================
                Your task is to write a single, clear, and complete natural language requirement that accurately describes the function, behavior, or purpose of the model element. The output should be suitable for direct inclusion in a requirements specification.

                The requirement must adhere to the following principles:
                - Use a unique, non-overlapping statement.
                - Start with a clear and specific verb.
                - Avoid ambiguous or vague terminology.
                - Provide full context so the requirement is understandable on its own.
                - Ensure traceability between the wording and the SysML model structure.
                - The requirement should be concise and focused, and should be detailed enough.

                Only output the final requirement sentence. Do not include any labels, explanations, formatting, or extra commentary.
            """ 
        elif self.dataset_type == "XMI":
            self.template = """
                You are a systems engineer with expertise in requirements analysis and system modeling.

                Below is a structured XMI snippet exported from MagicDraw, representing a SysML model element of type "{diagram_type}-{type}". SysML elements are implemented as UML base elements, extended with SysML-specific stereotypes.

                The data includes the main model element (focus_elements) and its contextual elements (related_elements), encoded in raw XMI format.

                focus_elements:
                {focus_elements}
                ===============================

                related_elements:
                {related_elements}
                ===============================

                Your task is to write one clear, concise, and complete natural language requirement that accurately describes the structure, function, behavior, or purpose of the focus_element based on its structure and relations.

                The requirement must:
                - Begin with a clear, specific action verb.
                - Be unambiguous, unique, and self-contained.
                - Include enough context to be understood independently.
                - Avoid using modeling-specific terms (e.g., "block", "property").
                - Preserve traceability to the SysML model's structure and meaning.

                Only output the final requirement sentence. Do not include any labels, explanations, or commentary.
            """



    def get_pretty_format(self, data):
        if self.dataset_type == "JSON":
            if isinstance(data, list):
                return "\n".join([self.get_pretty_format(item) for item in data])
            else:
                return json.dumps(data, ensure_ascii=False, indent=2)

        elif self.dataset_type == "XMI":
            def format_xml_string(xml_str):
                try:
                    # 包装一个根节点，防止 parseString 失败（多段 XML 元素时）
                    wrapped = f"<root>{xml_str}</root>"
                    dom = xml.dom.minidom.parseString(wrapped)
                    pretty_xml = dom.toprettyxml(indent="  ")
                    # 移除根节点包装部分
                    pretty_clean = pretty_xml.replace('<?xml version="1.0" ?>\n<root>\n', '').replace('</root>\n', '')
                    return pretty_clean.strip()
                except Exception:
                    return xml_str  # fallback 原样输出

            if isinstance(data, list):
                return "\n".join([format_xml_string(item) for item in data])
            else:
                return format_xml_string(data)

    def set_template(self, template: str):
        self.template = template


    def build_prompt(self, **kwargs):
        """
        Build a prompt from the template and the given keyword arguments.
        Supports missing key handling and variable number of parameters.
        """
        if not self.template:
            raise ValueError("Template is not set. Use set_template() first.")

        # Extract field names from the template
        formatter = string.Formatter()
        required_fields = {field_name for _, field_name, _, _ in formatter.parse(self.template) if field_name}

        # Identify missing fields
        missing = required_fields - kwargs.keys()
        if missing:
            raise KeyError(f"Missing required fields for prompt: {missing}")

        try:
            return self.template.format(**kwargs)
        except Exception as e:
            raise ValueError(f"Error formatting template: {e}")

    def get_requirement_from_llm(self,input_file,type="Block",diagram_type="BDD", max_workers=4):
        with open(input_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        results=[]

        def process_entry(entry):
            focus_element = entry["focus_elements"]
            related_elements = entry.get("related_elements", [])
            element_id = entry["id"]

            try:
                prompt = self.build_prompt(
                    focus_elements=self.get_pretty_format(focus_element),
                    related_elements=self.get_pretty_format(related_elements),
                    type=type,
                    diagram_type=self.name_map.get(diagram_type, diagram_type)
                )
                print(f"Processing Entry ID {element_id}: {prompt}") 
                requirement = self.client.chat(prompt)  # 调用 LLM
                return {
                    "id": element_id,
                    "focus_elements": self.get_pretty_format(focus_element),
                    "related_elements": self.get_pretty_format(related_elements),
                    "type": type,
                    "prompt": prompt,
                    "requirement": requirement
                }
            except Exception as e:
                error_str = str(e)
                if "context length" in error_str or "maximum context length" in error_str:
                    print(f"[Token Limit] Entry ID {element_id} prompt too long, skipping.")
                else:
                    print(f"[Error] LLM 调用失败 Entry ID {element_id}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_entry, entry) for entry in dataset]
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)

        return results


    def process_data(self,record):
        # Preprocess the loaded data
        for entry in record:
            if entry is not None and "requirement" in entry:
                self.dataset.append({
                    "id": entry["id"],
                    "focus_element": entry["focus_elements"],
                    "related_elements": entry["related_elements"],
                    "requirement": entry["requirement"]
                })
                self.count += 1
                if self.count % self.write_frequency == 0:
                    # Save the dataset to the output file every `write_frequency` records
                    self.save_data()
        self.save_data()


    def save_data(self, output_file=None):
        import os
        import csv

        output_file = output_file or self.output_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        file_exists = os.path.isfile(output_file)

        # 打开文件写入数据
        with open(output_file, "a", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "focus_element", "related_elements", "requirement"])

            # 如果文件不存在，先写入标题行
            if not file_exists:
                writer.writeheader()

            for entry in self.dataset:
                writer.writerow({
                    "id": entry["id"],
                    "focus_element": json.dumps(entry["focus_element"], ensure_ascii=False),
                    "related_elements": json.dumps(entry["related_elements"], ensure_ascii=False),
                    "requirement": entry["requirement"]
                })

        self.dataset = []
        self.count = 0

    def build_dataset(self,input_file=None, output_file=None,type="Block",diagram_type="BDD"):
        """
        Construct the dataset by loading, processing, and saving data.
        """
        self.output_file = output_file or self.output_file
        record=self.get_requirement_from_llm(input_file,type,diagram_type)
        self.process_data(record)
        
    

    def build_dataset_for_single_project(self, input_directory, output_directory):
        """
        Construct the dataset for a single project.
        """

        import os
        #循环遍历input_directory下的所有第一级子目录
        for diagram_type_name in tqdm(os.listdir(input_directory)):
            dataset_path = os.path.join(input_directory, diagram_type_name)
            if not os.path.isdir(dataset_path):
                continue
            # 遍历dataset_path下的每个json文件
            for file_name in os.listdir(dataset_path):
                if not file_name.endswith(".json"):
                    continue
                input_file = os.path.join(dataset_path, file_name)
                output_file = os.path.join(output_directory, diagram_type_name, file_name.replace(".json", ".csv"))
                # 如果output_file存在，则跳过
                if os.path.exists(output_file):
                    print(f"Output file {output_file} already exists, skipping.")
                    continue
                element_type_info=file_name.split(".")[0][:-8]  # 只取element_type_info的除去后9个字符的部分
                print(f"Processing {input_file} for {diagram_type_name} with type {element_type_info}")
                self.build_dataset(
                    input_file=input_file,
                    output_file=output_file,
                    type=element_type_info,
                    diagram_type=diagram_type_name
                )

