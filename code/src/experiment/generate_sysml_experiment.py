"""
Experiment class for managing and running experiments.
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from .experiment_config import ExperimentConfig
"""

import os
from typing import Any, Dict
from datetime import datetime
from uuid import uuid4
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
import os

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'



Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-mpnet-base-v2",
)



class Experiment:
    """
    Class representing an experiment with configuration and results.
    """
    def __init__(self):
        """
        Initializes the Experiment with the given configuration.

        :param config: An instance of ExperimentConfig containing experiment settings.
        """
        self.id = str(uuid4())
        self.start_time = datetime.now()
        self.end_time = None
        self.results=   []  # List to store results of the experiment
        self.llm = None  # Placeholder for LLM instance
        self.dataset_dictory = None  # Placeholder for dataset directory
        self.mode="native"  # Placeholder for experiment mode
        self.max_workers=5  # Number of worker threads for multi-threaded requests
        self.save_path=None  # Placeholder for save path
        self.file_name=None  # Placeholder for file name
        self.rules_dir="dataset/rag_content/sysml-rules"  # Directory containing SysML rules
        self.example_index_dir="dataset/rag_content/index_by_model_type"  # Directory containing SysML few-shot examples
        self.shots_num=1  # Number of few-shot examples to use
        self.is_online_sevice=False  # Whether the service is provided online
        self.model_type2rule_file = {
            "Block":"BDD-Block.txt",
            "ConstraintBlock":"BDD-ConstraintBlock.txt",
            "InterfaceBlock":"BDD-InterfaceBlock.txt",
            "ValueType":"BDD-ValueType.txt",
            "BindingConnector":"IBD-BindingConnector.txt",
            "Connector":"IBD-Connector.txt",
            "InformationFlow":"IBD-InformationFlow.txt",
            "ItemFlow":"IBD-ItemFlow.txt",
            "Activity":"ACT-Activity.txt",
            "TestCase":"ACT-TestCase.txt",
            "caused_by":"PAR-caused_by.txt",
            "Package":"PKG-Package.txt",
            "DeriveReqt":"REQ-DeriveReqt.txt",
            "Requirement":"REQ-Requirement.txt",
            "Trace":"REQ-Trace.txt",
            "Interaction":"SEQ-Interaction.txt",
            "StateMachine":"STM-StateMachine.txt",
            "Stakeholder":"UCD-Stakeholder.txt",
            "UseCase":"UCD-UseCase.txt",
        }
        self.model_type="Block"

        self.native_prompt = """
You are a SysML expert. Your task is to generate a SysML model in XMI format 
based on the given modeling requirement and the current context.

Definitions:
- requirement: the modeling requirement, i.e., what the model should capture or achieve.
- context: the modeling context, i.e., existing model elements, constraints, or related information that can be referenced.
- target model: the SysML model to generate; it can reference elements present in the context.

Input:
Requirement: {requirement}
Context: {related_elements}

Instructions:
- Generate only valid SysML XMI representing the target model.
- Do NOT include any explanations, comments, or other text outside the XMI.
- Do NOT wrap your response with ```xml or other code blocks.

Output:
- The complete SysML XMI representing the target model.
"""

        self.rule_only_prompt = """
You are a SysML expert. Your task is to generate a SysML model in XMI format 
based on the given modeling requirement and the current context.

Definitions:
- requirement: the modeling requirement, i.e., what the model should capture or achieve.
- context: the modeling context, i.e., existing model elements, constraints, or related information that can be referenced.
- retrive_rule: the SysML rule knowledge to use for generating the model.
- target model: the SysML model to generate; it can reference elements present in the context.

Input:
Requirement: {requirement}
Context: {related_elements}

sysml rule knowledge: {retrive_rule}

Instructions:
- Generate only valid SysML XMI representing the target model.
- Do NOT include any explanations, comments, or other text outside the XMI.
- Do NOT wrap your response with ```xml or other code blocks.

Output:
- The complete SysML XMI representing the target model.
"""
        self.example_shots_prompt = """
You are a SysML expert. Your task is to generate a SysML model in XMI format
based on the given modeling requirement and the current context.
Definitions:
- requirement: the modeling requirement, i.e., what the model should capture or achieve.
- context: the modeling context, i.e., existing model elements, constraints, or related information that can be referenced.
- few_shot_examples: the few shot examples to use for generating the model.
- target model: the SysML model to generate; it can reference elements present in the context.

Input:
Requirement: {requirement}
Context: {related_elements}
few shot examples: {few_shot_examples}

===================================================
Instructions:
- Generate only valid SysML XMI representing the target model.
- Do NOT include any explanations, comments, or other text outside the XMI.
- Do NOT wrap your response with ```xml or other code blocks.

Output:
- The complete SysML XMI representing the target model.
    """
        

        self.rule_and_example_shots_prompt = """
You are a SysML expert. Your task is to generate a SysML model in XMI format
based on the given modeling requirement and the current context.
Definitions:
- requirement: the modeling requirement, i.e., what the model should capture or achieve.
- context: the modeling context, i.e., existing model elements, constraints, or related information that can be referenced.
- retrive_rule: the SysML rule knowledge to use for generating the model.
- few_shot_examples: the few shot examples to use for generating the model.
- target model: the SysML model to generate; it can reference elements present in the context.

Input:
Requirement: {requirement}
Context: {related_elements}
sysml rule knowledge: {retrive_rule}
few shot examples: {few_shot_examples}

===================================================
Instructions:
- Generate only valid SysML XMI representing the target model.
- Do NOT include any explanations, comments, or other text outside the XMI.
- Do NOT wrap your response with ```xml or other code blocks.

Output:
- The complete SysML XMI representing the target model.
        """



    def _format_few_shot_examples(self,examples: list) -> str:
        """
        Format few-shot examples with numbering.
        :param examples: list of examples, each element is a string (or dict you can stringify)
        :return: formatted string with numbered examples
        """
        formatted = []
        for i, ex in enumerate(examples, start=1):
            formatted.append(f"Example {i}:\n{ex}")
        return "\n\n".join(formatted)


    def init_rag_index_by_model_type(self, dataset_dir):
        """
        Initializes the RAG index by model type from the specified directory.
        Loads few-shot examples and saves them per model type, and builds a RAG index
        for each model type in a separate folder under self.example_index_dir.
        """
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory {dataset_dir} does not exist.")

        if not os.path.exists(self.example_index_dir):
            os.makedirs(self.example_index_dir, exist_ok=True)

        # Step 1: 读取 CSV 并组织成 record[project][diagram_type][model_type] 的字典
        record = {}
        for project_dir in os.listdir(dataset_dir):
            project_path = os.path.join(dataset_dir, project_dir)
            if not os.path.isdir(project_path):
                continue
            record[project_dir] = {}
            for diagram_type_dir in os.listdir(project_path):
                diagram_type_path = os.path.join(project_path, diagram_type_dir)
                if not os.path.isdir(diagram_type_path):
                    continue
                record[project_dir][diagram_type_dir] = {}
                for file_name in os.listdir(diagram_type_path):
                    if file_name.endswith('.csv'):
                        file_cleaned = file_name.replace('_dataset.xmi.csv', '')
                        file_path = os.path.join(diagram_type_path, file_name)
                        df = pd.read_csv(file_path)
                        record[project_dir][diagram_type_dir][file_cleaned] = df.to_dict(orient='records')

        # Step 2: 针对每个 model_type 保存 CSV 并构建 RAG 索引
        for model_type in self.model_type2rule_file.keys():
            print(f"Processing model type: {model_type}")
            model_type_record = []

            for project, diagram_types in record.items():
                for diagram_type, types in diagram_types.items():
                    record_for_model_type = types.get(model_type, [])
                    model_type_record.extend(record_for_model_type)
            model_type_dir = os.path.join(self.example_index_dir, model_type)
            os.makedirs(model_type_dir, exist_ok=True)

            if not model_type_record:
                print(f"No records found for model_type {model_type}")
                fewshot_docs = [Document(text="There is no example for "+model_type, metadata={"id": "null id", "Element-Type": model_type})]
            else:

                # 保存 CSV 文件
                
                
                model_type_file_path = os.path.join(model_type_dir, f"{model_type}_fewshot_examples.csv")
                df = pd.DataFrame(model_type_record)
                df.to_csv(model_type_file_path, index=False)
                print(f"Saved {len(model_type_record)} records for model type {model_type} to {model_type_file_path}")

                # 读回检查
                df_loaded = pd.read_csv(model_type_file_path)
                loaded_count = len(df_loaded)
                if loaded_count == len(model_type_record):
                    print(f"✔ Read back check passed: {loaded_count} records match saved records.")
                else:
                    print(f"❌ Read back check FAILED: saved {len(model_type_record)}, loaded {loaded_count}")

                # Step 3: 构建 RAG 索引
                fewshot_docs = []
                for row in model_type_record:
                    text = f"requirement: {row['requirement']}\n context: {row['related_elements']}\n target model: {row['focus_element']}"
                    fewshot_docs.append(Document(text=text, metadata={"id": row["id"], "Element-Type": model_type}))

            persist_path = model_type_dir  # 每个 model_type 的索引独立存储在自己的文件夹
            try:
                storage_context = StorageContext.from_defaults(persist_dir=persist_path)
                fewshot_index = load_index_from_storage(storage_context)
                print(f"Loaded existing index for model_type {model_type} at {persist_path}")
            except:
                splitter = SentenceSplitter(chunk_size=8192, chunk_overlap=0)
                fewshot_index = VectorStoreIndex.from_documents(
                    fewshot_docs,
                    transformations=[splitter]
                )
                fewshot_index.storage_context.persist(persist_dir=persist_path)
                print(f"Built and persisted RAG index for model_type {model_type} with {len(fewshot_docs)} documents.")



    def run_experiment_from_dictory(self):
        """
        Runs the experiment using the dataset from the specified directory.
        """
        if not self.dataset_dictory:
            raise ValueError("Dataset directory is not set.")
        
        file_names=os.listdir(self.dataset_dictory)

        
        for filename in file_names:
            self.save_path=os.path.join(self.dataset_dictory, f"experiment_results_{filename}_{self.id}.csv")
            self.file_name=filename
            if "experiment_results" in filename:
                print(f"Skipping file {filename} as it is an experiment result file.")
                continue
            if self.save_path and os.path.exists(self.save_path):
                print(f"Experiment result file {self.save_path} already exists. Skipping this file.")
                continue

            dataset = []

            if filename.endswith('.csv'):
                file_path = os.path.join(self.dataset_dictory, filename)
                print(f"Loading dataset from {file_path}")
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    dataset.append(row.to_dict())
                if dataset is None or len(dataset) == 0:
                    print(f"No data found in {file_path}. Skipping this file.")
                    continue
                else:
                    print(f"Loaded {len(dataset)} records from {file_path}.")
                    self.run(dataset)
                    self.save(self.save_path)

        




    def run(self, dataset=None):
        """
        Runs the experiment with multi-threaded requests to the LLM.

        :param dataset: List of dicts, each containing requirement and related_elements
        :param max_workers: Number of worker threads to use
        """
        print(f"Running experiment {self.id} in mode {self.mode}...")
        if self.mode == "native":
            if not self.llm:
                raise ValueError("LLM instance is not set.")
            if not dataset:
                raise ValueError("Dataset is not provided for native mode.")

            def process_data(data):
                print(f"Processing data with id {data.get('id', 'unknown')} in thread...")
                result = dict(data)
                start_time = datetime.now()
                sysml_xmi = self.get_sysml(data)
                end_time = datetime.now()
                result['sysml_xmi_native'] = sysml_xmi
                result['time_cost_native'] = (end_time - start_time).total_seconds()
                return result

            results = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_data = {executor.submit(process_data, data): data for data in dataset}
                for future in as_completed(future_to_data):
                    try:
                        result = future.result()
                        self.results.append(result)

                        
                        if not self.is_online_sevice:
                            # 👉 每 10 个结果就写一次
                            if len(self.results) >= 10:
                                if self.save_path:
                                    self.save(self.save_path)
                                    print(f"已保存 10 条结果到 {self.save_path}")
                    except Exception as e:
                        print(f"线程处理出错: {e}")

            self.results.extend(results)



        elif self.mode == "rule-only":
            # Implement logic for other modes
            if not self.llm:
                raise ValueError("LLM instance is not set.")
            if not dataset:
                raise ValueError("Dataset is not provided for native mode.")

            def process_data(data):
                print(f"Processing data with id {data.get('id', 'unknown')} in thread...")
                result = dict(data)
                
                model_type="_".join(self.file_name.split("_")[:-1])
                rule_file = self.model_type2rule_file.get(model_type, None)
                rule_file_path = os.path.join(self.rules_dir, rule_file) if rule_file else None
                if not rule_file_path or not os.path.exists(rule_file_path):
                    raise ValueError(f"Rule file {rule_file_path} does not exist for model type {model_type}.")
                with open(rule_file_path, 'r', encoding='utf-8') as f:
                    retrive_rule = f.read()

                data['retrive_rule'] = retrive_rule
                start_time = datetime.now()
                sysml_xmi = self.get_sysml(data)
                end_time = datetime.now()
                result['sysml_xmi_rule_only'] = sysml_xmi
                result['time_cost_rule_only'] = (end_time - start_time).total_seconds()
                return result

            results = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_data = {executor.submit(process_data, data): data for data in dataset}
                for future in as_completed(future_to_data):
                    try:
                        result = future.result()
                        self.results.append(result)

                        # 👉 每 10 个结果就写一次
                        if len(self.results) >= 10:
                            if self.save_path:
                                self.save(self.save_path)
                                print(f"已保存 10 条结果到 {self.save_path}")
                    except Exception as e:
                        print(f"线程处理出错: {e}")

            self.results.extend(results)
        elif self.mode == "example-shots":
            if not self.llm:
                raise ValueError("LLM instance is not set.")
            if not dataset:
                raise ValueError("Dataset is not provided for example-shots mode.")

            # Load few-shot examples index
            model_type="_".join(self.file_name.split("_")[:-1])
            index_path = os.path.join(self.example_index_dir, model_type)
            if not os.path.exists(index_path):
                raise ValueError(f"Few-shot examples index path {index_path} does not exist for model type {model_type}. Please initialize the RAG index first.")

            try:
                storage_context = StorageContext.from_defaults(persist_dir=index_path)
                fewshot_index = load_index_from_storage(storage_context)
                retriver = fewshot_index.as_retriever(similarity_top_k=self.shots_num)
                print(f"Loaded few-shot examples index for model_type {model_type} from {index_path}")
            except Exception as e:
                raise ValueError(f"Failed to load few-shot examples index from {index_path}: {e}")

            def process_data(data):
                print(f"Processing data with id {data.get('id', 'unknown')} in thread...")
                result = dict(data)

                nodes = retriver.retrieve(data.get('requirement', ''))
                few_shot_examples = [n.node.get_content() for n in nodes]
                if not few_shot_examples:
                    raise ValueError(f"No few-shot examples found for requirement: {data.get('requirement', '')}")
                few_shot_examples_str = self._format_few_shot_examples(few_shot_examples)
                data['few_shot_examples'] = few_shot_examples_str
                start_time = datetime.now()
                sysml_xmi = self.get_sysml(data)
                end_time = datetime.now()
                result['retrived_shots_' + str(self.shots_num)] = few_shot_examples_str
                result['sysml_xmi_shots_'+ str(self.shots_num)] = sysml_xmi
                result['time_cost_shots_'+ str(self.shots_num)] = (end_time - start_time).total_seconds()
                return result
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_data = {executor.submit(process_data, data): data for data in dataset}
                for future in as_completed(future_to_data):
                    try:
                        result = future.result()
                        self.results.append(result)

                        # 👉 每 10 个结果就写一次
                        if len(self.results) >= 10:
                            if self.save_path:
                                self.save(self.save_path)
                                print(f"已保存 10 条结果到 {self.save_path}")
                    except Exception as e:
                        print(f"线程处理出错: {e}")
            self.results.extend(results)
        
        elif self.mode == "rule-and-shots":
            if not self.llm:
                raise ValueError("LLM instance is not set.")
            if not dataset:
                raise ValueError("Dataset is not provided for rule-and-example-shots mode.")

            # Load few-shot examples index
            if not self.is_online_sevice:
                model_type="_".join(self.file_name.split("_")[:-1])
            else:
                model_type=self.model_type
            index_path = os.path.join(self.example_index_dir, model_type)
            if not os.path.exists(index_path):
                raise ValueError(f"Few-shot examples index path {index_path} does not exist for model type {model_type}. Please initialize the RAG index first.")

            try:
                storage_context = StorageContext.from_defaults(persist_dir=index_path)
                fewshot_index = load_index_from_storage(storage_context)
                retriver = fewshot_index.as_retriever(similarity_top_k=self.shots_num)
                print(f"Loaded few-shot examples index for model_type {model_type} from {index_path}")
            except Exception as e:
                raise ValueError(f"Failed to load few-shot examples index from {index_path}: {e}")

            def process_data(data):
                print(f"Processing data with id {data.get('id', 'unknown')} in thread...")
                result = dict(data)

                nodes = retriver.retrieve(data.get('requirement', ''))
                few_shot_examples = [n.node.get_content() for n in nodes]
                if not few_shot_examples:
                    raise ValueError(f"No few-shot examples found for requirement: {data.get('requirement', '')}")
                few_shot_examples_str = self._format_few_shot_examples(few_shot_examples)
                
                if not self.is_online_sevice:
                    model_type="_".join(self.file_name.split("_")[:-1])
                else:  
                    model_type=self.model_type
                    
                rule_file = self.model_type2rule_file.get(model_type, None)
                rule_file_path = os.path.join(self.rules_dir, rule_file) if rule_file else None
                if not rule_file_path or not os.path.exists(rule_file_path):
                    raise ValueError(f"Rule file {rule_file_path} does not exist for model type {model_type}.")
                with open(rule_file_path, 'r', encoding='utf-8') as f:
                    retrive_rule = f.read()

                data["retrive_rule"] = retrive_rule
                data['few_shot_examples'] = few_shot_examples_str
                #data['few_shot_examples_for_rule_and_shots'+self.shots_num] = few_shot_examples_str
                start_time = datetime.now()
                sysml_xmi = self.get_sysml(data)
                end_time = datetime.now()
                result['retrived_shots_for_rule_and_shots_'+str(self.shots_num)] = few_shot_examples_str
                result['sysml_xmi_for_rule_and_shots_'+ str(self.shots_num)] = sysml_xmi 
                result['time_cost_for_rule_and_shots_'+ str(self.shots_num)] = (end_time - start_time).total_seconds()
                return result
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_data = {executor.submit(process_data, data): data for data in dataset}
                for future in as_completed(future_to_data):
                    try:
                        result = future.result()
                        self.results.append(result)

                        # 👉 每 10 个结果就写一次
                        if len(self.results) >= 10:
                            if self.save_path:
                                self.save(self.save_path)
                                print(f"已保存 10 条结果到 {self.save_path}")
                    except Exception as e:
                        print(f"线程处理出错: {e}")
            self.results.extend(results)
        

        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Supported modes are 'native', 'rule-only', and 'example-shots'.")


        return self.results
    

        

    def get_sysml(self, data):
        """
        get sysml generated from llm
        the input data is a dict with keys:
        - 'requirement': The requirement text.
        - 'related_elements': The related elements text.
        """
        print("Calling LLM...")
        if not self.llm:
            raise ValueError("LLM instance is not set.")
        
        #print(f"Processing data: {data}")
        if self.mode == "native":
            # if not have requirement, raise ValueError
            if 'requirement' not in data or not data['requirement']:
                raise ValueError("Requirement is required for native mode.")
            
            # if not have related_elements, raise ValueError
            if 'related_elements' not in data or not data['related_elements']:
                raise ValueError("Related elements are required for native mode.")
            
            promt = self.native_prompt.format(
                requirement=data.get('requirement', ''),
                related_elements=data.get('related_elements', '')
            )
        elif self.mode == "rule-only":
            if 'requirement' not in data or not data['requirement']:
                raise ValueError("Requirement is required for rule-only mode.")
            if 'related_elements' not in data or not data['related_elements']:
                raise ValueError("Related elements are required for rule-only mode.")
            if 'retrive_rule' not in data or not data['retrive_rule']:
                raise ValueError("Retrive rule is required for rule-only mode.")
            promt = self.rule_only_prompt.format(
                requirement=data.get('requirement', ''),
                related_elements=data.get('related_elements', ''),
                retrive_rule=data.get('retrive_rule', '')
            )
        elif self.mode == "example-shots":
            if 'requirement' not in data or not data['requirement']:
                raise ValueError("Requirement is required for example-shots mode.")
            if 'related_elements' not in data or not data['related_elements']:
                raise ValueError("Related elements are required for example-shots mode.")
            if 'few_shot_examples' not in data or not data['few_shot_examples']:
                raise ValueError("Few-shot examples are required for example-shots mode.")
            promt = self.example_shots_prompt.format(
                requirement=data.get('requirement', ''),
                related_elements=data.get('related_elements', ''),
                few_shot_examples=data.get('few_shot_examples', '')
            )
        elif self.mode == "rule-and-shots":
            if 'requirement' not in data or not data['requirement']:
                raise ValueError("Requirement is required for rule-and-example-shots mode.")
            if 'related_elements' not in data or not data['related_elements']:
                raise ValueError("Related elements are required for rule-and-example-shots mode.")
            if 'retrive_rule' not in data or not data['retrive_rule']:
                raise ValueError("Retrive rule is required for rule-and-example-shots mode.")
            # print("rule is")
            # print(data.get("retrive_rule"))
            if 'few_shot_examples' not in data or not data['few_shot_examples']:
                raise ValueError("Few-shot examples are required for rule-and-example-shots mode.")
            promt = self.rule_and_example_shots_prompt.format(
                requirement=data.get('requirement', ''),
                related_elements=data.get('related_elements', ''),
                retrive_rule=data.get('retrive_rule', ''),
                few_shot_examples=data.get('few_shot_examples', '')
            )


        id= data.get('id', 'unknown')
        print(f"Processing data with id {id}... ", end="")
        res=self.llm.complete(promt)

        return res
    

    def save(self, path: str):
        if not self.results:
            return

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        file_exists = os.path.isfile(path)
        headers = self.results[0].keys() if self.results else []

        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.results)

        # 👉 写入后清空
        self.results = []





        