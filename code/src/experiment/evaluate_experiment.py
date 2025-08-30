
import os
import csv
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from ragas.dataset_schema import SingleTurnSample
from code.src.evaluators.ragas.config import EvaluationConfig
from code.src.evaluators.ragas.types import MetricType
from code.src.evaluators.ragas.evaluator import evaluate_sysml_generation
from ragas import SingleTurnSample, EvaluationDataset

import time
import math

class EvaluateExperiment:
    """
    class that evaluates an experiment
    """

    def __init__(self):
        super().__init__()
        self.results=   [] 
        self.id=None
        self.dataset_dictory = None
        self.save_path = None
        self.file_name = None
        self.column_to_evaluate_list = (
            "sysml_xmi_native",
            "sysml_xmi_rule_only",
            "sysml_xmi_shots_1",
            "sysml_xmi_shots_3",
            "sysml_xmi_for_rule_and_shots_1",
            "sysml_xmi_for_rule_and_shots_3",
        )
        self.mode="sematic-evaluate"  # default mode
        self.flash_fre=2
        self.evaluator_llm = None
        self.config=None
        self.max_retries=5
        self.wait_time=5
        self.syntaxValidator=None

    def run_experiment_from_dictory(self):
        """
        Runs the experiment using the dataset from the specified directory.
        """
        if not self.dataset_dictory:
            raise ValueError("Dataset directory is not set.")
        
        file_names=os.listdir(self.dataset_dictory)

        
        for filename in file_names:
            self.save_path=os.path.join(self.dataset_dictory, f"evaluate_results_{self.id}_{filename}")
            self.file_name=filename
            if "evaluate_results" in filename:
                print(f"Skipping file {filename} as it is an evaluate result file.")
                continue
            if self.save_path and os.path.exists(self.save_path):
                print(f"Evaluate Experiment result file {self.save_path} already exists. Skipping this file.")
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
        print(f"Running evaluate experiment...")
        

        if not dataset:
            raise ValueError("Dataset is not provided for native mode.")
        if self.mode == "sematic-evaluate":
            if not self.evaluator_llm:
                raise ValueError("LLM instance is not set.")
            print("Running in semantic evaluation mode...")
            def process_data(data):
                print(f"Processing data with id {data.get('id', 'unknown')} in thread...")
                result = dict(data)
                for column in self.column_to_evaluate_list:
                    if column in data:
                        print(f"Processing column {column} for semantic evaluation...")
                        data["to_evaluate"] = data[column]
                        score = self.get_evaluate_result(data)
                        result[column + "_score_by_llm"] = score
                    else:
                        raise ValueError(f"Column {column} not found in data.")
                return result
        elif self.mode =="syntactic-evaluate":
            if not self.syntaxValidator:
                raise ValueError("Syntax Validator instance is not set.")
            print("Running in syntactic evaluation mode...")
            def process_data(data):
                print(f"Processing data with id {data.get('id', 'unknown')} in thread...")
                result = dict(data)
                for column in self.column_to_evaluate_list:
                    if column in data:
                        print(f"Processing column {column} for syntactic evaluation...")
                        data["to_evaluate"] = data[column]
                        score = self.get_evaluate_result(data)
                        result[column + "_syntax_pass"] = score
                    else:
                        raise ValueError(f"Column {column} not found in data.")
                return result
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_data = {executor.submit(process_data, data): data for data in dataset}
            for future in as_completed(future_to_data):
                try:
                    result = future.result()
                    self.results.append(result)

                    # 👉 每 self.flush_fre 个结果就写一次
                    if len(self.results) >= self.flash_fre:
                        if self.save_path:
                            self.save(self.save_path)
                            print(f"已保存 {self.flash_fre} 条结果到 {self.save_path}")
                except Exception as e:
                    print(f"线程处理出错: {e}")

        


    def get_evaluate_result(self,data):
        """获取评估结果
        Args:
            data (dict): 包含评估数据的字典
        Returns:
            score (float): 评估得分
        """
        print("Calling LLM to get evaluate result...")

        if self.mode == "sematic-evaluate":
            if not self.evaluator_llm:
                raise ValueError("LLM instance is not set.")
            if not data:
                raise ValueError("Data is not provided for evaluation.")
            # 👉 调用 LLM 获取评估结果
            if "to_evaluate" not in data:
                raise ValueError("Data must contain 'to_evaluate' key for evaluation.")
            if "requirement" not in data:
                raise ValueError("Data must contain 'requirement' key for evaluation.")
            if "related_elements" not in data:
                raise ValueError("Data must contain 'related_elements' key for evaluation.")
            if "focus_element" not in data:
                raise ValueError("Data must contain 'focus_element' key for evaluation.")
            if self.evaluator_llm is None:
                raise ValueError("Evaluator LLM instance is not set.")
            if self.config is None:
                raise ValueError("Evaluation configuration is not set.")
            to_evaluate = data["to_evaluate"]
            requirement = data["requirement"]
            context = data["related_elements"]
            target_model = data.get("focus_element", None)
            attempt = 0
            while attempt < self.max_retries:
                try:
                    sample = SingleTurnSample(
                        user_input=requirement,
                        response=to_evaluate,
                        reference=target_model
                    )
                    dataset = EvaluationDataset(samples=[sample])
                    result = evaluate_sysml_generation(
                        dataset=dataset,
                        llm=self.evaluator_llm,
                        config=self.config
                    )
                    print(f"Evaluation result: {result}")

                    score = result._repr_dict.get("nv_accuracy", None)
                    if score is None or math.isnan(score):
                        raise ValueError("Evaluation result does not contain a valid 'nv_accuracy'.")
                    return score

                except TimeoutError as e:
                    attempt += 1
                    print(f"TimeoutError encountered, retrying {attempt}/{self.max_retries} in {self.wait_time}s...")
                    time.sleep(self.wait_time)
                except Exception as e:
                    attempt += 1
                    print(f"Error encountered: {e}. Retry {attempt}/{self.max_retries} in {self.wait_time}s...")
                    time.sleep(self.wait_time)

            # 如果多次重试仍失败
            print("Failed to get evaluation result after maximum retries.")
            return None


        elif self.mode == "syntactic-evaluate":
            # 返回0或1
            if "to_evaluate" not in data:
                raise ValueError("Data must contain 'to_evaluate' key for evaluation.")
            if self.syntaxValidator is None:
                raise ValueError("Syntax Validator instance is not set.")
            to_evaluate = data["to_evaluate"]
            try:
                is_valid= self.syntaxValidator.validate(to_evaluate,use_magic_valid=True)
            except Exception as e:
                print(f"Syntax validation error: {e}")
                is_valid = False
                
            if is_valid:
                print(f"Syntax validation passed for data with id {data.get('id', 'unknown')}.")
                return 1
            else:
                print(f"Syntax validation failed for data with id {data.get('id', 'unknown')}.")
                return 0
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")








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
    