# from code.src.experiment.generate_sysml_experiment import Experiment
# from code.src.llm.llm_factory import NativeLLMFactory
import os

# llm=NativeLLMFactory.create_llm_client("deepseek-chat","DeepSeek")
# exp=Experiment()
# exp.id="native"
# exp.mode="native"
# exp.llm=llm


# exp.dataset_dictory="result/spacecraft/1-part/ACT"

# # parent_dir="result/spacecraft/1-part"
# parent_dir="result/spacecraft/2-part"

# print("start experiment...")
# # 再加一层循环处理
# for subdir in os.listdir(parent_dir):
#     subdir_path = os.path.join(parent_dir, subdir)
#     if os.path.isdir(subdir_path):
#         exp.dataset_dictory = subdir_path
#         print(f"Running experiment for directory: {subdir_path}")
#         exp.run_experiment_from_dictory()

# print("Experiment completed.")



# llm=NativeLLMFactory.create_llm_client("deepseek-chat","DeepSeek")
# exp=Experiment()
# exp.id="rule-only"
# exp.mode="rule-only"
# exp.llm=llm


# # parent_dir="result/spacecraft/1-part"
# parent_dir="result/spacecraft/2-part"

# print("start experiment...")
# # 再加一层循环处理
# for subdir in os.listdir(parent_dir):
#     subdir_path = os.path.join(parent_dir, subdir)
#     if os.path.isdir(subdir_path):
#         exp.dataset_dictory = subdir_path
#         print(f"Running experiment for directory: {subdir_path}")
#         exp.run_experiment_from_dictory()

# print("Experiment completed.")


# dataset_dir="dataset/dataset_native-xmi-csv/train_set"
# exp=Experiment()
# exp.init_rag_index_by_model_type(dataset_dir)
# print("RAG index initialized successfully.")



# llm=NativeLLMFactory.create_llm_client("deepseek-chat","DeepSeek")
# exp=Experiment()
# exp.id="shot_1"
# exp.mode="example-shots"
# exp.shots_num=1
# exp.llm=llm


# # parent_dir="result/spacecraft/1-part"
# parent_dir="result/spacecraft/2-part"

# print("start experiment...")
# # 再加一层循环处理
# for subdir in os.listdir(parent_dir):
#     subdir_path = os.path.join(parent_dir, subdir)
#     if os.path.isdir(subdir_path):
#         exp.dataset_dictory = subdir_path
#         print(f"Running experiment for directory: {subdir_path}")
#         exp.run_experiment_from_dictory()

# print("Experiment completed.")



# llm=NativeLLMFactory.create_llm_client("deepseek-chat","DeepSeek")
# exp=Experiment()
# exp.id="shot_3"
# exp.mode="example-shots"
# exp.shots_num=3
# exp.llm=llm


# parent_dir="result/spacecraft/1-part"
# # parent_dir="result/spacecraft/2-part"

# print("start experiment...")
# # 再加一层循环处理
# for subdir in os.listdir(parent_dir):
#     subdir_path = os.path.join(parent_dir, subdir)
#     if os.path.isdir(subdir_path):
#         exp.dataset_dictory = subdir_path
#         print(f"Running experiment for directory: {subdir_path}")
#         exp.run_experiment_from_dictory()

# print("Experiment completed.")



# llm=NativeLLMFactory.create_llm_client("deepseek-chat","DeepSeek")
# exp=Experiment()
# exp.id="rule-and-shots_1"
# exp.mode="rule-and-shots"
# exp.shots_num=1
# exp.llm=llm


# # parent_dir="result/spacecraft/1-part"
# parent_dir="result/spacecraft/2-part"

# print("start experiment...")
# # 再加一层循环处理
# for subdir in os.listdir(parent_dir):
#     subdir_path = os.path.join(parent_dir, subdir)
#     if os.path.isdir(subdir_path):
#         exp.dataset_dictory = subdir_path
#         print(f"Running experiment for directory: {subdir_path}")
#         exp.run_experiment_from_dictory()

# print("Experiment completed.")



# llm=NativeLLMFactory.create_llm_client("deepseek-chat","DeepSeek")
# exp=Experiment()
# exp.id="rule-and-shots_3"
# exp.mode="rule-and-shots"
# exp.shots_num=3
# exp.llm=llm


# # parent_dir="result/spacecraft/1-part"
# parent_dir="result/spacecraft/2-part"

# print("start experiment...")
# # 再加一层循环处理
# for subdir in os.listdir(parent_dir):
#     subdir_path = os.path.join(parent_dir, subdir)
#     if os.path.isdir(subdir_path):
#         exp.dataset_dictory = subdir_path
#         print(f"Running experiment for directory: {subdir_path}")
#         exp.run_experiment_from_dictory()

# print("Experiment completed.")



# from ragas.dataset_schema import SingleTurnSample
# from code.src.evaluators.ragas.config import EvaluationConfig
# from code.src.evaluators.ragas.types import MetricType
# from code.src.evaluators.ragas.evaluator import evaluate_sysml_generation
# from code.src.llm.llm_factory import LlamaIndexLLMFactory
# from ragas.llms import LlamaIndexLLMWrapper
# from ragas import SingleTurnSample, EvaluationDataset



# sentence_inference = """
# <xmi:XMI xmi:version="2.1" xmlns:xmi="http://schema.omg.org/spec/XMI/2.1" xmlns:uml="http://www.eclipse.org/uml2/5.0.0/UML" xmlns:sysml="http://www.omg.org/spec/SysML/1.4/SysML">
#   <packagedElement xmi:id="_test_case_1" name="Verify Sensor Sensitivity Test Case" xmi:type="uml:Class">
#     <ownedAttribute xmi:id="_base_operation_1" name="base_Operation" xmi:type="uml:Property">
#       <type xmi:idref="_operation_1"/>
#     </ownedAttribute>
#     <extension xmi:id="_stereotype_application_1" xmi:type="uml:Extension">
#       <ownedEnd xmi:id="_extension_end_1" xmi:type="uml:ExtensionEnd" type="_stereotype_1"/>
#     </extension>
#   </packagedElement>
#   <packagedElement xmi:id="_operation_1" name="Verify Sensor Sensitivity Operation" xmi:type="uml:Operation">
#     <method xmi:idref="_18_0_1_ae502ce_1420060459539_606171_49027"/>
#   </packagedElement>
#   <packagedElement xmi:id="_stereotype_1" name="TestCase" xmi:type="uml:Stereotype" baseClass="Class">
#     <ownedAttribute xmi:id="_base_operation_attr" name="base_Operation" xmi:type="uml:Property"/>
#   </packagedElement>
# </xmi:XMI>
# """

# sentence_true="""
# <ownedOperation xmi:type="uml:Operation" xmi:id="_18_0_1_ae502ce_1420131698846_237695_52170" name="verify sensor sensitivity" visibility="public">
#   <method xmi:idref="_18_0_1_ae502ce_1420060459539_606171_49027"/>
# </ownedOperation>
# <sysml:TestCase xmi:id="_18_0_1_ae502ce_1420131827671_77751_52210" base_Operation="_18_0_1_ae502ce_1420131698846_237695_52170"/>

# """


# # # 准备样本数据
# sample = SingleTurnSample(
#     user_input="The system shall verify sensor sensitivity by executing the 'Verify Sensor Sensitivity' test procedure.",
#     response=sentence_inference,
#     reference=sentence_true,
# )
# dataset = EvaluationDataset(samples=[sample])

# # # 构建 LLM Wrapper
# llm = LlamaIndexLLMFactory.create_llm_client("deepseek-chat", "DeepSeek")
# evaluator_llm = LlamaIndexLLMWrapper(llm=llm)

# # # 配置评估指标
# config = EvaluationConfig(
#     metrics=[
#         MetricType.ANSWER_ACCURACY,
#         # MetricType.CONTEXT_RELEVANCE,
#         # MetricType.RESPONSE_GROUNDEDNESS,
#         # MetricType.RUBRIC
#     ]
# )
    

# # # 执行评估
# result = evaluate_sysml_generation(dataset, evaluator_llm, config)
# print(type(result._repr_dict.get("nv_accuracy")))



# from code.src.evaluators.ragas.config import EvaluationConfig

# from code.src.llm.llm_factory import LlamaIndexLLMFactory
# from ragas.llms import LlamaIndexLLMWrapper
# from code.src.evaluators.ragas.types import MetricType

# # # 构建 LLM Wrapper
# # llm = LlamaIndexLLMFactory.create_llm_client("deepseek-chat", "DeepSeek")

# llm= LlamaIndexLLMFactory.create_llm_client("qwen-plus-latest", "BaiLian",is_chat_model=True)
# evaluator_llm = LlamaIndexLLMWrapper(llm=llm)

# # # 配置评估指标
# config = EvaluationConfig(
#     metrics=[
#         MetricType.ANSWER_ACCURACY,
#         # MetricType.CONTEXT_RELEVANCE,
#         # MetricType.RESPONSE_GROUNDEDNESS,
#         # MetricType.RUBRIC
#     ]
# )

# from code.src.experiment.evaluate_experiment import EvaluateExperiment
# exp = EvaluateExperiment()
# exp.id = "sematic-evaluate"
# exp.mode = "sematic-evaluate"
# exp.evaluator_llm = evaluator_llm
# exp.config = config
# exp.max_workers = 5  # 设置为1以避免并发问题
# exp.flash_fre=10 # 每10个更新下打印

# # exp.dataset_dictory="result/spacecraft/1-part/ACT"

# # parent_dir="result/spacecraft/1-part"
# parent_dir="result/spacecraft/2-part"

# print("start experiment...")
# # 再加一层循环处理
# for subdir in os.listdir(parent_dir):
#     subdir_path = os.path.join(parent_dir, subdir)
#     if os.path.isdir(subdir_path):
#         exp.dataset_dictory = subdir_path
#         print(f"Running experiment for directory: {subdir_path}")
#         exp.run_experiment_from_dictory()

# print("Experiment completed.")


# xmiContent="""<?xml version="1.0" encoding="UTF-8"?>
# <xmi:XMI xmi:version="2.1" xmlns:xmi="http://schema.omg.org/spec/XMI/2.1" xmlns:sysml="http://www.omg.org/spec/SysML/20150709/SysML">
#   <sysml:Model xmi:id="_1" name="SystemModel">
#     <ownedMember xmi:id="_2" xmi:type="sysml:Requirement" name="Reliability">
#       <text xmi:id="_3">The system shall exhibit the quality characteristic of reliability.</text>
#     </ownedMember>
#     <ownedMember xmi:id="_4" xmi:type="sysml:Requirement" name="Cost">
#       <text xmi:id="_5">The system shall exhibit the quality characteristic of cost.</text>
#     </ownedMember>
#   </sysml:Model>
# </xmi:XMI>
# """

from code.src.evaluators.syntax_evaluator.syntax_validator import XMISyntaxValidator


xmiSyntaxValidator = XMISyntaxValidator()


# res=xmiSyntaxValidator.validate(xmiContent,use_magic_valid=True)
# print(res)

from code.src.experiment.evaluate_experiment import EvaluateExperiment
exp = EvaluateExperiment()
exp.id = "syntactic-evaluate"
exp.mode = "syntactic-evaluate"
exp.syntaxValidator = xmiSyntaxValidator


exp.max_workers = 1  # 设置为1以避免并发问题
exp.flash_fre=10 # 每10个更新下打印

# exp.dataset_dictory="result/spacecraft/1-part/ACT"

parent_dir="result/spacecraft/1-part"
# parent_dir="result/spacecraft/2-part"

print("start experiment...")
# 再加一层循环处理
for subdir in os.listdir(parent_dir):
    subdir_path = os.path.join(parent_dir, subdir)
    if os.path.isdir(subdir_path):
        exp.dataset_dictory = subdir_path
        print(f"Running experiment for directory: {subdir_path}")
        exp.run_experiment_from_dictory()



# parent_dir="result/spacecraft/1-part"
parent_dir="result/spacecraft/2-part"

print("start experiment...")
# 再加一层循环处理
for subdir in os.listdir(parent_dir):
    subdir_path = os.path.join(parent_dir, subdir)
    if os.path.isdir(subdir_path):
        exp.dataset_dictory = subdir_path
        print(f"Running experiment for directory: {subdir_path}")
        exp.run_experiment_from_dictory()

print("Experiment completed.")