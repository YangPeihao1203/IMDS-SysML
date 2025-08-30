"""
This script counts the number of files in a dataset directory.
"""
import os
import csv
import sys
#flags=["STM","REQ","BDD","PKG","UCD","PAR","IBD","ACT","SEQ"]
csv.field_size_limit(sys.maxsize)





sysml_project_domain_map_short = {
    "WaterSupply-xml": "Others",
    "VehicleCCS Solution-xml": "Automotive",
    "OntologicalBehaviorModeling-xml": "Automotive",
    "DynamicBatteryPowerRollup-xml": "Electrical",
    "Orbital-xml": "Aerospace",
    "InvertedPendulum-xml": "Electrical",
    "VehicleStructure-xml": "Automotive",
    "HMI System Solution-xml": "Automotive",
    "VehicleCCS_Implementation-xml": "Automotive",
    "Heating System Solution-xml": "Building",
    "Categorization requirements-xml": "Software",
    "layout templates-xml": "Consumer Electronics",
    "VehicleCCS Configuration-xml": "Automotive",
    "Suspect Links-xml": "Automotive",
    "layers-xml": "Consumer Electronics",
    "MultiMediaExample-xml": "Consumer Electronics",
    "CruiseControl-xml": "Automotive",
    "HomeHeating-xml": "Building",
    "HingeMonteCarloAnalysis-xml": "Mechanical",
    "basic_geometries-xml": "Mechanical",
    "Behavior-to-Structure Synchronization-xml": "Consumer Electronics",
    "TemperatureRegulationLoop-xml": "Others",
    "OOSEM project structure example-xml": "Aerospace",
    "SwayOfCrane-xml": "Industrial",
    "TradeStudyExamples-xml": "Automotive",
    "Satellite-xml": "Aerospace",
    "VehicleCCU Problem-xml": "Automotive",
    "BlackJack-xml": "Consumer Electronics",
    "Requirements-xml": "Automotive",
    "SpaceStation-xml": "Aerospace",
    "UI System_Solution-xml": "Others",
    "MultiVariantCarModel-xml": "Automotive",
    "DHCPInteraction-xml": "Industrial",
    "modem cable-xml": "Industrial",
    "CruiseControl_Widgets-xml": "Automotive",
    "Auxilary resources-xml": "Software",
    "requirements verification in table-xml": "Automotive",
    "Trade-xml": "Finance",
    "CylinderPipe-xml": "Mechanical",
    "Conjugated Interface Block-xml": "Mechanical",
    "Circuit-xml": "Electrical",
    "VehicleMassRollup-xml": "Automotive",
    "OpAmp-xml": "Electrical",
    "SmallTestSamples-xml": "Others",
    "SpringDisplacementUsingTimevariable-xml": "Mechanical",
    "Insurance-xml": "Finance",
    "Acme_Library-xml": "Aerospace",
    "SimulationComponentsLibrary-xml": "Software",
    "SignalProcessor-xml": "Others",
    "Humidifier-xml": "Building",
    "diagram aspects-xml": "Automotive",
    "ProjectPlan-xml": "Software",
    "contextual relationships-xml": "Automotive",
    "CP System Solution-xml": "Industrial",
    "4_VehicleCCU_B1_W1_final-xml": "Automotive",
    "Information System for Training Organization-xml": "Software",
    "LinkageSystems-xml": "Automotive",
    "Implied Relations-xml": "Automotive",
    "Scope via Package Imports-xml": "Automotive",
    "LaptopCostAnalysis-xml": "Consumer Electronics",
    "UPDM-BPMN Sample-xml": "Military",
    "Sensors System_Solution-xml": "Others",
    "Excel_CSV Import SysML-xml": "Automotive",
    "MotionAnalysis-xml": "Mechanical",
    "hybrid sport utility vehicle-xml": "Automotive",
    "FlashingLight-xml": "Industrial",
    "Excel_Sync-xml": "Automotive",
    "Hydraulics-xml": "Mechanical",
    "TradeStudyPattern-xml": "Automotive",
    "Electronics-xml": "Electrical",
    "LittleEye-xml": "Aerospace",
    "distiller model-xml": "Others",
    "BouncingBall-xml": "Consumer Electronics",
    "Addition-xml": "Others",
    "SCARA manipulator-xml": "Mechanical",
    "VCCS (MG BoK v2 Safety and Reliability Analysis Sample)-xml": "Automotive",
    "Trade-Study for Brayton Cycle-xml": "Energy",
    "Medical FMEA and Hazard Analysis-xml": "Automotive",
    "CoffeeMachine-xml": "Consumer Electronics",
    "ForwardContractValuation-xml": "Finance",
    "ActParIngerate-xml": "Others",
    "ElectricCircuit-xml": "Electrical",
    "SpacecraftMassRollup-xml": "Aerospace",
    "Financial-xml": "Finance",
    "MATLAB_Examples-xml": "Electrical",
    "Thermostat-xml": "Others",
    "climate control system-xml": "Others",
    "StreamingActivity-xml": "Industrial",
    "Counting FTEs (requires CST)-xml": "Software",
    "context specific values-xml": "Software",
    "Transmission-xml": "Automotive",
    "VCCS [requires CST]-xml": "Automotive",
    "MagicLibrary requirements-xml": "Software",
    "DurationConstraint-xml": "Software",
    "HomeController-xml": "Others",
    "EmergencyResponseAnalysis-xml": "Military",
    "LittleEyeTradeStudy-xml": "Aerospace",
    "StopWatchTestingSysML-xml": "Software",
    "Filtering System Solution-xml": "Automotive",
    "Cooling System Solution-xml": "Automotive",
    "CarBrakingAnalysis-xml": "Automotive",
    "WaterTankFMI-xml": "Others",
    "secret aircraft model-xml": "Aerospace",
    "MODAF Sample-xml": "Military",
    "CommNetwork-xml": "Industrial",
    "basic units-xml": "Mechanical",
    "Home Appliances Enterprise-xml": "Others",
    "Automotive FMEA-xml": "Automotive",
    "SysML1.3 Interfaces Modeling-xml": "Industrial",
    "User needs - requirements module for MagicLibrary-xml": "Software",
    "Transmission_Widgets-xml": "Automotive",
    "SpringSystems-xml": "Mechanical",
    "NAF 4.0 Sample-xml": "Military",
    "DoDAF Sample-xml": "Military",
    "WebUIWidgets-xml": "Others",
    "Property Based Requirements-xml": "Automotive",
    "legends-xml": "Automotive",
    "simple_parametrics-xml": "Others",
    "Introduction to SysML-xml": "Aerospace",
    "Banking-xml": "Finance",
    "Electric Power Steering - Functional Safety Analysis-xml": "Automotive",
    "architecting-spacecraft-xml": "Aerospace"
}




industry_application = (
    "Automotive",           # 汽车工程
    "Aerospace",            # 航空航天
    "Mechanical",           # 机械工程与工业控制
    "Electrical",           # 电子工程与电力系统
    "Consumer Electronics", # 消费电子与多媒体系统
    "Finance",              # 金融与保险
    "Building",             # 建筑与能源
    "Industrial",           # 工业自动化与网络通信
    "Military",             # 应急响应与军事
    "Energy",               # 能源动力

)

# Methodology / Tool Support: projects related to system modeling methods, software engineering, and general tools
methodology_tool = (
    "Software",              # 软件工程与信息系统 / 系统建模与方法学
    "Others"                # 其他/通用
)


# 每种图下的类型的映射
diagram_type_map = {
    "STM": ("StateMachine"),
    "REQ": ("DeriveReqt","Requirement","Trace"),
    "BDD": ("AssociationClass","Block","InterfaceBlock","ValueType"),
    "PKG": ("Package"),
    "UCD": ("Actor","UseCase","Stakeholder","Viewpoint"),
    "PAR": ("caused_by","ConstraintBlock"),
    "IBD": ("Connector","BindingConnector","InformationFlow","ItemFlow"),
    "ACT": ("Activity","TestCase"),
    "SEQ": ("Interaction")
}



record={}
dataset_train_dir="dataset/dataset_native-xmi-csv/train_set"
dataset_test_dir="dataset/dataset_native-xmi-csv/test_set"

train_projects=[]
test_projects=[]

all_dirs=[dataset_train_dir,dataset_test_dir]
for i,base_dir in enumerate(all_dirs):
    for project_dir in os.listdir(base_dir):
        if i==0:
            train_projects.append(project_dir)
        else:
            test_projects.append(project_dir)

        project_path = os.path.join(base_dir, project_dir)
        if not os.path.isdir(project_path):
            continue
        record[project_dir] = {}
        for diagram_type_dir in os.listdir(project_path):
            diagram_type_path = os.path.join(project_path, diagram_type_dir)
            if not os.path.isdir(diagram_type_path):
                continue
            # 统计每个类型的文件数量
            record[project_dir][diagram_type_dir] = {}
            for file_name in os.listdir(diagram_type_path):
                if file_name.endswith('.csv'):
                    file_cleaned = file_name.replace('_dataset.xmi.csv', '')
                    file_path = os.path.join(diagram_type_path, file_name)
                    # 读csv,统计除了头部有多少数据量
                    with open(file_path, 'r', encoding='utf-8') as csvfile:
                        reader = csv.reader(csvfile)
                        header = next(reader)
                        count = sum(1 for _ in reader)
                    record[project_dir][diagram_type_dir][file_cleaned] = count



# print("Dataset file counts:")
# for project, types in record.items():
#     print(f"Project: {project}")
#     for diagram_type, files in types.items():
#         print(f"  Diagram Type: {diagram_type}")
#         for file_name, count in files.items():
#             print(f"    {file_name}: {count} records")


# 开始按照各项进行统计
# 1. 统计总的数目
total_count = 0
for project, types in record.items():
    for diagram_type, files in types.items():
        for file_name, count in files.items():
            total_count += count
print(f"Total records in dataset: {total_count}")
# 1.1 统计dataset_train_dir下总的数目和dataset_test_dir下总的数目
train_count = 0
test_count = 0
for project, types in record.items():
    
    for diagram_type, files in types.items():
        for file_name, count in files.items():
            file_path = os.path.join(project, diagram_type, file_name + '.csv')
            if project in train_projects:
                train_count += count
            else:
                test_count += count
print(f"Total records in train set: {train_count}")
print(f"Total records in test set: {test_count}")



# 1.2 统计各种图的数目，比如BDD、IBD等
diagram_type_counts = {}
for project, types in record.items():
    for diagram_type, files in types.items():
        flag=False
        for file_cleaned, count in files.items():
            for diagram_type_key, diagram_type_value in diagram_type_map.items():
                if file_cleaned in diagram_type_value:
                    if diagram_type_key not in diagram_type_counts:
                        diagram_type_counts[diagram_type_key] = 0
                    diagram_type_counts[diagram_type_key] += count
                    flag=True
                    break
        if not flag:
            print(f"Warning: {file_cleaned} in {project} does not match any known diagram type.")
print("Counts of each diagram type:")
for diagram_type, count in diagram_type_counts.items():
    print(f"{diagram_type}: {count} records")


# 1.3 统计每个项目的数目
project_counts = {}
for project, types in record.items():
    project_counts[project] = 0
    for diagram_type, files in types.items():
        for file_name, count in files.items():
            project_counts[project] += count
print("Counts of each project:")
for project, count in project_counts.items():
    print(f"{project}: {count} records")
print("project number:", len(project_counts))



num_project=len(sysml_project_domain_map_short)
print("Number of projects that we mannual set:", num_project)

# check if project_counts's name is in sysml_project_domain_map_short
for project in project_counts.keys():
    if project not in sysml_project_domain_map_short:
        print(f"Warning: {project} is not in sysml_project_domain_map_short.")
    else:
        pass

# 1.4 统计每个领域的项目数目
domain_map={}
for project, count in project_counts.items():
    if project in sysml_project_domain_map_short:
        domain = sysml_project_domain_map_short[project]
        if domain not in domain_map:
            domain_map[domain] = 0
        domain_map[domain] += count
    else:
        print(f"Warning: {project} not found in domain map.")
print("Counts of each domain:")
for domain, count in domain_map.items():
    print(f"{domain}: {count} records")

# 1.4.1 进一步的，统计大类的数目
large_domain_map = {}

for domain, count in domain_map.items():
    if domain in industry_application:
        large_domain = "Industry Application"
    elif domain in methodology_tool:
        large_domain = "Methodology / Tool Support"
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    if large_domain not in large_domain_map:
        large_domain_map[large_domain] = 0
    large_domain_map[large_domain] += count
print("Counts of each large domain:")
for large_domain, count in large_domain_map.items():
    print(f"{large_domain}: {count} records")

# 2 统计每个每种图的模型的长度，我好画箱体图

from code.src.evaluators.syntax_evaluator.syntax_validator import XMISyntaxValidator

xmiSyntaxValidator=XMISyntaxValidator()

from code.src.experiment.util import get_elements_size



diagram_modelsize_count_record={}

for diagram_type in diagram_type_map.keys():
    diagram_modelsize_count_record[diagram_type] = {}


for base_dir in all_dirs:
    for project_dir in os.listdir(base_dir):
        project_path = os.path.join(base_dir, project_dir)
        if not os.path.isdir(project_path):
            continue
        for diagram_type_dir in os.listdir(project_path):
            diagram_type_path = os.path.join(project_path, diagram_type_dir)
            if not os.path.isdir(diagram_type_path):
                continue

            for file_name in os.listdir(diagram_type_path):
                if file_name.endswith('.csv'):
                    file_cleaned = file_name.replace('_dataset.xmi.csv', '')
                    diagram_type =None
                    for diagram_type_key, diagram_type_value in diagram_type_map.items():
                        if file_cleaned in diagram_type_value:
                            diagram_type = diagram_type_key
                            break
                    if diagram_type is None:
                        print(f"Warning: {file_cleaned} in {project_dir} does not match any known diagram type.")
                        continue
                    file_path = os.path.join(diagram_type_path, file_name)
                    # 读文件的每条数据,每个文件的第一行是表头，然后有一列的名称叫做 "focus_element"，我们统计这列
                    with open(file_path, 'r', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            if "focus_element" in row and row["focus_element"]:
                                focus_element = row["focus_element"]
                                content=xmiSyntaxValidator.wrap_parent_content(focus_element)
                                # 统计这个focus_element的模型大小
                                xml_content = content
                                xml_size = get_elements_size(xml_content)
                                if xml_size not in diagram_modelsize_count_record[diagram_type]:
                                    diagram_modelsize_count_record[diagram_type][xml_size] = 0
                                diagram_modelsize_count_record[diagram_type][xml_size] += 1

# output the modelsize
for diagram_type, size_count in diagram_modelsize_count_record.items():
    print(f"Diagram Type: {diagram_type}")
    for size, count in sorted(size_count.items()):
        print(f"  Model Size: {size}, Count: {count}")
    print()  # Add a newline for better readability

                    



import matplotlib.pyplot as plt
import numpy as np

# # 假设你已有 diagram_modelsize_count_record
# # 先把数据展开成一个列表，方便统计
expanded_data = {}

for diagram_type, size_count in diagram_modelsize_count_record.items():
    sizes = []
    for size, count in size_count.items():
        sizes.extend([size] * count)  # 按出现次数展开
    expanded_data[diagram_type] = sizes

# # 计算统计量
stats = {}
for diagram_type, sizes in expanded_data.items():
    if len(sizes) == 0:
        continue
    sizes_array = np.array(sizes)
    stats[diagram_type] = {
        "count": len(sizes),
        "min": np.min(sizes_array),
        "q1": np.percentile(sizes_array, 25),
        "median": np.median(sizes_array),
        "q3": np.percentile(sizes_array, 75),
        "max": np.max(sizes_array),
        "mean": np.mean(sizes_array)
    }

# 打印统计量
for diagram_type, s in stats.items():
    print(f"\nDiagram Type: {diagram_type}")
    for k, v in s.items():
        print(f"  {k}: {v}")


# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(12, 6))

# # 画箱体图
# box = plt.boxplot(
#     [expanded_data[d] for d in expanded_data.keys()],
#     tick_labels=list(expanded_data.keys()),
#     patch_artist=True,
#     showfliers=False
# )

# plt.yscale('log')
# plt.xticks(rotation=45)
# plt.ylabel("Model Size (#elements)")
# plt.title("Model Size Distribution Across Diagram Types")
# plt.tight_layout()

# # 给每个箱体加数值标签：最小值、最大值、中位数
# for i, diagram_type in enumerate(expanded_data.keys(), start=1):
#     sizes = np.array(expanded_data[diagram_type])
#     median = np.median(sizes)
#     min_val = np.min(sizes)
#     max_val = np.max(sizes)
    
#     plt.text(i, median, f"{median:.0f}", ha='center', va='bottom', color='red', fontsize=10)

# # 保存高分辨率 PNG
# plt.savefig("diagram_model_size_distribution_with_labels.pdf")
# plt.show()



# 饼状图-领域占比 饼状图我都是 用excel手动画的



# 小提琴图
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(12, 6))

# seaborn violinplot 需要把数据整理成长格式
diagram_types = []
sizes = []
for diagram_type, size_list in expanded_data.items():
    diagram_types.extend([diagram_type] * len(size_list))
    sizes.extend(size_list)

# 绘制小提琴图
sns.violinplot(x=diagram_types, y=sizes, inner="quartile", palette="Blues")

# 对数 y 轴，方便展示数量级差异大时
plt.yscale('log')

plt.xticks(rotation=45)
plt.ylabel("Model Size (#elements)")
plt.title("Model Size Distribution Across Diagram Types (Violin Plot)")
plt.tight_layout()

# 保存为 PDF
plt.savefig("diagram_model_size_violin_plot.pdf")
plt.show()



# 堆叠的条形图
# import matplotlib.pyplot as plt

# # 数据
# domain_counts = {
#     "Others": 835,
#     "Software": 378,
#     "Automotive": 2214,
#     "Electrical": 392,
#     "Aerospace": 1390,
#     "Energy": 80,
#     "Building": 200,
#     "Consumer Electronics": 818,
#     "Mechanical": 344,
#     "Industrial": 315,
#     "Finance": 301,
#     "Military": 68
# }

# # 大类划分
# industry_application = (
#     "Automotive",
#     "Aerospace",
#     "Mechanical",
#     "Electrical",
#     "Consumer Electronics",
#     "Finance",
#     "Building",
#     "Industrial",
#     "Military",
#     "Energy"
# )

# methodology_tool = (
#     "Software",
#     "Others"
# )

# # 准备数据
# ind_counts = [domain_counts[d] for d in industry_application]
# meth_counts = [domain_counts[d] for d in methodology_tool]

# # 专业配色（低饱和度）
# color_industry = '#4C72B0'  # 沉稳蓝色
# color_method = '#DD8452'    # 柔和橙色

# # 横向条形图
# fig, ax = plt.subplots(figsize=(10, 6))

# # y 位置
# y_ind = range(len(industry_application))
# y_meth = range(len(industry_application), len(industry_application) + len(methodology_tool))

# # 绘制条形
# ax.barh(y=y_ind, width=ind_counts, color=color_industry, label='Industry Application')
# ax.barh(y=y_meth, width=meth_counts, color=color_method, label='Methodology / Tool')

# # 标注数量
# for i, v in enumerate(ind_counts):
#     ax.text(v + 20, i, str(v), va='center', fontsize=9)
# for i, v in enumerate(meth_counts):
#     ax.text(v + 20, i + len(industry_application), str(v), va='center', fontsize=9)

# # 设置纵坐标标签
# yticks = list(y_ind) + list(y_meth)
# yticklabels = list(industry_application) + list(methodology_tool)
# ax.set_yticks(yticks)
# ax.set_yticklabels(yticklabels)

# # 坐标轴与图例
# ax.set_xlabel("Count")
# ax.set_title("Comparison of SysML Dataset: Industry vs Methodology/Tool", fontsize=14)
# ax.legend(loc='upper right')

# plt.tight_layout()

# # 保存为 PDF
# plt.savefig("sysml_industry_vs_methodology_professional.pdf")
# plt.show()


# 统计了下 测试集的具体组成
dataset_test_dir="dataset/dataset_native-xmi-csv/test_set"

train_projects=[]
test_projects=[]
record={}
all_dirs=[dataset_test_dir]
for i,base_dir in enumerate(all_dirs):
    for project_dir in os.listdir(base_dir):
        if i==0:
            train_projects.append(project_dir)
        else:
            test_projects.append(project_dir)

        project_path = os.path.join(base_dir, project_dir)
        if not os.path.isdir(project_path):
            continue
        record[project_dir] = {}
        for diagram_type_dir in os.listdir(project_path):
            diagram_type_path = os.path.join(project_path, diagram_type_dir)
            if not os.path.isdir(diagram_type_path):
                continue
            # 统计每个类型的文件数量
            record[project_dir][diagram_type_dir] = {}
            for file_name in os.listdir(diagram_type_path):
                if file_name.endswith('.csv'):
                    file_cleaned = file_name.replace('_dataset.xmi.csv', '')
                    file_path = os.path.join(diagram_type_path, file_name)
                    # 读csv,统计除了头部有多少数据量
                    with open(file_path, 'r', encoding='utf-8') as csvfile:
                        reader = csv.reader(csvfile)
                        header = next(reader)
                        count = sum(1 for _ in reader)
                    record[project_dir][diagram_type_dir][file_cleaned] = count
print("Dataset file counts:")
for project, types in record.items():
    print(f"Project: {project}")
    for diagram_type, files in types.items():
        print(f"  Diagram Type: {diagram_type}")
        for file_name, count in files.items():
            print(f"    {file_name}: {count} records")