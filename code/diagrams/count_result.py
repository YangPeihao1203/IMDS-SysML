import pandas as pd
import numpy as np
import os
# id	requirement	focus_element	related_elements	sysml_xmi_native	time_cost_native	sysml_xmi_rule_only	time_cost_rule_only	retrived_shots_1	sysml_xmi_shots_1	time_cost_shots_1	retrived_shots_3	sysml_xmi_shots_3	time_cost_shots_3	retrived_shots_for_rule_and_shots_1	sysml_xmi_for_rule_and_shots_1	time_cost_for_rule_and_shots_1	retrived_shots_for_rule_and_shots_3	sysml_xmi_for_rule_and_shots_3	time_cost_for_rule_and_shots_3	
# sysml_xmi_native_score_by_llm	sysml_xmi_rule_only_score_by_llm	sysml_xmi_shots_1_score_by_llm	sysml_xmi_shots_3_score_by_llm	sysml_xmi_for_rule_and_shots_1_score_by_llm	sysml_xmi_for_rule_and_shots_3_score_by_llm	
# sysml_xmi_native_syntax_pass	sysml_xmi_rule_only_syntax_pass	sysml_xmi_shots_1_syntax_pass	sysml_xmi_shots_3_syntax_pass	sysml_xmi_for_rule_and_shots_1_syntax_pass	sysml_xmi_for_rule_and_shots_3_syntax_pass

# columns_interest = ["time_cost_native", 
#                     "time_cost_rule_only",
#                     "time_cost_shots_1",
#                     "time_cost_shots_3",
#                     "time_cost_for_rule_and_shots_1",
#                     "time_cost_for_rule_and_shots_3",
#                     "sysml_xmi_native_score_by_llm",
#                     "sysml_xmi_rule_only_score_by_llm",
#                     "sysml_xmi_shots_1_score_by_llm",
#                     "sysml_xmi_shots_3_score_by_llm",
#                     "sysml_xmi_for_rule_and_shots_1_score_by_llm",
#                     "sysml_xmi_for_rule_and_shots_3_score_by_llm",
#                     "sysml_xmi_native_syntax_pass",
#                     "sysml_xmi_rule_only_syntax_pass",
#                     "sysml_xmi_shots_1_syntax_pass",
#                     "sysml_xmi_shots_3_syntax_pass",
#                     "sysml_xmi_for_rule_and_shots_1_syntax_pass",
#                     "sysml_xmi_for_rule_and_shots_3_syntax_pass"]




# def print_mean(dir):
#     record_list = {} # 记录每种图的数据
#     for subdir in os.listdir(dir):
#         if subdir not in record_list:
#             record_list[subdir] = {}
#         subdir_path = os.path.join(dir, subdir)
#         pd_record = pd.DataFrame(columns=columns_interest)
#         if os.path.isdir(subdir_path):
#             for file in os.listdir(subdir_path):
#                 if file.endswith(".csv"):
#                     file_path = os.path.join(subdir_path, file)
#                     df = pd.read_csv(file_path, usecols=columns_interest)
#                     pd_record = pd.concat([pd_record, df], ignore_index=True)
#             # 计算平均值
#             pd_record_mean = pd_record.mean()
#             pd_record_mean = pd_record_mean.to_frame().T
#             pd_record_mean.index = [subdir]
#             record_list[subdir] = pd_record_mean
#     return record_list



# parent_dirs = [
#     "result/spacecraft/1-part",
#     "result/spacecraft/2-part"
# ]


# all_results = []
# for parent_dir in parent_dirs:
#     record_list = print_mean(parent_dir)
#     for subdir, record in record_list.items():
#         record["dataset"] = parent_dir  # 记录来源
#         record["subdir"] = subdir
#         all_results.append(record)

# final_df = pd.concat(all_results)
# final_df.to_csv("experiment_results.csv")


# import matplotlib.pyplot as plt
# import numpy as np

# # 数据
# diagram_types = ["BDD","IBD","STM","REQ","PKG","UCD","PAR","ACT","SEQ"]

# time_cost_native = [18.18,21.35,28.28,24.54,27.08,41.68,19.01,24.23,55.62]
# time_cost_rule_only = [24.54,17.48,36.80,36.46,28.72,37.94,29.40,30.27,95.15]
# time_cost_1_shot = [16.76,16.40,24.49,13.60,27.29,39.88,17.87,24.29,50.45]
# time_cost_3_shot = [20.39,17.57,29.50,14.48,30.05,46.91,20.16,31.01,64.52]
# time_cost_rule_1_shot = [18.82,17.54,25.32,20.07,25.26,34.85,20.80,26.88,60.87]
# time_cost_rule_3_shot = [21.74,19.95,29.53,16.09,29.42,38.58,23.86,30.08,95.09]
# mean_time_cost = [20.07,18.38,28.99,20.87,27.97,39.97,21.85,27.79,70.28]

# mean_size_of_model = [9.03,4.75,39.14,8.65,41.22,24.67,6.51,14.15,60.25]
# mean_time_of_per_element = [2.22,3.87,0.74,2.41,0.68,1.62,3.36,1.96,1.17]

# strategies = ["native","rule-only","1-shot","3-shot","rule-1-shot","rule-3-shot"]
# time_matrix = np.array([
#     time_cost_native,
#     time_cost_rule_only,
#     time_cost_1_shot,
#     time_cost_3_shot,
#     time_cost_rule_1_shot,
#     time_cost_rule_3_shot
# ])

# # -------- Figure 1: 分组柱状图 --------
# x = np.arange(len(diagram_types))
# bar_width = 0.13

# plt.figure(figsize=(12,6))
# for i in range(len(strategies)):
#     plt.bar(x + i*bar_width, time_matrix[i], width=bar_width, label=strategies[i])

# plt.xticks(x + bar_width*2.5, diagram_types)
# plt.ylabel("Time Cost (s)")
# plt.xlabel("Diagram Type")
# plt.title("Time Cost of Different Strategies across Diagram Types")
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.savefig("time_cost_comparison.png")

# # -------- Figure 2: 散点图 (平均规模 vs 平均耗时) --------
# plt.figure(figsize=(8,6))
# plt.scatter(mean_size_of_model, mean_time_cost, c='blue', s=80)

# for i, txt in enumerate(diagram_types):
#     plt.annotate(txt, (mean_size_of_model[i], mean_time_cost[i]), fontsize=9, xytext=(5,5), textcoords='offset points')

# plt.xlabel("Mean Model Size")
# plt.ylabel("Mean Time Cost (s)")
# plt.title("Correlation between Model Size and Time Cost")
# plt.tight_layout()
# plt.show()
# plt.savefig("size_vs_time_cost.png")
# # -------- Figure 3: per-element 耗时 --------
# plt.figure(figsize=(10,6))
# plt.bar(diagram_types, mean_time_of_per_element, color="skyblue")
# plt.ylabel("Mean Time per Element (s)")
# plt.xlabel("Diagram Type")
# plt.title("Normalized Time Cost per Model Element")
# plt.tight_layout()
# plt.show()
# plt.savefig("mean_time_per_element.png")




# import matplotlib.pyplot as plt
# import numpy as np

# # # 数据
# diagram_types = ["BDD","IBD","STM","REQ","PKG","UCD","PAR","ACT","SEQ"]
# mean_size_of_model = np.array([9.03,4.75,39.14,8.65,41.22,24.67,6.51,14.15,60.25])
# mean_time_cost = np.array([20.07,18.38,28.99,20.87,27.97,39.97,21.85,27.79,70.28])
# mean_time_of_per_element = np.array([2.221622807,3.866351949,0.740534776,2.413466146,
#                                      0.678560338,1.620546441,3.355709455,1.963892273,1.166519073])

# # # ------- 气泡散点图（点大小/颜色 = per-element time） -------
# plt.figure(figsize=(6,3.6))
# scatter = plt.scatter(
#     mean_size_of_model,
#     mean_time_cost,
#     s=mean_time_of_per_element * 120,  # 气泡大小比例，必要时调大/调小
#     c=mean_time_of_per_element,        # 颜色映射
#     cmap="viridis",
#     alpha=0.85,
#     edgecolor="k",
#     linewidth=0.6
# )

# # ------- 手动可调标签偏移：根据需要改这里即可 -------
# # (dx, dy) 单位是像素；正数向右/上，负数向左/下
# label_offsets = {
#     "BDD": (9, 1),
#     "IBD": (10, -12),
#     "STM": (-16, 6),
#     "REQ": (-3, 10),
#     "PKG": (6, 6),
#     "UCD": (6, 6),
#     "PAR": (-8, 12),
#     "ACT": (6, 6),
#     "SEQ": (-16, -16),
# }

# for i, name in enumerate(diagram_types):
#     dx, dy = label_offsets.get(name, (6,6))
#     plt.annotate(
#         name,
#         (mean_size_of_model[i], mean_time_cost[i]),
#         xytext=(dx, dy),
#         textcoords="offset points",
#         fontsize=9
#     )

# # ------- 线性回归（numpy.polyfit，无需额外依赖） -------
# slope, intercept = np.polyfit(mean_size_of_model, mean_time_cost, 1)
# x_line = np.linspace(mean_size_of_model.min(), mean_size_of_model.max(), 200)
# y_line = slope * x_line + intercept

# # 计算 R^2
# y_pred = slope * mean_size_of_model + intercept
# ss_res = np.sum((mean_time_cost - y_pred)**2)
# ss_tot = np.sum((mean_time_cost - mean_time_cost.mean())**2)
# r2 = 1 - ss_res/ss_tot

# plt.plot(x_line, y_line, linestyle="--", label=f"Linear Fit (R$^2$={r2:.2f})")

# # 轴标签/标题/色条
# plt.xlabel("Mean Model Size")
# plt.ylabel("Mean Time Cost (s)")
# plt.title("Model Size vs Time Cost (bubble size/color = per-element time)")
# cbar = plt.colorbar(scatter)
# cbar.set_label("Mean Time per Element (s)")
# plt.legend(loc="best")
# plt.tight_layout()

# plt.xlabel("Mean Model Size", fontsize=10)
# plt.ylabel("Mean Time Cost (s)", fontsize=10)
# plt.title("Model Size vs Time Cost", fontsize=11)
# plt.legend(loc="best", fontsize=9)
# cbar.ax.tick_params(labelsize=9)
# cbar.set_label("Mean Time per Element (s)", fontsize=10)

# # 如需直接保存PDF，取消下一行注释
# plt.savefig("figure2_bubble_reg.pdf", bbox_inches="tight", pad_inches=0.1)

# plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# # 数据
# diagram_types = ["BDD","IBD","STM","REQ","PKG","UCD","PAR","ACT","SEQ"]
# strategies = ["native","rule-only","1-shot","3-shot","rule-1-shot","rule-3-shot"]

# time_matrix = np.array([
#     [18.18, 24.54, 16.76, 20.39, 18.82, 21.74],
#     [21.35, 17.48, 16.40, 17.57, 17.54, 19.95],
#     [28.28, 36.80, 24.49, 29.50, 25.32, 29.53],
#     [24.54, 36.46, 13.60, 14.48, 20.07, 16.09],
#     [27.08, 28.72, 27.29, 30.05, 25.26, 29.42],
#     [41.68, 37.94, 39.88, 46.91, 34.85, 38.58],
#     [19.01, 29.40, 17.87, 20.16, 20.80, 23.86],
#     [24.23, 30.27, 24.29, 31.01, 26.88, 30.08],
#     [55.62, 95.15, 50.45, 64.52, 60.87, 95.09],
# ])

# # --------- Figure 1: 分组柱状图 ---------
# x = np.arange(len(diagram_types))
# bar_width = 0.13

# plt.figure(figsize=(12,6))
# for i in range(len(strategies)):
#     plt.bar(x + i*bar_width, time_matrix[:, i], width=bar_width, label=strategies[i])

# plt.xticks(x + bar_width*2.5, diagram_types)
# plt.ylabel("Time Cost (s)")
# plt.xlabel("Diagram Type")
# plt.title("Time Cost of Different Strategies across Diagram Types")
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.savefig("time_cost_comparison.png")

# # --------- Figure 2: 热力图 ---------
# plt.figure(figsize=(10,6))
# sns.heatmap(time_matrix, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=strategies, yticklabels=diagram_types)
# plt.xlabel("Strategy")
# plt.ylabel("Diagram Type")
# plt.title("Heatmap of Time Cost for Different Strategies and Diagram Types")
# plt.tight_layout()
# plt.show()
# plt.savefig("time_cost_heatmap.png")




# import matplotlib.pyplot as plt
# import numpy as np

# # 数据
# diagram_types = ["BDD","IBD","STM","REQ","PKG","UCD","PAR","ACT","SEQ"]
# strategies = ["native","rule-only","1-shot","3-shot","rule-1-shot","rule-3-shot"]

# # 原表是 diagram_types × strategies，转置后得到策略 × 图类型
# time_matrix = np.array([
#     [18.18, 24.54, 16.76, 20.39, 18.82, 21.74],
#     [21.35, 17.48, 16.40, 17.57, 17.54, 19.95],
#     [28.28, 36.80, 24.49, 29.50, 25.32, 29.53],
#     [24.54, 36.46, 13.60, 14.48, 20.07, 16.09],
#     [27.08, 28.72, 27.29, 30.05, 25.26, 29.42],
#     [41.68, 37.94, 39.88, 46.91, 34.85, 38.58],
#     [19.01, 29.40, 17.87, 20.16, 20.80, 23.86],
#     [24.23, 30.27, 24.29, 31.01, 26.88, 30.08],
#     [55.62, 95.15, 50.45, 64.52, 60.87, 95.09],
# ])

# # 转置矩阵：策略为主轴
# time_matrix_T = time_matrix.T  # shape: strategies × diagram_types

# # 绘制分组柱状图
# x = np.arange(len(strategies))  # 横坐标是策略
# bar_width = 0.13

# plt.figure(figsize=(12,6))
# for i in range(len(diagram_types)):
#     plt.bar(x + i*bar_width, time_matrix_T[:, i], width=bar_width, label=diagram_types[i])

# plt.xticks(x + bar_width*4, strategies)  # 横坐标放策略名称
# plt.xlabel("Strategy")
# plt.ylabel("Time Cost (s)")
# plt.title("Time Cost across Diagram Types for Each Strategy")
# plt.legend(title="Diagram Type", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()
# plt.show()
# plt.savefig("time_cost_comparison_transposed.png")