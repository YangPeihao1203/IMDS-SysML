
from enum import Enum

class MetricType(str, Enum):
    ANSWER_ACCURACY = "answer_accuracy" # 生成的答案和标准答案做对比，与correctness相比则是缺少事实性的额外校验
    CONTEXT_RELEVANCE = "context_relevance" # 检索到的内容和用户输入是否相关
    RESPONSE_GROUNDEDNESS = "response_groundedness" # 生成的答案和检索到的内容做对比，是否有事实性错误
    FAITHFULNESS = "faithfulness"
    RUBRIC = "rubric"  # 自己设计打分的维度,针对的sysml