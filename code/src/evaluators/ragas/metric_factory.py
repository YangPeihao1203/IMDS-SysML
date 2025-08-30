# src/evaluation/metric_factory.py
from ragas.metrics import (
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
    Faithfulness,
    RubricsScore,
)
from code.src.evaluators.ragas.types import MetricType

def get_metric(metric_type: MetricType, llm, extra_config: dict = None):
    # print("extra_config:", extra_config)
    if metric_type == MetricType.ANSWER_ACCURACY:
        print("Creating AnswerAccuracy metric")
        metric = AnswerAccuracy(llm=llm)

        # Inject custom templates if provided
        if "template_accuracy1" in extra_config:
            # print("Using custom template_accuracy1")
            # print(extra_config["template_accuracy1"])
            setattr(metric, "template_accuracy1", extra_config["template_accuracy1"])
        if "template_accuracy2" in extra_config:
            setattr(metric, "template_accuracy2", extra_config["template_accuracy2"])

        return metric
    elif metric_type == MetricType.CONTEXT_RELEVANCE:
        metric = ContextRelevance(llm=llm)
        if "template_relevance1" in extra_config:
            # print("Using custom template_relevance1")
            setattr(metric, "template_relevance1", extra_config["template_relevance1"])
        if "template_relevance2" in extra_config:
            setattr(metric, "template_relevance2", extra_config["template_relevance2"])
        return metric
    
    elif metric_type == MetricType.RESPONSE_GROUNDEDNESS:
        metric = ResponseGroundedness(llm=llm)
        if "template_groundedness1" in extra_config:
            # print("Using custom template_groundedness1")template_groundedness2
            setattr(metric, "template_groundedness1", extra_config["template_groundedness1"])
        if "template_groundedness2" in extra_config:
            setattr(metric, "template_groundedness2", extra_config["template_groundedness2"])
        return metric
    elif metric_type == MetricType.FAITHFULNESS:
        return Faithfulness(llm=llm)
    elif metric_type == MetricType.RUBRIC:
        rubrics = extra_config.get("rubric_score_descriptions", {})
        return RubricsScore(rubrics=rubrics, llm=llm)
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")