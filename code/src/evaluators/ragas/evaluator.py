# src/evaluation/evaluator.py
from ragas import evaluate

from ragas import SingleTurnSample, EvaluationDataset
from code.src.evaluators.ragas.config import EvaluationConfig
from code.src.evaluators.ragas.metric_factory import get_metric

def evaluate_sysml_generation(dataset: EvaluationDataset, llm, config: EvaluationConfig):
    metrics = []
    # extra_config = config.model_dump(exclude={"metrics"})
    extra_config = config.to_extra_config()

    for metric_type in config.metrics:
        metric = get_metric(metric_type, llm, extra_config=extra_config)
        metrics.append(metric)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm
    )
    return result
