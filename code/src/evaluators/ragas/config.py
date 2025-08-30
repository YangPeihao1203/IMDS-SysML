# src/evaluation/config.py
from pydantic import BaseModel
from typing import List, Optional, Dict
from code.src.evaluators.ragas.types import MetricType
from typing import List, Optional, Dict, ClassVar

class EvaluationConfig(BaseModel):
    metrics: List[MetricType]
    rubric_score_descriptions: Optional[Dict[str, str]] = {
    "score1_description": "The generated SysML model is entirely incorrect, irrelevant, or incomprehensible.",
    "score2_description": "The model has some useful components, but contains major structural errors (e.g., incorrect blocks, missing ports).",
    "score3_description": "The model is mostly correct but lacks key details or deviates from SysML conventions.",
    "score4_description": "The model is structurally sound and semantically clear, with only minor omissions.",
    "score5_description": "The model is complete, correct, and conforms to SysML standards and domain expectations."
    ""
}
    template_accuracy1: ClassVar[str] = (
        "Instruction: You are a SysML modeling expert. Evaluate the Generated SysML model fragment "
        "by comparing it against the Requirement and the Reference model.\n"
        "Consider three aspects:\n"
        "1. Semantic consistency: Are the SysML elements and relationships semantically equivalent?\n"
        "2. Requirement coverage: Does the generated model satisfy all key points of the Requirement?\n"
        "3. Model size similarity: Is the level of detail and complexity comparable to the Reference model?\n"
        "Scoring rules:\n"
        "4 → Fully correct across all three aspects.\n"
        "3 → Mostly correct: minor omissions or inconsistencies in one aspect.\n"
        "2 → Partially correct: noticeable gaps in semantics or requirement coverage, or clear size mismatch.\n"
        "1 → Weak match: only minimal overlap with requirement or reference model.\n"
        "0 → Completely invalid: irrelevant, nonsensical, or fails to satisfy the requirement at all.\n"
        "You may give a decimal score (e.g., 2.5) to reflect intermediate quality.\n"
        "Only output a single number between 0 and 4. No explanation.\n"
        "### Requirement:\n{query}\n"
        "### Generated Model:\n{sentence_inference}\n"
        "### Reference Model:\n{sentence_true}\n"
        "Rating: "
    )
    template_accuracy2: ClassVar[str] = (
        "You are reviewing SysML models for alignment with requirements.\n"
        "Rate how well the Generated model aligns with the Reference model in terms of:\n"
        "1. Semantic consistency of SysML elements and relations.\n"
        "2. Coverage of the original Requirement.\n"
        "3. Similarity of model size and complexity to the Reference model.\n"
        "Scoring rules:\n"
        "4 → Fully aligned on semantics, requirement coverage, and size.\n"
        "3 → Largely aligned, with only minor issues in one dimension.\n"
        "2 → Partially aligned, several important mismatches.\n"
        "1 → Weak alignment, only small fragments match.\n"
        "0 → No alignment, irrelevant or invalid model.\n"
        "You may give a decimal score between 0 and 4 to reflect partial correctness.\n"
        "Only output a single number between 0 and 4. No explanation.\n\n"
        "Requirement:\n{query}\n\n"
        "Reference Model:\n{sentence_true}\n\n"
        "Generated Model:\n{sentence_inference}\n\n"
        "Rating: "
    )
    
    template_relevance1: ClassVar[str] = (
    "### Instructions\n\n"
    "You are a world class expert designed to evaluate the relevance score of a Context"
    " in order to answer the Question.\n"
    "Your task is to determine if the Context contains proper information to answer the Question.\n"
    "Do not rely on your previous knowledge about the Question.\n"
    "Use only what is written in the Context and in the Question.\n"
    "Follow the instructions below:\n"
    "0. If the context does not contains any relevant information to answer the question, say 0.\n"
    "1. If the context partially contains relevant information to answer the question, say 1.\n"
    "2. If the context contains all relevant information to fully answer the question, say 2.\n"
    "You must provide the relevance score of 0, 1, or 2, nothing else.\n"
    "Do not explain.\n\n"
    "### Example 1\n"
    "Question: The Vehicle In Use shall include a Climate Control Unit, an Energy Supply, and at least one Vehicle Occupant to provide a comfortable temperature environment for occupants.\n"
    "Context:\n"
    "<packagedElement name=\"Feel Comfortable Temperature\" xmi:type=\"uml:UseCase\"/>\n"
    "<packagedElement name=\"Climate Control Unit\" xmi:type=\"uml:Class\"/>\n"
    "<packagedElement name=\"Energy Supply\" xmi:type=\"uml:Class\"/>\n"
    "<packagedElement name=\"Vehicle Occupant\" xmi:type=\"uml:Class\"/>\n"
    "Relevance score: 2\n\n"
    "### Question: {query}\n\n"
    "### Context: {context}\n\n"
    "Analyzing Context and Question, the Relevance score is "
)
    template_relevance2: ClassVar[str] = (
    "As a specially designed expert to assess the relevance score of a given Context in relation to a Question, "
    "my task is to determine the extent to which the Context provides information necessary to answer the Question. "
    "I will rely solely on the information provided in the Context and Question, and not on any prior knowledge.\n\n"
    "Here are the instructions I will follow:\n"
    "* If the Context does not contain any relevant information to answer the Question, I will respond with a relevance score of 0.\n"
    "* If the Context partially contains relevant information to answer the Question, I will respond with a relevance score of 1.\n"
    "* If the Context contains all relevant information to fully answer the Question, I will respond with a relevance score of 2.\n\n"
    "### Example\n"
    "Question: The Vehicle In Use shall include a Climate Control Unit, an Energy Supply, and at least one Vehicle Occupant to provide a comfortable temperature environment for occupants.\n"
    "Context:\n"
    "<packagedElement name=\"Feel Comfortable Temperature\" xmi:type=\"uml:UseCase\"/>\n"
    "<packagedElement name=\"Climate Control Unit\" xmi:type=\"uml:Class\"/>\n"
    "<packagedElement name=\"Energy Supply\" xmi:type=\"uml:Class\"/>\n"
    "<packagedElement name=\"Vehicle Occupant\" xmi:type=\"uml:Class\"/>\n"
    "Relevance score: 2\n\n"
    "### Question: {query}\n\n"
    "### Context: {context}\n\n"
    "Based on the provided Question and Context, the Relevance score is "
)
    template_groundedness1: ClassVar[str]  = (
    "### Instruction\n\n"
    "You are a world class expert designed to evaluate the groundedness of an assertion.\n"
    "You will be provided with an assertion and a context.\n"
    "Your task is to determine if the assertion is supported by the context.\n"
    "Follow the instructions below:\n"
    "A. If there is no context or no assertion or context is empty or assertion is empty, say 0.\n"
    "B. If the assertion is not supported by the context, say 0.\n"
    "C. If the assertion is partially supported by the context, say 1.\n"
    "D. If the assertion is fully supported by the context, say 2.\n"
    "You must provide a rating of 0, 1, or 2, nothing else.\n\n"

    "### Example\n"
    "Context:\n"
    "<packagedElement xmi:id=\"_18_5_2_8850270_1512474944659_986984_14977\" name=\"Feel Comfortable Temperature\" xmi:type=\"uml:UseCase\"/>\n"
    "<packagedElement xmi:id=\"_18_5_2_8850270_1509371139827_39907_14934\" name=\"Climate Control Unit\" xmi:type=\"uml:Class\"/>\n"
    "<packagedElement xmi:id=\"_18_5_2_8850270_1509371707856_518581_15000\" name=\"Energy Supply\" xmi:type=\"uml:Class\"/>\n"
    "<packagedElement xmi:id=\"_18_5_2_8850270_1509371701451_525129_14967\" name=\"Vehicle Occupant\" xmi:type=\"uml:Class\"/>\n\n"

    "Assertion:\n"
    "<packagedElement xmi:type=\"uml:Class\" xmi:id=\"_18_5_2_8850270_1509106922453_570967_14687\" name=\"Vehicle In Use\">\n"
    "  <useCase xmi:idref=\"_18_5_2_8850270_1512474944659_986984_14977\"/>\n"
    "  <ownedAttribute xmi:type=\"uml:Property\" xmi:id=\"_18_5_2_8850270_1509371130853_999744_14909\" visibility=\"public\" aggregation=\"composite\" type=\"_18_5_2_8850270_1509371139827_39907_14934\"/>\n"
    "  <ownedAttribute xmi:type=\"uml:Property\" xmi:id=\"_18_5_2_8850270_1509372050140_150945_15005\" visibility=\"private\" aggregation=\"composite\" type=\"_18_5_2_8850270_1509371707856_518581_15000\"/>\n"
    "  <ownedAttribute xmi:type=\"uml:Property\" xmi:id=\"_18_5_2_8850270_1509372058374_481543_15041\" visibility=\"private\" aggregation=\"composite\" type=\"_18_5_2_8850270_1509371701451_525129_14967\"/>\n"
    "</packagedElement>\n\n"

    "Analyzing Context and Response, the Groundedness score is 2\n\n"

    "### Context:\n<{context}>\n\n"
    "### Assertion:\n<{response}>\n\n"
    "Analyzing Context and Response, the Groundedness score is "
)
    
    template_groundedness2: ClassVar[str]  = (
    "As a specialist in assessing the strength of connections between statements and their given contexts, "
    "I will evaluate the level of support an assertion receives from the provided context. Follow these guidelines:\n\n"
    "* If the assertion is not supported or context is empty or assertion is empty, assign a score of 0.\n"
    "* If the assertion is partially supported, assign a score of 1.\n"
    "* If the assertion is fully supported, assign a score of 2.\n\n"
    "I will provide a rating of 0, 1, or 2, without any additional information.\n\n"

    "---\n**Context:**\n"
    "[<packagedElement xmi:id=\"_18_5_2_8850270_1512474944659_986984_14977\" name=\"Feel Comfortable Temperature\" xmi:type=\"uml:UseCase\"/>\n"
    "<packagedElement xmi:id=\"_18_5_2_8850270_1509371139827_39907_14934\" name=\"Climate Control Unit\" xmi:type=\"uml:Class\"/>\n"
    "<packagedElement xmi:id=\"_18_5_2_8850270_1509371707856_518581_15000\" name=\"Energy Supply\" xmi:type=\"uml:Class\"/>\n"
    "<packagedElement xmi:id=\"_18_5_2_8850270_1509371701451_525129_14967\" name=\"Vehicle Occupant\" xmi:type=\"uml:Class\"/>]\n\n"

    "**Assertion:**\n"
    "[<packagedElement xmi:type=\"uml:Class\" xmi:id=\"_18_5_2_8850270_1509106922453_570967_14687\" name=\"Vehicle In Use\">\n"
    "  <useCase xmi:idref=\"_18_5_2_8850270_1512474944659_986984_14977\"/>\n"
    "  <ownedAttribute xmi:type=\"uml:Property\" xmi:id=\"_18_5_2_8850270_1509371130853_999744_14909\" visibility=\"public\" aggregation=\"composite\" type=\"_18_5_2_8850270_1509371139827_39907_14934\"/>\n"
    "  <ownedAttribute xmi:type=\"uml:Property\" xmi:id=\"_18_5_2_8850270_1509372050140_150945_15005\" visibility=\"private\" aggregation=\"composite\" type=\"_18_5_2_8850270_1509371707856_518581_15000\"/>\n"
    "  <ownedAttribute xmi:type=\"uml:Property\" xmi:id=\"_18_5_2_8850270_1509372058374_481543_15041\" visibility=\"private\" aggregation=\"composite\" type=\"_18_5_2_8850270_1509371701451_525129_14967\"/>\n"
    "</packagedElement>]\n\n"

    "Do not explain.\n"
    "Based on the provided context and response, the Groundedness score is:"
)

    def to_extra_config(self) -> dict:
        return {
            "rubric_score_descriptions": self.rubric_score_descriptions,
            "template_accuracy1": self.template_accuracy1,
            "template_accuracy2": self.template_accuracy2,
            "template_relevance1": self.template_relevance1,
            "template_relevance2": self.template_relevance2,
            "template_groundedness1": self.template_groundedness1,
            "template_groundedness2": self.template_groundedness2
        }