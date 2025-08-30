import jsonschema
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import numpy as np
from code.src.evaluators.semantic_evaluator.semantic_evaluator import XMISemanticEvaluator,JSONSemanticEvaluator
from code.src.evaluators.syntax_evaluator.syntax_validator import JsonSyntaxValidator,XMISyntaxValidator
from code.src.evaluators.structure_evaluator.structure_evaluator import StructureEvaluator,JSONStructureEvaluator,XMIStructureEvaluator


class Evaluator:
    def __init__(self, schema, embedding_model=None,structure_match_strategy: str = "ged", semantic_threshold: float = 0.75):
        self.schema = schema
        self.structure_match_strategy = structure_match_strategy
        self.semantic_threshold = semantic_threshold
        self.embedding_model = embedding_model or SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.syntax_validator=None
        self.semantic_evaluator=None
        self.structure_evaluator=None


    def validate_syntax(self, content: dict) -> bool:
        # syntax_validator = SyntaxValidator(self.schema)
        return self.syntax_validator.validate_syntax(content)

    def semantic_score(self, content_generated: dict, content_standard: dict) -> float:
        # semantic_evaluator = SemanticEvaluator(self.embedding_model)
        return self.semantic_evaluator.semantic_score(content_generated, content_standard)

    def structure_score(self, content_generated: dict, content_standard: dict) -> float:
        # structure_evaluator=StructureEvaluator(embedded_model=self.embedding_model,mode=self.structure_match_strategy)
        return self.structure_evaluator.evaluate(content_generated,content_standard)


    def evaluate(self, content_generated: dict, content_standard: dict) -> dict:
        syntax_ok = self.validate_syntax(content_generated)
        semantic = self.semantic_score(content_generated, content_standard)
        structure = self.structure_score(content_generated, content_standard)
        return {
            "syntax": syntax_ok,
            "semantic": semantic,
            "structure": structure,
        }


class JsonEvaluator(Evaluator):
    def __init__(self, schema: dict, embedding_model=None,structure_match_strategy: str = "ged", semantic_threshold: float = 0.75):
        super().__init__(schema, embedding_model, structure_match_strategy, semantic_threshold)
        self.syntax_validator = JsonSyntaxValidator(self.schema)
        self.semantic_evaluator = JSONSemanticEvaluator(self.embedding_model)
        self.structure_evaluator = JSONStructureEvaluator(embedded_model=self.embedding_model, mode=self.structure_match_strategy)




class XMIEvaluator(Evaluator):
    def __init__(self, schema: dict, embedding_model=None,structure_match_strategy: str = "ged", semantic_threshold: float = 0.75):
        super().__init__(schema, embedding_model, structure_match_strategy, semantic_threshold)
        self.syntax_validator = XMISyntaxValidator(self.schema)
        self.semantic_evaluator = XMISemanticEvaluator(self.embedding_model)
        self.structure_evaluator = XMIStructureEvaluator(embedded_model=self.embedding_model, mode=self.structure_match_strategy)
        
