from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List
import networkx as nx
import alibabacloud_oss_v2 as oss


def build_id_map(obj):
    """
    构建一个从 @id 到完整对象的映射字典
    """
    id_map = {}

    def recurse(o):
        if isinstance(o, dict):
            if "@id" in o:
                id_map[o["@id"]] = o
            for v in o.values():
                recurse(v)
        elif isinstance(o, list):
            for item in o:
                recurse(item)

    recurse(obj)
    return id_map


def resolve_refs(obj, id_map, visited=None):
    """递归替换所有 value 是 id 的字段为对应对象，避免死循环。"""
    if visited is None:
        visited = set()

    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(v, str) and v in id_map:
                if v in visited:
                    new_obj[k] = v  # 防止死循环
                else:
                    visited.add(v)
                    new_obj[k] = resolve_refs(id_map[v], id_map, visited.copy())
            else:
                new_obj[k] = resolve_refs(v, id_map, visited.copy())
        return new_obj

    elif isinstance(obj, list):
        return [resolve_refs(item, id_map, visited.copy()) for item in obj]

    else:
        return obj


def remove_ids(o):
    """
    递归地删除对象中的 @id 字段，返回一个新的对象。
    如果对象是列表，则返回一个新的列表。
    """
    if isinstance(o, dict):
        return {k: remove_ids(v) for k, v in o.items() if k != "@id"}
    elif isinstance(o, list):
        return [remove_ids(item) for item in o]
    else:
        return o
    


def dict_to_sorted_str(d):
    """
    将字典转换为按 key 排序的字符串，适合用于语义编码
    """
    if isinstance(d, dict):
        return "{" + ", ".join(f"{k}:{dict_to_sorted_str(v)}" for k, v in sorted(d.items())) + "}"
    elif isinstance(d, list):
        return "[" + ", ".join(dict_to_sorted_str(item) for item in d) + "]"
    else:
        return str(d)
    

def get_node_text(data: dict) -> str:
    """
    从节点数据中提取文本信息，忽略 @id 字段
    """

    res= ' '.join(
        f"{k}: {v}" for k, v in data.items()
        if k != '@id' and isinstance(v, (str, int, float)) and v
    )
    #print("get_node_text:", res)
    return res if res else "empty_node"


def compute_semantic_alignment(g1: nx.DiGraph, g2: nx.DiGraph, model, threshold=0.75):
    """
    返回 g1 中每个节点与 g2 中最相似的节点的映射
    """

    g1_nodes = list(g1.nodes(data=True))
    g2_nodes = list(g2.nodes(data=True))
    #print("g1_nodes:", len(g1_nodes), "g2_nodes:", len(g2_nodes))
    # print("g1.nodes:", g1.nodes)
    # print("list of g1 nodes:", g1_nodes)
    #print("g2_nodes:", g2_nodes)
    
    g1_vecs = model.encode([get_node_text(n) for k,n in g1_nodes], convert_to_tensor=True)
    g2_vecs = model.encode([get_node_text(n) for k,n in g2_nodes], convert_to_tensor=True)

    sim_matrix = util.cos_sim(g1_vecs, g2_vecs).cpu().numpy()
    #print("similarity matrix:", sim_matrix)

    mapping = {}
    for i, (n1,n1_attr) in enumerate(g1_nodes):
        j_best = np.argmax(sim_matrix[i])
        score = sim_matrix[i][j_best]
        if score >= threshold:
            mapping[n1] = g2_nodes[j_best][0]  # 使用 g2 的 @id 作为映射值
    return mapping


import re
import json
from typing import Set, Tuple, Optional
from difflib import SequenceMatcher
from nltk.stem import WordNetLemmatizer
from code.src.database.xmi_parser.base.namespace import NS
import xml.etree.ElementTree as ET
from lxml import etree

class XMIEvaluatorUtil:
    def __init__(self, synonym_path: Optional[str] = "code/src/config/synonyms.json"):
        self.lemmatizer = WordNetLemmatizer()
        self.synonym_dict = self._load_synonym_dict(synonym_path)

    def _load_synonym_dict(self, path: Optional[str]) -> dict:
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load synonym dictionary from {path}: {e}")
        # 默认同义词映射
        return {
            'thermal': 'temperature',
            'structure & mechanisms': 'structure and mechanisms',
            'structure_and_mechanisms': 'structure and mechanisms',
            'gn&c': 'gnc',
            'gnc': 'guidance navigation control',
            'comm': 'communications',
            'rf': 'radio frequency'
        }

    def _normalize(self, text: str) -> str:
        if not text:
            return ""

        text = text.lower()
        text = text.replace("&", "and")
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        words = text.split()
        normalized_words = []

        for word in words:
            lemma = self.lemmatizer.lemmatize(word)
            mapped = self.synonym_dict.get(lemma, lemma)
            normalized_words.append(mapped)

        return ' '.join(normalized_words)


    def _is_match(self, items1: Tuple, items2: Tuple,similarity_threshold=0.8) -> bool:
        """
        判断两个 元组各项是否匹配。
        基于阈值判断是否匹配。
        """
        # xmiEvaluatorUtil = XMIEvaluatorUtil()
        for item1, item2 in zip(items1, items2):
            item_single_1 = self._normalize(item1)
            item_single_2 = self._normalize(item2)
            # print("item1:", item1, "item2:", item2)
            # print("item_single_1:", item_single_1, "item_single_2:", item_single_2)
            temp_sim = SequenceMatcher(None, item_single_1, item_single_2).ratio()
            if temp_sim < similarity_threshold:
                return False
        return True

        # type1, name1 = map(self._normalize, item1)
        # type2, name2 = map(self._normalize, item2)
        # # print("item1:", item1, "item2:", item2)
        # # print("type1:", type1, "name1:", name1)
        # # print("type2:", type2, "name2:", name2)
        # type_sim = SequenceMatcher(None, type1, type2).ratio()
        # name_sim = SequenceMatcher(None, name1, name2).ratio()

        # return type_sim >= similarity_threshold and name_sim >= similarity_threshold
    

    def build_id_map(self, xmi_str: str) -> dict:
        """
        从 XMI 字符串中构建一个从 @id 到完整对象的映射字典
        """
        id_map = {}
        # 先构造一个 XML 树
        try:
            parser = ET.XMLParser(recover=True)
            root = ET.fromstring(f"<root>{xmi_str}</root>", parser=parser)
        except ET.ParseError as e:
            print(f"[ERROR] Invalid XMI content: {e}")
            return id_map
        XMI=NS['xmi']
        XMI_ID=f"{{{XMI}}}id"
        # 遍历 XML 树，提取 
        for elem in root.iter():
            if XMI_ID in elem.attrib:
                id_map[elem.attrib[XMI_ID]] = elem
        return root,id_map

    def _xmi_to_graph(self, xmi_str: str) -> nx.DiGraph:
        """
        Converts XMI content to a NetworkX directed graph.
        """
        ID_LIKE_PATTERN = re.compile(r"(^|[#])_[a-zA-Z0-9_]+$")
        G = nx.DiGraph()
        try:
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(f"<root>{xmi_str}</root>", parser=parser)
            id_element_map={}

            for elem in root.iter():
                xmi_id= elem.get('{%s}id' % NS['xmi']) or elem.get('xmi:id')
                xmi_id= xmi_id.strip() if isinstance(xmi_id, str) else xmi_id
                if xmi_id:
                    id_element_map[xmi_id] = elem
                else:
                    continue



            for elem in root.iter():
                tag_name = elem.tag.split('}')[-1]
                # print("tag:", tag_name)

                xmi_id= elem.get('{%s}id' % NS['xmi']) or elem.get('xmi:id')
                xmi_id= xmi_id.strip() if isinstance(xmi_id, str) else xmi_id
                if not xmi_id:
                    continue
                node_attrs = {}
                for k, v in elem.attrib.items():
                    if k in ['xmi:id', '{%s}id' % NS['xmi']]:
                        continue
                    # elif v.strip() in id_element_map:
                    #     continue
                    elif v and ID_LIKE_PATTERN.match(v) and v.strip() in id_element_map:
                        continue
                    # elif k.endswith('id') or k.endswith('ref'):
                    #     continue
                    node_attrs[k] = v
                node_attrs['tag'] = tag_name  # 添加标签名作为属性
                G.add_node(xmi_id, **node_attrs)

            # 添加边
            for elem in root.iter():
                xmi_id = elem.get('{%s}id' % NS['xmi']) or elem.get('xmi:id')
                if not xmi_id:
                    continue

                for k, v in elem.attrib.items():
                    if k in ['xmi:id', '{%s}id' % NS['xmi']]:
                        continue

                    v= v.strip() if isinstance(v, str) else v
                    if " " in v:
                        v_list= v.split()
                        for v_item in v_list:
                            if v_item.strip() in id_element_map:
                                G.add_edge(xmi_id, v_item, label=k)
                    elif v.strip() in id_element_map:
                        G.add_edge(xmi_id, v, label=k)
                
                for child in elem:
                    child_id = child.get('{%s}id' % NS['xmi']) or child.get('xmi:id')
                    if child_id and child_id in id_element_map:
                        tag_name = child.tag.split('}')[-1]  # 获取标签名
                        # print("tag:", tag_name)

                        G.add_edge(xmi_id, child_id, label=tag_name)
        except etree.XMLSyntaxError as e:
            print(f"[ERROR] Invalid XMI content: {e}")
        
        return G


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # 不早停
        else:
            self.counter += 1
            return self.counter >= self.patience
        


class TrainLossCallback:
    def __init__(self, patience, model, output_path):
        self.patience = patience
        self.best_loss = float('inf')
        self.no_improve_epochs = 0
        self.model = model
        self.output_path = output_path

    def on_epoch_end(self, epoch, logs=None):
        avg_loss = logs.get('loss') if logs else None
        if avg_loss is not None:
            print(f"📉 Epoch {epoch+1} 平均训练损失: {avg_loss:.4f}")

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.no_improve_epochs = 0
                print(f"✅ Loss 降低，保存模型（{self.output_path}）")
                self.model.save(self.output_path)
            else:
                self.no_improve_epochs += 1
                print(f"⚠️ Loss 未降低，连续 {self.no_improve_epochs} 轮")

            if self.no_improve_epochs >= self.patience:
                print(f"⛔️ 连续 {self.patience} 轮 loss 无下降，提前停止训练")
                return True  # 触发 early stop

        return False  # 继续训练

from sentence_transformers import losses
from torch.utils.tensorboard import SummaryWriter

class MyLoss(losses.MultipleNegativesRankingLoss):
    def __init__(self, model, writer: SummaryWriter):
        super().__init__(model)
        self.writer = writer
        self.global_step = 0

    def forward(self, sentence_embeddings, labels=None):
        loss_value = super().forward(sentence_embeddings, labels)
        self.writer.add_scalar("Loss/train", loss_value.item(), self.global_step)
        # print(f"[TensorBoard] Step {self.global_step}, Loss: {loss_value.item():.6f}")
        self.global_step += 1
        return loss_value
    


# oss_uploader.py
import os
from typing import Optional
import alibabacloud_oss_v2 as oss
from code.src.config import config


class OSSUtil:
    def __init__(self, region=None, bucket=None, endpoint: Optional[str] = None):
        # 初始化凭证（默认从环境变量读取）
        credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()

        # 加载默认配置
        cfg = oss.config.load_default()
        cfg.credentials_provider = credentials_provider
        cfg.region = region or config.OSS_REGION
        if endpoint:
            cfg.endpoint = endpoint

        # 创建客户端
        self.client = oss.Client(cfg)
        self.bucket = bucket or config.OSS_BUCKET

    def upload_string(self, object_key: str, content: str):
        data = content.encode('utf-8')
        return self._put_object(object_key, data)

    def upload_file(self, object_key: str, local_file_path: str):
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"文件不存在: {local_file_path}")

        with open(local_file_path, 'rb') as f:
            data = f.read()

        return self._put_object(object_key, data)

    def _put_object(self, object_key: str, data: bytes):
        result = self.client.put_object(oss.PutObjectRequest(
            bucket=self.bucket,
            key=object_key,
            body=data,
        ))
        print(f"✅ 上传成功: {object_key}")
        print(f"  📦 状态码: {result.status_code}")
        print(f"  📄 请求 ID: {result.request_id}")
        print(f"  🔐 ETag: {result.etag}")
        print(f"  💡 CRC64: {result.hash_crc64}")
        return result