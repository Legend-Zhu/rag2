"""
RAG² 方法层统一接口

所有方法（baseline 与 W2 三支柱）必须实现 Method 抽象基类，
返回统一 Result schema，供评测层无差别消费。

关键设计：trace 字段记录方法内部过程（工具调用/索引命中/校验步骤），
供论文 appendix 与错误分析用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rag2.gateway import ModelGateway


@dataclass
class RetrievedDoc:
    """单条检索结果。"""
    title: str
    text: str
    score: float = 0.0           # 相关性分数（检索/重排后）
    is_supporting: bool = False  # 是否 gold 支撑文档（评测用，方法自身不知）


@dataclass
class Result:
    """方法输出的统一 schema。"""
    answer: str                          # 最终答案
    retrieved_docs: list[RetrievedDoc] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)   # claim 级信息（C3 用）
    citations: list[dict] = field(default_factory=list)  # claim→source 映射（C3 用）
    trace: dict = field(default_factory=dict)   # 内部过程记录（工具调用/索引命中等）
    raw_response: Any = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # RetrievedDoc 转 dict
        d["retrieved_docs"] = [asdict(r) if isinstance(r, RetrievedDoc) else r
                               for r in self.retrieved_docs]
        return d


class Method(ABC):
    """所有方法的抽象基类。"""

    name: str = "base"

    def __init__(self, gateway: ModelGateway, role: str = "generator"):
        self.gw = gateway
        self.role = role

    @abstractmethod
    def run(self, sample: dict) -> Result:
        """
        处理单个样本。

        Args:
            sample: DataLoader 产出的内部 schema
                    {id, question, answer, supporting_docs, metadata}
        Returns:
            Result
        """
        ...

    def run_batch(self, samples: list[dict]) -> list[Result]:
        """批量跑（默认串行，子类可改并发）。"""
        return [self.run(s) for s in samples]
