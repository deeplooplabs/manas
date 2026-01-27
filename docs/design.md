# HiRAG-Haystack 设计文档

## 1. 概述

### 1.1 目标

使用 Haystack 框架完整复刻 HiRAG (Hierarchical Retrieval-Augmented Generation) 系统。HiRAG 是一种基于分层知识的检索增强生成方法，通过构建知识图谱并利用分层社区结构实现更精准的知识检索。

### 1.2 核心特性

- **分层知识结构**：通过 Leiden 算法构建多层级社区
- **多种检索模式**：支持 naive、hi_local、hi_global、hi_bridge、hi 五种模式
- **路径规划**：通过最短路径算法连接跨社区的关键实体
- **社区报告**：为每个社区生成结构化摘要

### 1.3 架构映射

| HiRAG 模块 | 功能描述 | Haystack 映射 |
|------------|----------|---------------|
| 文档分块 | 按token大小分割文档 | `DocumentSplitter` / `RecursiveSplitter` |
| 实体提取 | LLM提取实体和关系 | 自定义 `EntityExtractor` 组件 |
| 知识图谱 | NetworkX/Neo4j存储 | 自定义 `GraphDocumentStore` |
| 向量存储 | NanoVectorDB | `InMemoryDocumentStore` + 嵌入器 |
| 社区聚类 | Leiden算法聚类 | 自定义 `CommunityDetector` 组件 |
| 社区报告 | LLM生成摘要 | 自定义 `CommunityReportGenerator` 组件 |
| 分层检索 | 多模式检索 | 自定义 `HierarchicalRetriever` 组件 |

---

## 2. 系统架构

### 2.1 索引流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        HiRAG Indexing Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐     │
│  │ 原始文档  │ -> │ DocumentSplitter│ -> │  EntityExtractor   │     │
│  └──────────┘    └──────────────┘    └────────────────────┘     │
│                                            |                      │
│                         ┌─────────────────┼─────────────────┐   │
│                         v                 v                 v   │
│              ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│              │ GraphDocument│  │TextEmbedder  │  │Document  │  │
│              │    Store      │  │              │  │  Writer  │  │
│              └──────────────┘  └──────────────┘  └──────────┘  │
│                    |                 |                              │
│                    v                 v                              │
│         ┌──────────────────┐  ┌──────────────┐                   │
│         │CommunityDetector │  │ InMemory     │                   │
│         │                  │  │ DocumentStore│                   │
│         └──────────────────┘  └──────────────┘                   │
│                    |                                                │
│                    v                                                │
│         ┌──────────────────────┐                                  │
│         │CommunityReportGenerator│                                 │
│         └──────────────────────┘                                  │
│                    |                                                │
│                    v                                                │
│         ┌──────────────────┐                                      │
│         │  JSONKVStorage   │                                      │
│         └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 查询流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         HiRAG Query Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐     │
│  │  用户查询  │ -> │ TextEmbedder │ -> │ EntityRetriever     │     │
│  └──────────┘    └──────────────┘    └────────────────────┘     │
│                                            |                      │
│                                            v                      │
│                                   ┌──────────────┐               │
│                                   │    Router    │               │
│                                   │  (检索模式)   │               │
│                                   └──────┬───────┘               │
│                  ┌─────────────────────────┼─────────────────┐  │
│                  v                         v                 v  │
│         ┌───────────────┐      ┌───────────────┐   ┌──────────┐ │
│         │ LocalContext  │      │GlobalContext  │   │  Bridge  │ │
│         │    Builder    │      │    Builder    │   │  Builder  │ │
│         └───────┬───────┘      └───────┬───────┘   └────┬─────┘ │
│                 |                      |                |        │
│                  ┌──────────────────────┼────────────────┐       │
│                  v                      v                v       │
│         ┌─────────────────────────────────────────────────┐     │
│         │              HierarchicalContextBuilder           │     │
│         └─────────────────────────────────────────────────┘     │
│                              |                                  │
│                              v                                  │
│                     ┌──────────────┐                           │
│                     │ PromptBuilder│                           │
│                     └──────┬───────┘                           │
│                            |                                    │
│                            v                                    │
│                     ┌──────────────┐                           │
│                     │ ChatGenerator│                           │
│                     └──────┬───────┘                           │
│                            |                                    │
│                            v                                    │
│                     ┌──────────────┐                           │
│                     │   最终答案    │                           │
│                     └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 检索模式

| 模式 | 描述 | 组成部分 |
|------|------|----------|
| `naive` | 基础 RAG | 文档块向量检索 |
| `hi_local` | 本地知识 | 实体 + 关系 + 文档块 |
| `hi_global` | 全局知识 | 社区报告 + 文档块 |
| `hi_bridge` | 桥接知识 | 跨社区推理路径 |
| `hi` | 完整分层 | 以上所有组合 |

---

## 3. 核心组件设计

### 3.1 GraphDocumentStore

图数据库抽象层，支持多种后端实现。

```python
from haystack.document_stores import DocumentStore
from haystack.dataclasses import Document
from typing import Optional

class GraphDocumentStore(DocumentStore):
    """支持图操作的文档存储基类"""

    # ===== 节点操作 =====
    async def has_node(self, node_id: str) -> bool:
        """检查节点是否存在"""
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Optional[dict]:
        """获取节点数据"""
        raise NotImplementedError

    async def upsert_node(self, node_id: str, data: dict):
        """创建或更新节点"""
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        """获取节点度数"""
        raise NotImplementedError

    async def get_node_edges(self, node_id: str) -> list[tuple]:
        """获取节点所有边"""
        raise NotImplementedError

    # ===== 边操作 =====
    async def has_edge(self, src: str, tgt: str) -> bool:
        """检查边是否存在"""
        raise NotImplementedError

    async def get_edge(self, src: str, tgt: str) -> Optional[dict]:
        """获取边数据"""
        raise NotImplementedError

    async def upsert_edge(self, src: str, tgt: str, data: dict):
        """创建或更新边"""
        raise NotImplementedError

    async def edge_degree(self, src: str, tgt: str) -> int:
        """获取边度数"""
        raise NotImplementedError

    # ===== 社区操作 =====
    async def clustering(self, algorithm: str = "leiden"):
        """执行图聚类"""
        raise NotImplementedError

    async def community_schema(self) -> dict:
        """获取社区结构"""
        raise NotImplementedError

    # ===== 路径操作 =====
    async def shortest_path(self, src: str, tgt: str) -> list[str]:
        """计算最短路径"""
        raise NotImplementedError

    async def subgraph_edges(self, nodes: list[str]) -> list:
        """获取子图的所有边"""
        raise NotImplementedError
```

### 3.2 EntityExtractor

从文档中提取实体和关系的组件。

```python
from haystack import component

@component
class EntityExtractor:
    """实体和关系提取组件"""

    def __init__(
        self,
        llm: ChatGenerator,
        entity_types: list[str] = None,
        max_gleaning: int = 1,
        summary_max_tokens: int = 500
    ):
        self.llm = llm
        self.entity_types = entity_types or self._default_entity_types()
        self.max_gleaning = max_gleaning
        self.summary_max_tokens = summary_max_tokens

    @component.output_types(entities=list, relations=list)
    def run(self, documents: list[Document]) -> dict:
        """
        输入: 分块后的文档列表
        输出: entities (实体列表), relations (关系列表)
        """
        # 1. 构建提取提示词
        # 2. 调用 LLM 提取
        # 3. Gleaning 机制补充遗漏
        # 4. 实体描述摘要
        return {
            "entities": [...],
            "relations": [...]
        }
```

### 3.3 CommunityDetector

图聚类组件。

```python
@component
class CommunityDetector:
    """社区检测组件"""

    def __init__(
        self,
        algorithm: str = "leiden",
        max_cluster_size: int = 10,
        seed: int = 0xDEADBEEF
    ):
        self.algorithm = algorithm
        self.max_cluster_size = max_cluster_size
        self.seed = seed

    @component.output_types(communities=list)
    def run(self, graph_store: GraphDocumentStore) -> dict:
        """
        输入: 图存储
        输出: communities (社区列表，带层级信息)
        """
        # 1. 执行 Leiden 聚类
        # 2. 构建分层结构
        # 3. 返回社区信息
        return {"communities": [...]}
```

### 3.4 CommunityReportGenerator

为社区生成摘要报告。

```python
@component
class CommunityReportGenerator:
    """社区报告生成器"""

    def __init__(self, llm: ChatGenerator):
        self.llm = llm

    @component.output_types(reports=dict)
    def run(
        self,
        graph_store: GraphDocumentStore,
        communities: list
    ) -> dict:
        """
        输入: 图存储, 社区列表
        输出: reports (社区ID -> 报告内容的映射)
        """
        # 按层级从低到高处理
        # 子社区报告可作为父社区上下文
        return {"reports": {...}}
```

### 3.5 HierarchicalRetriever

分层检索器，支持多种检索模式。

```python
@component
class HierarchicalRetriever:
    """分层知识检索器"""

    def __init__(
        self,
        graph_store: GraphDocumentStore,
        entity_store: InMemoryDocumentStore,
        community_reports: dict,
        query_param: QueryParam
    ):
        self.graph_store = graph_store
        self.entity_store = entity_store
        self.community_reports = community_reports
        self.query_param = query_param

    @component.output_types(
        entities=list,
        relations=list,
        communities=list,
        text_units=list,
        reasoning_path=list
    )
    def run(self, query: str, mode: str = "hi") -> dict:
        """
        根据 mode 选择不同的检索策略
        """
        if mode == "naive":
            return self._naive_retrieve(query)
        elif mode == "hi_local":
            return self._local_retrieve(query)
        elif mode == "hi_global":
            return self._global_retrieve(query)
        elif mode == "hi_bridge":
            return self._bridge_retrieve(query)
        elif mode == "hi":
            return self._hierarchical_retrieve(query)
        else:
            raise ValueError(f"Unknown mode: {mode}")
```

---

## 4. 数据结构

### 4.1 实体 (Entity)

```python
@dataclass
class Entity:
    entity_name: str
    entity_type: str
    description: str
    source_id: str          # 来源文档块ID
    clusters: list[dict]    # 所属社区信息 [{"level": int, "cluster": str}]
    embedding: list[float] | None = None
```

### 4.2 关系 (Relation)

```python
@dataclass
class Relation:
    src_id: str             # 源实体
    tgt_id: str             # 目标实体
    weight: float           # 权重
    description: str
    source_id: str          # 来源文档块ID
    order: int = 1
```

### 4.3 社区 (Community)

```python
@dataclass
class Community:
    level: int              # 层级
    title: str              # 社区标题
    nodes: list[str]        # 包含的实体
    edges: list[tuple]      # 包含的边
    chunk_ids: list[str]    # 关联的文档块
    occurrence: float       # 出现频次
    sub_communities: list[str]  # 子社区ID
    report_string: str = ""     # 报告文本
    report_json: dict = None    # 报告结构化数据
```

### 4.4 查询参数 (QueryParam)

```python
@dataclass
class QueryParam:
    mode: Literal["hi", "hi_local", "hi_global", "hi_bridge", "naive"] = "hi"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2              # 最大社区层级
    top_k: int = 20             # 检索实体数量
    top_m: int = 10             # 每社区关键实体数量

    # Token 限制
    max_token_for_text_unit: int = 20000
    max_token_for_local_context: int = 20000
    max_token_for_bridge_knowledge: int = 12500
    max_token_for_community_report: int = 12500
    naive_max_token_for_text_unit: int = 10000

    community_single_one: bool = False
```

---

## 5. 路径规划算法

### 5.1 跨社区路径规划

```python
async def find_path_with_required_nodes(
    graph_store: GraphDocumentStore,
    source: str,
    target: str,
    required_nodes: list[str]
) -> list[str]:
    """
    查找经过所有必经节点的路径

    算法:
    1. 从 source 出发，找到到 required_nodes[0] 的最短路径
    2. 从 required_nodes[0] 出发，找到到 required_nodes[1] 的最短路径
    3. 重复直到最后一个必经节点
    4. 从最后一个必经节点到 target 的最短路径
    5. 合并所有路径

    使用 NetworkX 或 Neo4j 的 shortest_path 算法
    """
    final_path = []
    current_node = source

    # 遍历所有必经节点
    for next_node in required_nodes:
        sub_path = await graph_store.shortest_path(current_node, next_node)
        if final_path:
            final_path.extend(sub_path[1:])  # 避免重复添加当前节点
        else:
            final_path.extend(sub_path)
        current_node = next_node

    # 最后到目标节点
    sub_path = await graph_store.shortest_path(current_node, target)
    final_path.extend(sub_path[1:])

    return final_path
```

---

## 6. 提示词设计

### 6.1 实体提取提示词

```python
ENTITY_EXTRACTION_PROMPT = """
从文本中提取实体和关系。

实体类型: {entity_types}

输出格式:
("entity", "实体名称", "实体类型", "实体描述")
("relationship", "源实体", "目标实体", "关系描述", 权重)

输入文本:
{input_text}
"""
```

### 6.2 社区报告提示词

```python
COMMUNITY_REPORT_PROMPT = """
基于以下社区信息生成报告:

实体:
{entities}

关系:
{relations}

生成包含以下部分的报告:
- title: 社区标题
- summary: 社区摘要
- findings: 关键发现列表
"""
```

---

## 7. 项目结构

```
hirag-haystack/
├── hirag_haystack/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── graph.py           # 图数据结构定义
│   │   ├── community.py       # 社区数据结构
│   │   └── query_param.py     # 查询参数
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── base.py            # GraphDocumentStore 基类
│   │   ├── networkx_store.py  # NetworkX 实现
│   │   └── neo4j_store.py     # Neo4j 实现
│   ├── components/
│   │   ├── __init__.py
│   │   ├── entity_extractor.py
│   │   ├── community_detector.py
│   │   ├── report_generator.py
│   │   ├── hierarchical_retriever.py
│   │   └── path_finder.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── indexing.py        # 索引 pipeline
│   │   └── query.py           # 查询 pipeline
│   └── prompts.py             # LLM 提示词模板
├── examples/
│   ├── basic_usage.py
│   └── advanced_queries.py
├── tests/
├── docs/
│   ├── design.md              # 本文档
│   └── implementation.md      # 实施计划
├── pyproject.toml
└── README.md
```

---

## 8. 依赖

```toml
[project]
name = "hirag-haystack"
dependencies = [
    "haystack-ai>=2.6",
    "networkx",
    "python-louvain",          # community 检测
    "tiktoken",
    "sentence-transformers",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "coverage"]
neo4j = ["neo4j>=5.0"]
openai = ["openai>=1.0", "haystack-ai[openai]"]
```

---

## 9. 参考文献

1. HiRAG Paper: https://arxiv.org/abs/2503.10150
2. HiRAG GitHub: https://github.com/hhy-huang/HiRAG
3. Haystack Documentation: https://docs.haystack.deepset.ai/
