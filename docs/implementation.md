# HiRAG-Haystack 实施计划

## 当前进度

- ✅ **M1**: 基础设施完成
- ✅ **M2**: 索引流程完成
- ✅ **M3**: 检索流程完成
- 🔄 **M4**: 完整功能实现中

## 待实现功能 (按优先级排序)

| 优先级 | 功能 | 状态 | 描述 |
|--------|------|------|------|
| 🔴 P0 | 提示词模板 | ⏳ | 从原始 HiRAG 迁移所有提示词 |
| 🔴 P0 | 向量存储封装 | ⏳ | EntityVectorStore, ChunkVectorStore |
| 🔴 P0 | Token 工具函数 | ⏳ | truncate_by_token_size 等 |
| 🟠 P1 | 分层实体提取 | ⏳ | 使用聚类增强的提取 (HiRAG 核心特性) |
| 🟠 P1 | 增量更新 | ⏳ | 支持增量添加文档 |
| 🟠 P1 | hi_nobridge 模式 | ⏳ | 不带桥接的分层检索 |
| 🟡 P2 | PathFinder 组件 | ⏳ | 独立的路径规划组件 |
| 🟡 P2 | Neo4j 存储后端 | ⏳ | 大规模生产环境支持 |
| 🟢 P3 | 节点嵌入 | ⏳ | node2vec 算法 |
| 🟢 P3 | LLM 缓存 | ⏳ | 响应缓存机制 |

---

## 详细任务列表

### 已完成 ✅

#### 1. 基础设施
- [x] 项目结构搭建
- [x] pyproject.toml 配置
- [x] 核心数据结构 (Entity, Relation, Community, QueryParam)
- [x] GraphDocumentStore 基类

#### 2. 存储层
- [x] NetworkX 实现
- [x] 图持久化

#### 3. 组件
- [x] EntityExtractor (基础版)
- [x] CommunityDetector
- [x] CommunityReportGenerator
- [x] HierarchicalRetriever

#### 4. Pipeline
- [x] HiRAGIndexingPipeline
- [x] HiRAGQueryPipeline
- [x] 高层 API (HiRAG 类)

---

### 待实现 ⏳

#### P0: 提示词模板

- [ ] `hirag_haystack/prompts.py`
  - [ ] ENTITY_EXTRACTION_PROMPT
  - [ ] HI_ENTITY_EXTRACTION_PROMPT (分层版)
  - [ ] RELATION_EXTRACTION_PROMPT
  - [ ] HI_RELATION_EXTRACTION_PROMPT
  - [ ] COMMUNITY_REPORT_PROMPT
  - [ ] SUMMARIZE_ENTITY_PROMPT
  - [ ] CONTINUE_EXTRACTION_PROMPT
  - [ ] IF_LOOP_PROMPT
  - [ ] NAIVE_RAG_RESPONSE_PROMPT
  - [ ] LOCAL_RAG_RESPONSE_PROMPT

#### P0: 向量存储封装

- [ ] `stores/vector_store.py`
  - [ ] `EntityVectorStore` 类
    - [ ] 实体嵌入和检索
    - [ ] 元数据过滤 (entity_name)
  - [ ] `ChunkVectorStore` 类
    - [ ] 文档块嵌入和检索
  - [ ] 嵌入函数适配器

#### P0: Token 工具函数

- [ ] `utils/token_utils.py`
  - [ ] `encode_string_by_tiktoken()`
  - [ ] `decode_tokens_by_tiktoken()`
  - [ ] `truncate_list_by_token_size()`
  - [ ] `count_tokens()`

#### P1: 分层实体提取

- [ ] `components/hierarchical_entity_extractor.py`
  - [ ] 两阶段提取 (实体 -> 关系)
  - [ ] 实体嵌入
  - [ ] 分层聚类
  - [ ] 聚类后的实体合并

#### P1: 增量更新

- [ ] `pipelines/indexing.py` 增强
  - [ ] 文档去重 (MD5 hash)
  - [ ] 增量实体提取
  - [ ] 社区报告更新策略

#### P1: hi_nobridge 模式

- [ ] `components/hierarchical_retriever.py`
  - [ ] `_nobridge_retrieve()` 方法
  - [ ] 类似 hi_local 但包含社区报告

#### P2: PathFinder 组件

- [ ] `components/path_finder.py`
  - [ ] `find_path_with_required_nodes()`
  - [ ] 跨社区路径优化
  - [ ] 路径评分算法

#### P2: Neo4j 存储

- [ ] `stores/neo4j_store.py`
  - [ ] 连接管理
  - [ ] Cypher 查询实现
  - [ ] 节点/边 CRUD 操作

#### P3: 节点嵌入

- [ ] `stores/node_embedding.py`
  - [ ] node2vec 实现
  - [ ] 嵌入持久化

#### P3: LLM 缓存

- [ ] `stores/llm_cache_store.py`
  - [ ] 基于 prompt hash 的缓存
  - [ ] JSON 文件持久化

---

## 实施顺序

### 第 1 批 (P0 - 核心功能)
1. prompts.py - 提示词模板
2. utils/token_utils.py - Token 工具
3. stores/vector_store.py - 向量存储

### 第 2 批 (P1 - 重要特性)
4. components/hierarchical_entity_extractor.py
5. pipelines/indexing.py 增量更新
6. hi_nobridge 模式

### 第 3 批 (P2 - 扩展功能)
7. components/path_finder.py
8. stores/neo4j_store.py

### 第 4 批 (P3 - 优化功能)
9. stores/node_embedding.py
10. stores/llm_cache_store.py
