"""
内部知识检索子智能体配置模块

把 prompts.yml 中的 knowledge_base 配置与 Qdrant 检索工具组装成
DeepAgents 可识别的字典式子智能体。主智能体依据 description 判断
是否需要查询企业内部私有文档。
"""

from app.agent.prompts import sub_agents_content
from app.tools.kb_search_tools import list_knowledge_domains, search_internal_knowledge

# 知识库子智能体遵循“先列专题 -> 再检索证据”的顺序
knowledge_base_agent = {
    "name": sub_agents_content["knowledge_base"]["name"],
    "description": sub_agents_content["knowledge_base"]["description"],
    "system_prompt": sub_agents_content["knowledge_base"]["system_prompt"],
    "tools": [list_knowledge_domains, search_internal_knowledge],
}
