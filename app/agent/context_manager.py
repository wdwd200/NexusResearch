"""
上下文窗口治理模块

把一次多轮工具调用的消息列表拆成四层来管理：
1. 稳定层：当前 system 指令、工作目录契约、最新一条用户请求，优先保留但会计入字符预算；
2. 当前层：滑动窗口内的近期对话，按字符预算从后往前保留；
3. 证据层：持久化在状态账本（checkpointer）中的完整历史，不进入本次请求；
4. 临时层：校验失败后的重试草稿，只在局部流转，不写入长期历史。

这里的预算是字符估算，不是模型 tokenizer 的严格 token 数。超预算时截断文本，不把请求裁成空窗口。

对外只暴露三个纯函数和一个中间件：
- budget_messages：按 max_chars 从后往前截取，返回新列表；
- pair_tool_messages：剥除窗口开头落单的 tool 消息，保证 assistant -> tool 严格配对；
- prepend_system：把最新 system 指令置顶；
- ContextWindowMiddleware：把以上逻辑挂到模型调用前。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import BaseMessage, SystemMessage

DEFAULT_MAX_CHARS = 24000


def _message_text(message: BaseMessage) -> str:
    """把消息内容统一转换为可计数的文本。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # 多模态内容块只统计文本部分，图片等按固定开销估算
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return str(content)


def _message_cost(message: BaseMessage) -> int:
    """估算单条消息的字符开销，工具调用参数也计入预算。"""
    cost = len(_message_text(message))
    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        cost += len(str(call.get("args", "")))
        cost += len(str(call.get("name", "")))
    return cost


_TOOL_RESULT_MAX_CHARS = 4000
_ASSISTANT_TYPES = {"assistant", "ai"}


def _message_type(message: BaseMessage) -> str:
    """LangChain 的 AIMessage.type 是 ai，部分序列化结果会写成 assistant。"""
    return str(getattr(message, "type", "") or "")


def _is_assistant(message: BaseMessage) -> bool:
    return _message_type(message) in _ASSISTANT_TYPES


def _tool_call_ids(message: BaseMessage) -> list[str]:
    ids: list[str] = []
    for call in getattr(message, "tool_calls", None) or []:
        call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        if call_id:
            ids.append(str(call_id))
    return ids


def _clip_tool_result(message: BaseMessage, max_chars: int = _TOOL_RESULT_MAX_CHARS) -> BaseMessage:
    """过大的工具返回只保留头尾摘要，避免单条结果撑爆窗口。"""
    if _message_type(message) != "tool":
        return message
    limit = max(1, min(_TOOL_RESULT_MAX_CHARS, max_chars))
    return _clip_to_budget(message, limit) if _message_cost(message) > limit else message


def _atomic_groups(messages: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    """
    把 assistant/ai 的 tool_calls 与紧随其后的 tool 结果收成不可拆组。

    system 与 human 各自独立成组，裁剪时只能整组丢弃，不能留下落单的 tool。
    """
    groups: list[list[BaseMessage]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if _is_assistant(message) and _tool_call_ids(message):
            expected = set(_tool_call_ids(message))
            group = [message]
            cursor = index + 1
            seen: set[str] = set()
            while cursor < len(messages) and _message_type(messages[cursor]) == "tool":
                tool_id = str(getattr(messages[cursor], "tool_call_id", "") or "")
                if tool_id not in expected or tool_id in seen:
                    break
                group.append(messages[cursor])
                seen.add(tool_id)
                cursor += 1
                if seen == expected:
                    break
            groups.append(group)
            index = cursor
            continue
        groups.append([message])
        index += 1
    return groups


_WORKSPACE_MARKER = "【工作环境指令】"


def _is_workspace_contract(message: BaseMessage) -> bool:
    return _message_type(message) in {"system", "human"} and _WORKSPACE_MARKER in _message_text(message)


def _split_workspace_contract(message: BaseMessage) -> tuple[BaseMessage, SystemMessage | None]:
    """把附在用户消息末尾的工作目录契约拆成稳定 system，避免被当成普通历史裁掉。"""
    if _message_type(message) != "human" or not _is_workspace_contract(message):
        return message, None
    text = _message_text(message)
    index = text.rfind(_WORKSPACE_MARKER)
    query = text[:index].rstrip()
    contract = SystemMessage(content=text[index:].strip())
    return message.model_copy(update={"content": query or text[:index]}), contract


def _latest_user_index(messages: Sequence[BaseMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if _message_type(messages[index]) == "human":
            return index
    return None


def _clip_to_budget(message: BaseMessage, max_chars: int) -> BaseMessage:
    """单条消息超过预算时截断文本，避免“至少保留一组”把窗口撑到无限大。"""
    if max_chars <= 0:
        return message.model_copy(update={"content": ""})
    text = _message_text(message)
    overhead = _message_cost(message) - len(text)
    allowance = max(0, max_chars - overhead)
    if len(text) <= allowance:
        return message
    marker = "\n...[消息已按预算截断]...\n"
    if allowance <= len(marker) + 2:
        return message.model_copy(update={"content": text[:allowance]})
    head = (allowance - len(marker)) // 2
    tail = allowance - len(marker) - head
    return message.model_copy(update={"content": f"{text[:head]}{marker}{text[-tail:]}"})


def _fit_group(group: Sequence[BaseMessage], max_chars: int) -> list[BaseMessage]:
    """超大组按成员截断，不拆掉 tool_call 或结果，避免配对后变成空窗口。"""
    fitted = list(group)
    if sum(_message_cost(item) for item in fitted) <= max_chars:
        return fitted
    largest = max(range(len(fitted)), key=lambda index: _message_cost(fitted[index]))
    others = sum(_message_cost(item) for index, item in enumerate(fitted) if index != largest)
    allowance = max(1, max_chars - others)
    fitted[largest] = _clip_to_budget(fitted[largest], allowance)
    overflow = sum(_message_cost(item) for item in fitted) - max_chars
    if overflow > 0:
        for index, item in enumerate(fitted):
            if index == largest or _message_type(item) == "tool":
                continue
            room = max(1, _message_cost(item) - overflow)
            fitted[index] = _clip_to_budget(item, room)
            overflow = sum(_message_cost(entry) for entry in fitted) - max_chars
            if overflow <= 0:
                break
    return fitted


def budget_messages(
    messages: Sequence[BaseMessage],
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[BaseMessage]:
    """
    按字符预算保留最近的若干完整消息组。

    最新用户请求优先保留。历史用户轮次超出预算会丢弃，不会无限保留。
    超大 tool 组截断文本，不拆成无法配对的碎片。
    """
    if not messages or max_chars <= 0:
        return []

    groups = [
        [_clip_tool_result(item, _TOOL_RESULT_MAX_CHARS) for item in group]
        for group in _atomic_groups(messages)
    ]
    pinned = _latest_user_index(messages)
    pinned_index = len(groups) - 1
    if pinned is not None:
        consumed = 0
        for index, group in enumerate(groups):
            if consumed <= pinned < consumed + len(group):
                pinned_index = index
                break
            consumed += len(group)

    chosen: set[int] = {pinned_index}
    groups[pinned_index] = _fit_group(groups[pinned_index], max_chars)
    pinned_cost = sum(_message_cost(item) for item in groups[pinned_index])
    used = min(pinned_cost, max_chars)

    # 先按时间倒序保留“最新用户请求之后”的工具组，再从历史向前补。
    # 早期实现只从 pinned_index 往前扫，导致最新用户消息之后的所有工具结果
    # （即刚刚发生的工具回执）全部被丢弃，模型会误以为操作失败而反复重试。
    for index in range(len(groups) - 1, pinned_index, -1):
        group = groups[index]
        cost = sum(_message_cost(item) for item in group)
        if used + cost > max_chars:
            continue
        chosen.add(index)
        used += cost

    for index in range(pinned_index - 1, -1, -1):
        group = groups[index]
        cost = sum(_message_cost(item) for item in group)
        if used + cost > max_chars and _is_assistant(group[0]) and _tool_call_ids(group[0]):
            group = _fit_group(group, max(1, max_chars - used))
            groups[index] = group
            cost = sum(_message_cost(item) for item in group)
        if used + cost > max_chars:
            break
        chosen.add(index)
        used += cost

    ordered: list[BaseMessage] = []
    for index, group in enumerate(groups):
        if index in chosen:
            ordered.extend(group)
    return ordered


def pair_tool_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """
    校验并修复工具调用配对。

    同时识别 type=ai 与 type=assistant。开头落单的 tool、缺少结果的 tool_calls、
    以及无法对上 tool_call_id 的结果都会被整组去掉，而不是只剥第一段 tool。
    system 与 human 消息原样保留。
    """
    repaired: list[BaseMessage] = []
    for group in _atomic_groups(messages):
        head = group[0]
        if _message_type(head) == "tool":
            continue
        if _is_assistant(head) and _tool_call_ids(head):
            expected = _tool_call_ids(head)
            returned = [
                str(getattr(item, "tool_call_id", "") or "")
                for item in group[1:]
                if _message_type(item) == "tool"
            ]
            # 按 id 集合和唯一性配对，不要求工具结果与调用顺序完全一致。
            # 缺结果或重复 id 的组直接丢弃，不补造工具输出。
            if len(returned) != len(set(returned)) or set(returned) != set(expected):
                continue
        if _is_assistant(head) and not _message_text(head) and not _tool_call_ids(head):
            continue
        repaired.extend(group)
    return repaired


def prepend_system(
    messages: Iterable[BaseMessage],
    system_prompt: str | None,
) -> list[BaseMessage]:
    """
    把最新的 system 指令置顶。

    历史中可能已经存在旧的 system 消息（例如上一轮注入的工作目录契约），
    这里统一移除，只保留当前这次调用应有的系统指令，避免规则叠加或互相冲突。
    """
    body = [m for m in messages if getattr(m, "type", "") != "system"]
    if not system_prompt:
        return body
    return [SystemMessage(content=system_prompt), *body]


def build_context_window(
    messages: Sequence[BaseMessage],
    system_prompt: str | None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[BaseMessage]:
    """
    按“裁剪 -> 修复配对 -> 置顶 system”的顺序构建最终请求消息列表。

    当前 system 和工作目录契约各保留一份，并与最新用户请求一起计入字符预算。
    历史用户轮次超出预算会丢弃。超预算时截断文本，不返回空窗口，也不承诺严格 token。
    """
    stable: list[BaseMessage] = []
    body: list[BaseMessage] = []
    for item in messages:
        if _message_type(item) == "system":
            continue
        query, contract = _split_workspace_contract(item)
        body.append(query)
        if contract is not None:
            stable = [contract]
    if system_prompt:
        stable = [SystemMessage(content=system_prompt), *stable]

    stable_cost = sum(_message_cost(item) for item in stable)
    if stable_cost > max_chars:
        room = max_chars
        bounded_stable: list[BaseMessage] = []
        for item in stable:
            allowance = max(1, room // max(1, len(stable) - len(bounded_stable)))
            bounded_stable.append(_clip_to_budget(item, allowance))
            room = max(0, max_chars - sum(_message_cost(entry) for entry in bounded_stable))
        return bounded_stable

    body_budget = max(1, max_chars - stable_cost)
    trimmed = budget_messages(body, max_chars=body_budget)
    paired = pair_tool_messages(trimmed)
    if not paired and body:
        paired = budget_messages(body[-1:], max_chars=max(1, max_chars - stable_cost))
    return [*stable, *paired]


class ContextWindowMiddleware(AgentMiddleware):
    """
    模型调用前的上下文治理钩子。

    必须在异步链路实现 awrap_model_call：DeepAgents 通过 astream 执行时不会回退到
    同步的 wrap_model_call，只实现同步版本会直接抛 NotImplementedError。
    """

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_model_calls: int | None = None,
        stop_after: str | None = None,
    ) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.max_model_calls = max_model_calls
        self.stop_after = stop_after
        self.model_calls = 0
        self._stop_tool_seen = False

    def _rewrite(self, request: ModelRequest) -> ModelRequest:
        # system_message 留在请求字段里，不重复放进 messages。它的字符开销先从预算扣除。
        system_text = _message_text(request.system_message) if request.system_message else ""
        contract_text = ""
        body_input: list[BaseMessage] = []
        for item in request.messages:
            query, contract = _split_workspace_contract(item)
            body_input.append(query)
            if contract is not None:
                contract_text = _message_text(contract)
        stable_cost = len(system_text) + len(contract_text)
        body_budget = max(1, self.max_chars - min(stable_cost, self.max_chars - 1))
        trimmed = budget_messages(body_input, max_chars=body_budget)
        body = pair_tool_messages(trimmed)
        if not body and body_input:
            body = budget_messages(body_input[-1:], max_chars=body_budget)
        system_message = request.system_message
        if contract_text:
            merged = f"{system_text}\n{contract_text}".strip()
            if len(merged) > self.max_chars:
                merged = merged[: self.max_chars]
            system_message = SystemMessage(content=merged)
        return request.override(messages=body, system_message=system_message)

    def _consume_call(self, request: ModelRequest) -> None:
        if self.stop_after and self._stop_tool_seen:
            raise RuntimeError(f"已完成 {self.stop_after} 后的下一次模型调用，停止继续调用模型")
        if self.stop_after:
            self._stop_tool_seen = any(
                _message_type(message) == "tool" and self.stop_after in _message_text(message)
                for message in request.messages
            )
        if self.max_model_calls is None:
            return
        self.model_calls += 1
        if self.model_calls > self.max_model_calls:
            raise RuntimeError(f"模型调用超过上限 {self.max_model_calls}")

    def wrap_model_call(self, request: ModelRequest, handler):
        self._consume_call(request)
        return handler(self._rewrite(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        self._consume_call(request)
        return await handler(self._rewrite(request))
