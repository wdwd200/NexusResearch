"""
Markdown 文件生成工具

供主智能体把最终整理后的内容写入当前会话工作目录。工具会把模型传入的
filename/path 交给 resolve_path 统一解析，避免模型直接操作真实绝对路径。
"""

from pathlib import Path

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
from langchain_core.tools import tool

from app.api.context import get_session_context
from app.api.monitor import monitor
from app.utils.path_utils import PathEscapeError, resolve_path


@tool
def generate_markdown(
    content: Annotated[str, "要写入Markdown文档的文本内容"],
    filename: Annotated[str, "Markdown文档的文件名（不包含扩展名或包含.md）"],
    path: Annotated[str, "文件保存的绝对路径"] = "",
):
    """
    根据提供的文本内容生成 Markdown 文件

    :param content: 要写入 Markdown 文档的完整文本
    :param filename: 输出文件名，缺少 .md 后缀时会自动补全
    :param path: 可选保存路径；通常由运行时工作目录指令约束为相对路径
    :return: 文件生成结果说明
    """
    print(f"[MarkdownTool] 输入保存路径: {path or '当前会话目录'}")
    monitor.report_tool("Markdown文档生成工具", {"写入的文本内容": content})
    if not filename.endswith(".md"):
        filename += ".md"

    # session_dir 由 run_deep_agent 写入 ContextVar，保证文件写入当前会话工作目录
    session_dir = get_session_context()
    print(f"[MarkdownTool] 当前会话目录: {session_dir}")

    # 先把模型传入的 path/filename 合成一个逻辑路径，再交给 resolve_path 做统一清洗
    if path and path != ".":
        full_input_path = str(Path(path) / filename)
    else:
        full_input_path = filename
    try:
        full_path_str = resolve_path(full_input_path, session_dir)
    except PathEscapeError as exc:
        return f"生成Markdown文件失败: {exc}"
    file_path = Path(full_path_str)

    parent_dir = file_path.parent

    print(
        f"[MarkdownTool] Debug: parent_dir={parent_dir}, filename={filename}, full_path={file_path}"
    )

    try:
        if not session_dir:
            return "生成Markdown文件失败: 缺少会话目录，拒绝写入"
        # 只允许在当前会话目录内创建子目录
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
            print(f"[MarkdownTool] 已创建目录: {parent_dir}")

        # 幂等短路：同名文件已存在且内容一致时不再重写。
        # 部分模型（尤其是中转的 flash 版本）在拿到成功回执后仍会重复调用本工具，
        # 这里直接返回“已完成”并强制引导下一步，避免无意义的写盘和调用预算消耗。
        if file_path.exists():
            existing = file_path.read_text(encoding="utf-8")
            if existing == content:
                return (
                    f"该文件已存在且内容完全一致：{file_path.name}（{file_path.stat().st_size} 字节），"
                    "本次未重复写入。请不要再调用 generate_markdown。"
                    f"若用户需要 PDF，立刻调用 convert_md_to_pdf，参数 md_filename='{file_path.name}'。"
                )

        file_path.write_text(content, encoding="utf-8")

        print(f"[MarkdownTool] 文件写入完成: {file_path}")
        # 明确告知“已完成 + 下一步”，否则模型会误以为没写成而反复重写同一文件
        return (
            f"Markdown文件已成功生成并保存：{file_path.name}"
            f"（{file_path.stat().st_size} 字节）。"
            "内容写入已完成，不要重复调用本工具重写同一文件。"
            f"如果用户要求 PDF，请立刻调用 convert_md_to_pdf，参数 md_filename='{file_path.name}'。"
        )
    except Exception as e:
        print(f"[MarkdownTool] 文件写入失败: {e}")
        return f"生成Markdown文件失败: {e!s}"


if __name__ == "__main__":
    # 本地调试入口：直接运行本文件可验证 Markdown 写入和路径解析效果
    def get_session_context():
        return "./data/debug_docs"

    test_content = "# 测试文档\n这是 Markdown 生成工具的本地测试内容"
    test_filename = "测试文件"
    test_path = "sub_dir"

    print("===== 开始测试：Markdown 文件生成 =====")
    result = generate_markdown.invoke(
        {"content": test_content, "filename": test_filename, "path": test_path}
    )

    print(f"\n调用结果：{result}")
    if "已成功生成" in result:
        file_path = Path(result.split("'")[1])
        print(f"验证结果：文件 {file_path} {'存在' if file_path.exists() else '不存在'}")
