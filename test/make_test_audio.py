"""
生成测试 MP3（交互式小工具）

作用：运行后按提示交互式输入，用 edge-tts 合成中文 mp3，按用途分类输出：
      - 检索测试（音频提问）  -> mp3/query/
      - 导入测试（音频导入）  -> mp3/import/

依赖：edge-tts（联网合成）；未安装时脚本会提示。

用法：
    cd listenbook_rag
    uv run python test/make_test_audio.py
"""
import asyncio
import re
import sys
from pathlib import Path

# 保证可以直接 `python test/make_test_audio.py` 运行
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# 输出目录：与项目模块命名对应（import_process / query_process）
MP3_DIR = PROJECT_ROOT_DIR / "mp3"
QUERY_DIR = MP3_DIR / "query"
IMPORT_DIR = MP3_DIR / "import"

VOICE = "zh-CN-XiaoxiaoNeural"

# 各用途的默认文本（便于回车快速生成）
DEFAULT_QUERY_TEXT = "三体这本书的作者是谁？"
DEFAULT_IMPORT_TEXT = "三体是刘慈欣创作的科幻小说，讲述了地球文明与三体文明的信息交流，以及两个文明在宇宙中的兴衰历程。"


async def synthesize(text: str, out_path: Path) -> None:
    try:
        import edge_tts
    except ImportError:
        print("未安装 edge-tts，请先执行：uv pip install edge-tts")
        sys.exit(1)
    tts = edge_tts.Communicate(text, VOICE)
    await tts.save(str(out_path))


def clean_filename(name: str) -> str:
    """清洗文件名：去除 Windows 非法字符与空白，空则回退 sample"""
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", name.strip())
    return name or "sample"


def main():
    print("=" * 52)
    print("生成测试 MP3")
    print("=" * 52)

    # 1. 选择用途类型（检索 / 导入）
    print("\n用途类型：")
    print("  [1] 检索测试（音频提问）-> mp3/query/")
    print("  [2] 导入测试（音频导入）-> mp3/import/")
    choice = input("请选择（回车默认 1）：").strip() or "1"

    if choice == "2":
        target_dir = IMPORT_DIR
        kind = "导入测试"
        default_text = DEFAULT_IMPORT_TEXT
        test_cmd = "uv run python test/04_import_test.py"
    else:
        target_dir = QUERY_DIR
        kind = "检索测试"
        default_text = DEFAULT_QUERY_TEXT
        test_cmd = "uv run python test/06_asr_test.py"

    # 2. 输入文本
    print(f"\n[{kind}] 默认文本：{default_text}")
    text = input("请输入要合成的文本（回车用默认）：").strip() or default_text

    # 3. 输入文件名
    name = clean_filename(input("文件名（不含后缀，回车默认 sample）："))

    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / f"{name}.mp3"

    print(f"\n合成中... 文本：{text}")
    asyncio.run(synthesize(text, out))

    print(f"\n完成：{out}（{out.stat().st_size / 1024:.1f} KB）")
    print(f"测试命令：{test_cmd} \"{out}\"")


if __name__ == "__main__":
    main()
