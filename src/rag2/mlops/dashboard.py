"""
RAG² MLOps: 仪表盘。

Streamlit 在 Python 3.14 上不可用，改用 FastAPI 内嵌 HTML 仪表盘。
此文件提供启动便利：启动 API 服务 + 打开浏览器。

用法：
  python -m rag2.mlops.dashboard --corpus arxiv_2026 --corpus-file data/arxiv_2026_corpus.json

仪表盘地址：http://localhost:8000/dashboard
API 文档：http://localhost:8000/docs
"""
import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RAG² MLOps Dashboard")
    parser.add_argument("--corpus", default="scifact")
    parser.add_argument("--corpus-file", default="")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    # 启动 API 服务
    cmd = [
        sys.executable, "-m", "rag2.mlops.api_server",
        "--corpus", args.corpus,
        "--corpus-file", args.corpus_file,
        "--port", str(args.port),
    ]
    env = {
        "PYTHONPATH": "src",
        "HF_HUB_OFFLINE": "1",
        "TQDM_DISABLE": "1",
        "PATH": __import__("os").environ.get("PATH", ""),
    }
    # 传递 API key
    for key in ("DMX_API_KEY", "KIMI_API_KEY", "QWEN_API_KEY"):
        if key in __import__("os").environ:
            env[key] = __import__("os").environ[key]

    url = f"http://localhost:{args.port}/dashboard"

    if not args.no_browser:
        # 延迟打开浏览器（等服务启动）
        def open_browser():
            time.sleep(10)
            webbrowser.open(url)
        import threading
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"Dashboard: {url}")
    print(f"API Docs:  http://localhost:{args.port}/docs")
    print(f"启动 API 服务...")

    proc = subprocess.Popen(cmd, env=env)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n已停止")


if __name__ == "__main__":
    main()
