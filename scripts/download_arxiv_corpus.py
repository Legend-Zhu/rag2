"""下载 arXiv 2026 最新论文，建"模型不知道的语料"语料库。

模型训练截止 ~2025 年中。2026 年的论文模型不可能知道。
这是验证 RAG² "答得更准"的核心语料。

下载 cs.CL + cs.AI 最新论文，取 title + abstract。
"""
import sys, time, json, re
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

NS = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

def fetch_arxiv(query, start=0, max_results=100):
    """从 arXiv API 拉取论文。"""
    params = {
        'search_query': query,
        'start': start,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending',
    }
    url = f"https://export.arxiv.org/api/query?{urlencode(params)}"
    req = Request(url, headers={'User-Agent': 'RAG2-Research/1.0'})
    with urlopen(req, timeout=30) as resp:
        data = resp.read().decode('utf-8')
    return data

def parse_arxiv(xml_str):
    """解析 arXiv XML 返回论文列表。"""
    root = ET.fromstring(xml_str)
    papers = []
    for entry in root.findall('atom:entry', NS):
        arxiv_id = entry.find('atom:id', NS).text.strip()
        # 提取纯 ID（去掉版本号 v1）
        pid = arxiv_id.split('/abs/')[-1].split('v')[0]
        title = entry.find('atom:title', NS).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', NS).text.strip()
        published = entry.find('atom:published', NS).text.strip()
        categories = [c.get('term') for c in entry.findall('atom:category', NS)]
        authors = [a.find('atom:name', NS).text for a in entry.findall('atom:author', NS)]

        papers.append({
            'arxiv_id': pid,
            'title': title,
            'text': summary,
            'published': published,
            'categories': categories,
            'authors': authors[:5],  # 最多5个作者
        })
    return papers

def main():
    all_papers = {}
    categories = ['cs.CL', 'cs.AI']
    per_batch = 100

    for cat in categories:
        print(f'下载 {cat}...', flush=True)
        for start in range(0, 300, per_batch):
            try:
                xml = fetch_arxiv(f'cat:{cat}', start=start, max_results=per_batch)
                papers = parse_arxiv(xml)
                new = 0
                for p in papers:
                    if p['arxiv_id'] not in all_papers:
                        all_papers[p['arxiv_id']] = p
                        new += 1
                print(f'  {cat} batch {start}: +{new} papers (total {len(all_papers)})', flush=True)
                time.sleep(3)  # arXiv 限流 3s/请求
            except Exception as e:
                print(f'  {cat} batch {start} 失败: {e}', flush=True)
                time.sleep(5)

    # 按日期排序，保留 2026 年的
    papers_2026 = [p for p in all_papers.values() if p['published'].startswith('2026')]
    papers_2026.sort(key=lambda x: x['published'], reverse=True)

    print(f'\n总计: {len(all_papers)} 篇, 2026 年: {len(papers_2026)} 篇', flush=True)
    if papers_2026:
        print(f'  日期范围: {papers_2026[-1]["published"][:10]} ~ {papers_2026[0]["published"][:10]}', flush=True)

    # 保存
    out = Path('data/arxiv_2026_corpus.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus = {p['arxiv_id']: {'title': p['title'], 'text': p['text'], 'published': p['published']} for p in papers_2026}
    out.write_text(json.dumps(corpus, ensure_ascii=False, indent=2))
    print(f'  保存: {out} ({len(corpus)} 篇, {out.stat().st_size/1024:.0f}KB)', flush=True)

    # 打印前5篇预览
    print('\n前 5 篇预览:', flush=True)
    for p in papers_2026[:5]:
        print(f'  [{p["published"][:10]}] {p["title"][:80]}', flush=True)
        print(f'    {p["text"][:120]}...', flush=True)

if __name__ == '__main__':
    main()
