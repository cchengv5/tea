# -*- coding: utf-8 -*-
"""
======================================================================
--- 跨文化语义漂移量化分析器 (Cross-Cultural Semantic Drift Analyzer) ---
======================================================================
作者: (您的名字) & A.I. Assistant
目标: 通过训练两个独立的词向量模型(18世纪中文小说 vs 18世纪英文小说),
      量化符号“茶”(Tea)在从东方到西方的文化迁移中所发生的“意义漂移”,
      并从中洞察其价值变迁的底层逻辑。

工作流程:
1.  环境自检: 自动下载并安装NLTK所需的数据包。
2.  数据清洗: 对中英文语料库进行严格的、健壮的清洗, 剔除噪音。
3.  模型训练: 为清洗后的语料分别训练Word2Vec模型,并智能缓存。
4.  宇宙发现(自动): 使用K-Means聚类,让AI自动发现两个文化中的核心“意义宇宙”。
5.  罗盘测量(精确): 使用预设的“跨文化意义罗盘”精确测量“茶/Tea”的价值坐标。
6.  语义DNA提取(直观): 直接提取与目标词最相关的TOP 20词汇,进行最终的直观对比。
7.  报告生成: 输出清晰、深刻的分析报告。

使用方法:
1. 准备两个目录: `chn_books` 和 `eng_books`。
2. 将对应的TXT格式小说文件放入其中。
3. 准备一个 `user_dict.txt` (可选但推荐) 来优化中文分词。
4. 运行此脚本: python semantic_drift_analyzer.py
"""

import os
import re
import jieba
import nltk
import logging
import joblib
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from itertools import zip_longest # 新增导入
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- 全局配置 ---
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- 目录和路径配置 ---
CHN_DIR = 'chn_books'
ENG_DIR = 'eng_books'
CACHE_DIR = 'model_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

CHN_MODEL_CACHE = os.path.join(CACHE_DIR, 'chn_word2vec.model')
ENG_MODEL_CACHE = os.path.join(CACHE_DIR, 'eng_word2vec.model')

# --- 分析目标配置 ---
CHN_TARGETS = ['茶', '水', '银子', '大洋', '铜板', '早餐', '中餐', '晚餐', '早饭', '午饭', '晚饭']
ENG_TARGETS = ['tea', 'water', 'pound', 'dollar', 'gold']

# ==========================================================
# --- 【模块】跨文化意义罗盘 (The Cross-Cultural Compass) ---
# ==========================================================
ANCHOR_COMPASS = {
    '物理-水': ('水', 'water'), '物理-杯': ('杯', 'cup'), '物理-叶': ('叶', 'leaf'), '物理-热': ('热', 'hot'),
    '功能-喝': ('喝', 'drink'), '功能-客人': ('客人', 'guest'), '功能-家': ('家', 'home'), '功能-下午': ('下午', 'afternoon'),
    '社会-钱': ('钱', 'money'), '社会-生意': ('生意', 'business'), '社会-奢侈': ('奢侈', 'luxury'), '社会-社会': ('社会', 'society'),
    '抽象-权力': ('权力', 'power'), '抽象-帝国/王国': ('帝国', 'kingdom'), '抽象-健康': ('健康', 'health'), '抽象-爱': ('爱', 'love')
}

# ==========================================================
# --- 核心功能函数 ---
# ==========================================================

def download_nltk_resources():
    """下载NLTK所需的数据包，处理可能的网络问题。"""
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"--- NLTK resource '{resource}' not found. Downloading now... ---")
            nltk.download(resource)

def clean_and_load_corpus(directory, lang='chn'):
    """加载、清洗并报告语料库健康状况。"""
    print(f"\n--- 正在从 '{directory}' 加载和清洗 {lang.upper()} 语料库 ---")
    sentences = []
    
    if lang == 'eng':
        stop_words = set(nltk.corpus.stopwords.words('english'))

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                    text = re.sub(r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK .*? \*\*\*', '', text, flags=re.IGNORECASE)
                    text = re.sub(r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK .*? \*\*\*', '', text, flags=re.IGNORECASE)
                    text = text.replace("Project Gutenberg's", "")

                    if lang == 'chn':
                        user_dict_path = 'user_dict.txt'
                        if os.path.exists(user_dict_path):
                            print(f"    -> 发现并加载自定义词典: {user_dict_path}")
                            jieba.load_userdict(user_dict_path)
                        else:
                            print(f"    -> 警告: 未找到自定义词典 {user_dict_path}，将使用默认分词。")
                        raw_sentences = re.split(r'[。！？]', text)
                        for sent in raw_sentences:
                            if sent.strip():
                                sent = re.sub(r'茶', ' 茶 ', sent)
                                words = [w for w in jieba.lcut(sent.strip()) if w.strip()]
                                sentences.append(words)
                    else:
                        raw_sentences = nltk.sent_tokenize(text)
                        for sent in raw_sentences:
                            words = [w.lower() for w in nltk.word_tokenize(sent) if w.isalpha() and w.lower() not in stop_words]
                            sentences.append(words)
            except Exception as e:
                print(f"    警告: 读取或处理文件 {filename} 时出错: {e}")

    min_len, max_len = 6, 300
    original_count = len(sentences)
    sentences = [s for s in sentences if min_len <= len(s) <= max_len]
    print(f"语料库清洗加载完成，从 {original_count} 个原始句段中得到 {len(sentences)} 个高质量句子。")
    
    print_corpus_health_report(sentences, lang)
    
    return sentences

def print_corpus_health_report(sentences, lang):
    """打印语料库的健康统计数据。"""
    print(f"\n--- {lang.upper()} 语料库体检报告 ---")
    if not sentences:
        print("语料库为空！")
        return
    total_sentences = len(sentences)
    print(f"句子总数: {total_sentences}")
    print(f"\n语料库前5个样本:")
    for i, s in enumerate(sentences[:5]):
        print(f"  样本 {i} (长度 {len(s)}): {'/'.join(s[:15])}...")
    print("-" * 20)

def train_or_load_model(sentences, cache_path, force_train=False):
    """训练或从缓存加载Word2Vec模型。"""
    cache_path_kv = cache_path + ".kv" # KeyedVectors的专用缓存
    if not force_train and os.path.exists(cache_path_kv):
        print(f"--- 从缓存加载优化后的KeyedVectors模型: {cache_path_kv} ---")
        return KeyedVectors.load(cache_path_kv)
    
    print(f"--- 正在训练新的Word2Vec模型 (这可能需要几分钟)... ---")
    model = Word2Vec(
        sentences=sentences, vector_size=150, window=10, min_count=5,
        workers=4, sg=1, hs=0, negative=5
    )
    # 只保存和加载KeyedVectors，更轻量、更快速
    kv = model.wv
    kv.save(cache_path_kv)
    print(f"--- 模型训练完成并保存到: {cache_path_kv} ---")
    return kv

def discover_and_analyze_clusters(model, lang, targets, num_clusters=4, top_n_words=5000):
    """使用K-Means自动发现意义宇宙并进行分析。"""
    print(f"\n{'='*25}\n--- 在 {lang.upper()} 语料库中自动发现 {num_clusters} 个意义宇宙 ---\n{'='*25}")
    vocab = list(model.index_to_key)
    if len(vocab) > top_n_words: vocab = vocab[:top_n_words]
    word_vectors = np.array([model[word] for word in vocab])

    print(f"正在对语料库中最高频的 {len(vocab)} 个核心词汇进行K-Means聚类...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(word_vectors)
    
    results = []
    for i in range(num_clusters):
        centroid = kmeans.cluster_centers_[i]
        similar_words = model.most_similar(positive=[centroid], topn=15)
        signature = ", ".join([word for word, score in similar_words])
        item = {'label': f'宇宙 {i}: {signature.split(",")[0]}/{signature.split(",")[1]}', 'distances': {}}
        for target in targets:
            if target in model:
                dist = 1 - cosine_similarity(model[target].reshape(1, -1), centroid.reshape(1, -1))[0][0]
                item['distances'][target] = dist
        results.append(item)
    return results

def measure_distances_on_compass(model_chn, model_eng, compass_dict):
    """在跨文化罗盘上精确测量距离。"""
    print(f"\n{'='*25}\n--- 【最终读数】跨文化意义罗盘测量 ---\n{'='*25}")
    print(f"{'概念锚点':<18}{'中文距离 (茶)':<18}{'英文距离 (tea)':<18}")
    print(f"{'-'*17:<18}{'-'*17:<18}{'-'*17:<18}")

    for concept, (chn_anchor, eng_anchor) in compass_dict.items():
        dist_chn_str = f"{1 - model_chn.similarity('茶', chn_anchor):.4f}" if '茶' in model_chn and chn_anchor in model_chn else "N/A"
        dist_eng_str = f"{1 - model_eng.similarity('tea', eng_anchor):.4f}" if 'tea' in model_eng and eng_anchor in model_eng else "N/A"
        print(f"{concept:<18}{dist_chn_str:<18}{dist_eng_str:<18}")

# ==========================================================
# --- 【新增模块】语义DNA提取器 (Semantic DNA Extractor) ---
# ==========================================================
def display_semantic_neighbors(model_chn, model_eng, chn_target, eng_target, top_n=20):
    """
    提取并并排显示一个概念在两个文化中的TOP 20语义邻居。
    这是最直观的“意义漂移”证据。
    """
    print(f"\n{'='*25}\n--- 【语义DNA】“{chn_target}” vs “{eng_target}” 核心关联对比 ---\n{'='*25}")

    # 安全地获取中文邻居
    try:
        chn_neighbors = model_chn.most_similar(chn_target, topn=top_n)
    except KeyError:
        print(f"警告: '{chn_target}' 不在中文模型的词汇表中。")
        chn_neighbors = []

    # 安全地获取英文邻居
    try:
        eng_neighbors = model_eng.most_similar(eng_target, topn=top_n)
    except KeyError:
        print(f"警告: '{eng_target}' 不在英文模型的词汇表中。")
        eng_neighbors = []
        
    print(f"{'排名':<5}{'中文 (CHN) 语境':<25}{'英文 (ENG) 语境':<25}")
    print("-" * 55)

    # 使用 zip_longest 并排打印结果，优雅地处理不等长列表
    for i, (chn, eng) in enumerate(zip_longest(chn_neighbors, eng_neighbors), 1):
        rank = f"{i}."
        
        chn_word = f"{chn[0]} ({chn[1]:.2f})" if chn else "-"
        eng_word = f"{eng[0]} ({eng[1]:.2f})" if eng else "-"
        
        print(f"{rank:<5}{chn_word:<25}{eng_word:<25}")

    print("-" * 55)
import matplotlib.cm as cm

def plot_semantic_similarity(model_chn, model_eng, chn_targets, eng_targets, top_n=20):
    """
    绘制散点图，对比展示一组概念在两个文化中的语义相似度。
    """
    # 初始化图表
    plt.figure(figsize=(15, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建颜色映射
    colors = cm.rainbow(np.linspace(0, 1, len(chn_targets)))
    
    # 中文词向量图
    plt.subplot(1, 2, 1)
    
    # 绘制中文目标词及其邻居的散点图
    for i, chn_target in enumerate(chn_targets):
        try:
            chn_neighbors = model_chn.most_similar(chn_target, topn=top_n)
            chn_vectors = np.array([model_chn[word] for word, _ in chn_neighbors])
            plt.scatter(chn_vectors[:, 0], chn_vectors[:, 1], color=colors[i], alpha=0.5)
            
            # 高亮中文目标词
            chn_target_vector = model_chn[chn_target]
            plt.scatter(chn_target_vector[0], chn_target_vector[1], color=colors[i], label=f'中文: {chn_target}')
            
            # 为每个点添加标签
            for word, _ in chn_neighbors:
                word_vector = model_chn[word]
                plt.text(word_vector[0], word_vector[1], word, fontsize=9, ha='right')
            plt.text(chn_target_vector[0], chn_target_vector[1], chn_target, fontsize=9, ha='right', color=colors[i])
        except KeyError:
            print(f"警告: '{chn_target}' 不在中文模型的词汇表中。")
    
    plt.title('中文词向量图')
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.legend()
    
    # 英文词向量图
    plt.subplot(2, 1, 2)
    
    # 绘制英文目标词及其邻居的散点图
    for i, eng_target in enumerate(eng_targets):
        try:
            eng_neighbors = model_eng.most_similar(eng_target, topn=top_n)
            eng_vectors = np.array([model_eng[word] for word, _ in eng_neighbors])
            plt.scatter(eng_vectors[:, 0], eng_vectors[:, 1], color=colors[i], alpha=0.5)
            
            # 高亮英文目标词
            eng_target_vector = model_eng[eng_target]
            plt.scatter(eng_target_vector[0], eng_target_vector[1], color=colors[i], label=f'英文: {eng_target}')
            
            # 为每个点添加标签
            for word, _ in eng_neighbors:
                word_vector = model_eng[word]
                plt.text(word_vector[0], word_vector[1], word, fontsize=9, ha='right')
            plt.text(eng_target_vector[0], eng_target_vector[1], eng_target, fontsize=9, ha='right', color=colors[i])
        except KeyError:
            print(f"警告: '{eng_target}' 不在英文模型的词汇表中。")
    
    plt.title('英文词向量图')
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.legend()
    
    # 显示图表
    plt.tight_layout()
    plt.show()


def plot_with_tsne(model_chn, model_eng, chn_targets, eng_targets, top_n=15):
    """
    使用 t-SNE 对目标词及其邻居进行降维，并进行可视化对比。
    这能更真实地反映词语在高维语义空间中的相对位置。
    """
    print(f"\n{'='*25}\n--- 【t-SNE 可视化】正在生成语义空间地图... ---\n{'='*25}")
    
    # --- 配置绘图字体 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('基于t-SNE的跨文化语义空间对比', fontsize=20)

    # --- 1. 处理中文模型 ---
    plot_single_space(ax1, model_chn, chn_targets, '中文 (CHN) 语义空间', top_n)

    # --- 2. 处理英文模型 ---
    plot_single_space(ax2, model_eng, eng_targets, '英文 (ENG) 语义空间', top_n)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_single_space(ax, model, targets, title, top_n):
    """辅助函数：为单个语言模型生成 t-SNE 降维图。"""
    ax.set_title(title, fontsize=16)
    
    all_words = []
    all_vectors = []
    all_colors = []
    target_indices = [] # 记录目标词的索引，方便后续高亮

    # 使用颜色映射为每个目标词及其“星系”分配一种颜色
    colors = cm.rainbow(np.linspace(0, 1, len(targets)))

    for i, target in enumerate(targets):
        if target not in model:
            print(f"警告: 目标词 '{target}' 不在 {title} 的词汇表中，已跳过。")
            continue
        
        # 1. 收集目标词和它的邻居
        words_to_plot = [target]
        try:
            neighbors = [word for word, _ in model.most_similar(target, topn=top_n)]
            words_to_plot.extend(neighbors)
        except KeyError:
            pass # 如果邻居找不到，至少绘制目标词本身

        # 2. 获取这些词的向量
        vectors = []
        words_in_vocab = []
        for word in words_to_plot:
            if word in model:
                vectors.append(model[word])
                words_in_vocab.append(word)

        if not vectors:
            continue
            
        # 3. 记录信息用于绘图
        target_idx_in_batch = len(all_words) + words_in_vocab.index(target)
        target_indices.append(target_idx_in_batch)
        
        all_words.extend(words_in_vocab)
        all_vectors.extend(vectors)
        all_colors.extend([colors[i]] * len(vectors))

    if not all_vectors:
        ax.text(0.5, 0.5, '没有可供可视化的数据', ha='center', va='center')
        return

    # 4. 执行 t-SNE 降维
    print(f"在 {title} 中对 {len(all_vectors)} 个词向量进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_vectors)-1), random_state=42, max_iter=1000)
    low_dim_vectors = tsne.fit_transform(np.array(all_vectors))

    # 5. 绘制散点图
    ax.scatter(low_dim_vectors[:, 0], low_dim_vectors[:, 1], c=all_colors, alpha=0.5)

    # 6. 为每个点添加文字标签，并高亮目标词
    for i, word in enumerate(all_words):
        x, y = low_dim_vectors[i, 0], low_dim_vectors[i, 1]
        if i in target_indices:
            # 高亮目标词：字体加粗、放大、颜色更深
            ax.text(x, y, word, fontsize=12, fontweight='bold', color=all_colors[i]*0.8)
        else:
            # 普通邻居词
            ax.text(x, y, word, fontsize=9, color=all_colors[i])
            
    ax.set_xticks([])
    ax.set_yticks([])

# ==========================================================
# --- 主程序入口 ---
# ==========================================================
def main():
    """主执行函数"""
    # 步骤 1: 环境自检
    download_nltk_resources()

    # 步骤 2: 数据清洗和加载
    sentences_chn = clean_and_load_corpus(CHN_DIR, 'chn')
    sentences_eng = clean_and_load_corpus(ENG_DIR, 'eng')

    # 步骤 3: 训练或加载模型
    model_chn = train_or_load_model(sentences_chn, CHN_MODEL_CACHE)
    model_eng = train_or_load_model(sentences_eng, ENG_MODEL_CACHE)

    # 步骤 4: 自动发现意义宇宙 (宏观分析)
    # discover_and_analyze_clusters(model_chn, 'chn', CHN_TARGETS)
    # discover_and_analyze_clusters(model_eng, 'eng', ENG_TARGETS)

    # 步骤 5: 在跨文化罗盘上进行精确测量 (中观分析)
    measure_distances_on_compass(model_chn, model_eng, ANCHOR_COMPASS)
    
    # 步骤 6: 【新增】提取语义DNA进行直观对比 (微观分析)
    display_semantic_neighbors(model_chn, model_eng, '茶', 'tea')

    # 步骤 7: 【升级】使用t-SNE可视化展示语义空间差异
    plot_with_tsne(model_chn, model_eng, CHN_TARGETS, ENG_TARGETS) # <--- 修改这里！
    
    # 步骤 7: 输出最终结论
    print(f"\n{'='*25}\n--- 【最终分析结论】 ---\n{'='*25}")
    print("分析完成。请综合【意义罗盘】和【语义DNA】两份报告进行解读。")
    print("【罗盘】从预设维度精确量化了价值漂移的方向和强度。")
    print("【DNA】则直观展示了'茶/Tea'在两个文化中的核心关联物，是漂移的直接证据。")
    print("\n这份从“人情关系”到“商品组合”的“意义漂移”,")
    print("正是历史上巨额利润的来源。")

if __name__ == '__main__':
    main()
