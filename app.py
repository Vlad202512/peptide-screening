import streamlit as st
import pandas as pd
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import requests
import os
import pickle
import hashlib

# ===== 页面设置 =====
st.set_page_config(page_title="多肽虚拟筛选工具", layout="wide")
st.title("多肽虚拟筛选工具")

# ===== 侧边栏参数 =====
st.sidebar.header("筛选参数")

drug = st.sidebar.text_input("靶点序列（参照肽）", value="HAEGTFTSDV")

min_len = st.sidebar.number_input("最小长度", value=10, min_value=5, max_value=50)
max_len = st.sidebar.number_input("最大长度", value=12, min_value=5, max_value=50)

cut_rule = st.sidebar.selectbox(
    "酶切规则",
    options=["KR (胰蛋白酶)", "FL (胃蛋白酶)", "FYWM (胰凝乳蛋白酶)", "KRFL (混合酶)"],
    index=0,
)
cut_map = {
    "KR (胰蛋白酶)": "KR",
    "FL (胃蛋白酶)": "FL",
    "FYWM (胰凝乳蛋白酶)": "FYWM",
    "KRFL (混合酶)": "KRFL",
}
CUT = cut_map[cut_rule]

top_n = st.sidebar.number_input("输出Top几", value=10, min_value=5, max_value=100)

content_threshold = st.sidebar.number_input("蛋白含量阈值(%)", value=10, min_value=0, max_value=100)

# ===== 生成缓存文件名（根据当前参数自动切换） =====
def get_cache_key():
    """根据靶点+酶切规则+长度范围生成唯一标识"""
    raw = f"{drug}_{CUT}_{min_len}_{max_len}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]

# ===== 核心函数 =====
@st.cache_resource
def load_model():
    with st.spinner("正在加载ProtBERT模型..."):
        model_name = "Rostlab/prot_bert"
        tok = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        mod = BertModel.from_pretrained(model_name, local_files_only=True)
    return tok, mod


def cut(seq, cut_aa, min_l, max_l):
    out = []
    now = ""
    for i, aa in enumerate(seq):
        now += aa
        if aa in cut_aa:
            if i + 1 >= len(seq) or seq[i + 1] != "P":
                if min_l <= len(now) <= max_l:
                    out.append(now)
                now = ""
    if now and min_l <= len(now) <= max_l:
        out.append(now)
    return out


def vec(p, tok, mod):
    s = " ".join(list(p))
    x = tok(s, return_tensors="pt")
    with torch.no_grad():
        y = mod(**x)
    return y.last_hidden_state.mean(dim=1).numpy().flatten()


def sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@st.cache_data
def fetch_proteins():
    SPECIES_IDS = [
        3847, 9913, 9031, 8090, 7955, 8030, 3888, 4565, 3818, 9823, 3994, 4497
    ]
    food_dict = {}
    for sid in SPECIES_IDS:
        url = f"https://rest.uniprot.org/uniprotkb/search?query=organism_id:{sid}&size=100&format=fasta"
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                lines = response.text.strip().split("\n")
                current_name = ""
                current_seq = ""
                for line in lines:
                    if line.startswith(">"):
                        if current_seq and len(current_seq) >= 50:
                            food_dict[current_name] = current_seq
                        parts = line[1:].split("|")
                        entry_name = (
                            parts[1].strip()
                            if len(parts) >= 2
                            else line[1:].strip()[:30]
                        )
                        species = ""
                        desc = parts[2].strip() if len(parts) >= 3 else ""
                        if "OS=" in desc:
                            species = desc.split("OS=")[-1].split()[0]
                            current_name = f"{entry_name} ({species})"
                        else:
                            current_name = entry_name
                        current_seq = ""
                    else:
                        current_seq += line.strip()
                if current_seq and len(current_seq) >= 50:
                    food_dict[current_name] = current_seq
        except Exception:
            pass
    return food_dict


PROTEIN_CONTENT = {
    "glycine max": 38, "glycine": 38,
    "bos taurus": 25, "bos": 25,
    "gallus gallus": 22, "gallus": 22,
    "oryzias latipes": 18, "oryzias": 18,
    "danio rerio": 18, "danio": 18,
    "salmo salar": 22, "salmo": 22,
    "parambassis": 16,
    "pisum sativum": 23, "pisum": 23,
    "triticum aestivum": 75, "triticum": 75,
    "arachis hypogaea": 28, "arachis": 28,
    "sus scrofa": 24, "sus": 24,
    "oryza sativa": 7, "oryza": 7,
    "avena sativa": 13, "avena": 13,
}


def get_content(src_name):
    src_lower = src_name.lower()
    if "(" in src_name and ")" in src_name:
        species_short = src_name.split("(")[-1].split(")")[0].strip().lower()
        for key, content in PROTEIN_CONTENT.items():
            if key.lower() == species_short:
                return content
    for key, content in PROTEIN_CONTENT.items():
        if key.lower() in src_lower:
            return content
    return 0


# ===== 主界面 =====
st.subheader("开始筛选")

if st.button("开始筛选", type="primary"):
    # 加载模型
    tok, mod = load_model()

    # 拉取蛋白数据（优先读本地缓存）
    food_cache_file = "food_cache.pkl"
    if os.path.exists(food_cache_file):
        with open(food_cache_file, "rb") as f:
            FOOD = pickle.load(f)
        st.success(f"已从缓存加载 {len(FOOD)} 条蛋白序列")
    else:
        with st.spinner("正在从UniProt下载食用蛋白序列..."):
            FOOD = fetch_proteins()
            with open(food_cache_file, "wb") as f:
                pickle.dump(FOOD, f)
        st.success(f"已获取并缓存 {len(FOOD)} 条蛋白序列")

    # 虚拟酶切
    allp = []
    for name, seq in FOOD.items():
        for p in cut(seq, CUT, min_len, max_len):
            allp.append({"src": name, "p": p})

    # 去重（按物种简称 + 肽段序列）
    seen = set()
    unip = []
    for x in allp:
        species_short = (
            x["src"].split("(")[-1].split(")")[0].strip()
            if "(" in x["src"]
            else x["src"]
        )
        key = (species_short, x["p"])
        if key not in seen:
            seen.add(key)
            unip.append(x)
    st.info(f"酶切并去重后共 {len(unip)} 条候选肽")

    # AI打分（智能缓存：根据当前参数自动切换缓存文件）
    cache_key = get_cache_key()
    score_cache_file = f"score_cache_{cache_key}.pkl"

    if os.path.exists(score_cache_file):
        with open(score_cache_file, "rb") as f:
            res = pickle.load(f)
        st.success(f"已从缓存加载 {len(res)} 条打分结果")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        v_drug = vec(drug, tok, mod)
        res = []
        total = len(unip)
        for idx, x in enumerate(unip):
            v_p = vec(x["p"], tok, mod)
            s = sim(v_p, v_drug)
            res.append({"src": x["src"], "p": x["p"], "score": round(s, 4)})
            if idx % 50 == 0:
                progress_bar.progress(min(idx / total, 1.0))
                status_text.text(f"打分进度: {idx}/{total}")
        progress_bar.progress(1.0)
        status_text.text(f"打分完成: {total}/{total}")
        with open(score_cache_file, "wb") as f:
            pickle.dump(res, f)
        st.success(f"已完成打分并缓存")

    # 含量标注
    for x in res:
        x["content_pct"] = get_content(x["src"])

    # 过滤
    filtered_res = [x for x in res if x["content_pct"] >= content_threshold]
    if filtered_res:
        res = filtered_res

    # 排序
    res.sort(key=lambda x: x["score"], reverse=True)

    # 显示Top结果
    st.subheader("结构相似度排名")
    df_top = pd.DataFrame(res[:top_n])
    show_cols = ["src", "p", "score"]
    if "content_pct" in df_top.columns:
        show_cols.append("content_pct")
    st.dataframe(df_top[show_cols], use_container_width=True)

    # 下载按钮
    csv_top = df_top.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载Top结果CSV", csv_top, "out_top.csv", "text/csv")

    df_all = pd.DataFrame(res)
    csv_all = df_all.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载全部结果CSV", csv_all, "out_all.csv", "text/csv")

    # 物种统计（全部候选肽）
    st.subheader("物种贡献统计（全部候选肽）")
    species_counts = {}
    for x in res:
        s = (
            x["src"].split("(")[-1].split(")")[0].strip()
            if "(" in x["src"]
            else "unknown"
        )
        species_counts[s] = species_counts.get(s, 0) + 1
    species_df = pd.DataFrame(
        list(species_counts.items()), columns=["物种", "候选肽数量"]
    )
    species_df = species_df.sort_values("候选肽数量", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(species_df, use_container_width=True)
    with col2:
        st.bar_chart(species_df.set_index("物种"))

    st.success("筛选完成！")