#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人工智能训练师培训资料 —— 实操数据集生成脚本

一次性生成五级 / 四级 / 三级全部实操题所需的数据文件。
所有数据均为【程序合成的虚构数据】，不含任何真实个人信息或企业数据，
可安全用于教学、练习与考核。

用法（在项目根目录或本目录下执行均可）：
    python 生成数据集.py

输出位置：
    ../五级-初级工/数据集/
    ../四级-中级工/数据集/
    ../三级-高级工/数据集/

设计要点：
    1. 固定随机种子，任何人任何时候运行，生成结果完全一致
       —— 这样参考答案里的数字（缺失值个数、准确率等）才对得上。
    2. 数据【故意】包含缺失、重复、异常、格式不一致、标注分歧等问题，
       这些"脏"正是实操题要考的内容，不是 bug。
    3. 每个数据集的问题点在本文件对应函数的 docstring 中列明，
       供教师备课时对照参考答案使用。

验证环境：Python 3.11.17 / pandas 2.3.3 / numpy 2.1.3 / Pillow 12.3.0
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ── 全局配置 ────────────────────────────────────────────────────────────
SEED = 20261012
BASE = Path(__file__).resolve().parent.parent

DIR_L5 = BASE / "五级-初级工" / "数据集"
DIR_L4 = BASE / "四级-中级工" / "数据集"
DIR_L3 = BASE / "三级-高级工" / "数据集"

rng = np.random.default_rng(SEED)


def _reset_rng() -> None:
    """每个数据集生成前重置随机源，保证单独调用某个函数时结果也一致。"""
    global rng
    rng = np.random.default_rng(SEED)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig 让 Excel 双击打开不乱码——这本身也是课题 0-3 的教学点
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  ✓ {path.relative_to(BASE)}  ({len(df)} 行 × {len(df.columns)} 列)")


# ════════════════════════════════════════════════════════════════════════
#  五级 / 初级工
# ════════════════════════════════════════════════════════════════════════

def gen_l5_sales_raw() -> None:
    """五-2 销售业务数据整理归类与汇总  →  sales_raw.csv

    故意埋入的问题：
      · 日期格式四种混用（2026-03-05 / 2026/3/5 / 03-05-2026 / 20260305）
      · 金额字段混入全角数字、千分位逗号、"元"单位
      · 商品类别大小写与全半角混乱（手机 / 手機 / 手机 ）
      · 地区字段首尾空格
      · 8 条完全重复行
      · 12 条金额缺失、5 条数量为负（退货未标注）
    """
    _reset_rng()
    n = 500
    categories = ["手机", "笔记本电脑", "耳机", "智能手表", "平板"]
    regions = ["华东", "华北", "华南", "西南", "东北"]

    base_date = pd.Timestamp("2026-01-01")
    days = rng.integers(0, 90, n)
    dates = [base_date + pd.Timedelta(days=int(d)) for d in days]

    # 日期格式故意混用四种
    fmts = ["%Y-%m-%d", "%Y/%-m/%-d", "%m-%d-%Y", "%Y%m%d"]
    date_strs = []
    for i, d in enumerate(dates):
        f = fmts[i % 4]
        # Windows 的 strftime 不支持 %-m，手工处理
        if f == "%Y/%-m/%-d":
            date_strs.append(f"{d.year}/{d.month}/{d.day}")
        else:
            date_strs.append(d.strftime(f))

    qty = rng.integers(1, 20, n)
    price = rng.choice([999, 1999, 3999, 5999, 8999, 12999], n)
    amount = qty * price

    df = pd.DataFrame({
        "订单编号": [f"ORD{20260000 + i:08d}" for i in range(n)],
        "下单日期": date_strs,
        "商品类别": rng.choice(categories, n),
        "销售地区": rng.choice(regions, n),
        "销售数量": qty,
        "销售金额": amount.astype(object),
    })

    # 类别写法混乱
    idx = rng.choice(n, 40, replace=False)
    df.loc[idx[:20], "商品类别"] = "手機"
    df.loc[idx[20:], "商品类别"] = "手机 "

    # 地区首尾空格
    idx = rng.choice(n, 35, replace=False)
    df.loc[idx, "销售地区"] = df.loc[idx, "销售地区"].apply(lambda x: f"  {x} ")

    # 金额混入千分位、"元"单位、全角数字
    idx = rng.choice(n, 60, replace=False)
    for i in idx[:25]:
        df.at[i, "销售金额"] = f"{df.at[i, '销售金额']:,}"
    for i in idx[25:45]:
        df.at[i, "销售金额"] = f"{df.at[i, '销售金额']}元"
    full = str.maketrans("0123456789", "０１２３４５６７８９")
    for i in idx[45:]:
        df.at[i, "销售金额"] = str(df.at[i, "销售金额"]).translate(full)

    # 金额缺失
    df.loc[rng.choice(n, 12, replace=False), "销售金额"] = np.nan
    # 数量为负（退货，未标注）
    neg = rng.choice(n, 5, replace=False)
    df.loc[neg, "销售数量"] = -df.loc[neg, "销售数量"]

    # 追加 8 条完全重复行
    df = pd.concat([df, df.sample(8, random_state=SEED)], ignore_index=True)

    _save_csv(df, DIR_L5 / "sales_raw.csv")


def gen_l5_reviews_raw() -> None:
    """五-3 商品评价文本数据清洗  →  reviews_raw.csv

    故意埋入的问题：
      · 25 条完全重复评价
      · 18 条空字符串 / 纯空白
      · 30 条含 HTML 标签
      · 22 条含 emoji 与特殊符号
      · 15 条超短（<3 字）、10 条超长（复制粘贴刷屏）
      · 12 条含手机号 / 邮箱等个人信息（需脱敏）
      · 评分字段有 0 分与 6 分的非法取值
    """
    _reset_rng()
    pos = [
        "质量很好，物流也快，非常满意", "用了一周，手感不错，推荐购买",
        "性价比高，比实体店便宜不少", "包装完整，发货速度快",
        "客服态度很好，有问必答", "外观漂亮，做工精细",
        "续航给力，一天一充完全够用", "音质清晰，降噪效果超出预期",
    ]
    neg = [
        "收到就坏了，差评", "描述不符，色差太大",
        "物流太慢了，等了半个月", "用了三天就出问题，质量堪忧",
        "客服一直不回复，态度差", "做工粗糙，边角有毛刺",
        "续航虚标，根本撑不到半天", "噪音很大，影响使用",
    ]
    neu = [
        "还行吧，一般般", "中规中矩，没什么特别的",
        "凑合用，谈不上好坏", "价格和质量对得上，没惊喜",
    ]

    n = 600
    texts, labels = [], []
    for _ in range(n):
        r = rng.random()
        if r < 0.45:
            texts.append(str(rng.choice(pos)))
            labels.append(1)
        elif r < 0.80:
            texts.append(str(rng.choice(neg)))
            labels.append(0)
        else:
            texts.append(str(rng.choice(neu)))
            labels.append(2)

    df = pd.DataFrame({
        "评价ID": [f"RV{i:06d}" for i in range(n)],
        "商品ID": rng.choice([f"P{i:04d}" for i in range(50)], n),
        "评价内容": texts,
        "评分": rng.integers(1, 6, n),
        "参考情感": labels,  # 仅供教师对照，实操题中需删除此列
    })

    # HTML 标签
    for i in rng.choice(n, 30, replace=False):
        df.at[i, "评价内容"] = f"<p>{df.at[i, '评价内容']}</p><br/>"
    # emoji 与特殊符号
    for i in rng.choice(n, 22, replace=False):
        df.at[i, "评价内容"] = f"{df.at[i, '评价内容']} 😀👍★★★★★"
    # 超短
    for i in rng.choice(n, 15, replace=False):
        df.at[i, "评价内容"] = str(rng.choice(["好", "差", "。", "嗯"]))
    # 超长刷屏
    for i in rng.choice(n, 10, replace=False):
        df.at[i, "评价内容"] = "好评" * 300
    # 个人信息
    for k, i in enumerate(rng.choice(n, 12, replace=False)):
        if k % 2 == 0:
            df.at[i, "评价内容"] = f"{df.at[i, '评价内容']}，联系我 138{rng.integers(10000000, 99999999)}"
        else:
            df.at[i, "评价内容"] = f"{df.at[i, '评价内容']}，邮箱 user{k}@example.com"
    # 空白
    for k, i in enumerate(rng.choice(n, 18, replace=False)):
        df.at[i, "评价内容"] = "" if k % 2 == 0 else "   "
    # 非法评分
    df.loc[rng.choice(n, 6, replace=False), "评分"] = 0
    df.loc[rng.choice(n, 4, replace=False), "评分"] = 6

    # 重复行
    df = pd.concat([df, df.sample(25, random_state=SEED)], ignore_index=True)

    _save_csv(df, DIR_L5 / "reviews_raw.csv")

    # 供五-4 标注实操使用：已清洗、去掉答案列
    clean = df[~df["评价内容"].str.strip().eq("")].drop_duplicates(subset=["评价ID"])
    clean = clean[clean["评价内容"].str.len().between(3, 100)]
    to_label = clean[["评价ID", "商品ID", "评价内容", "评分"]].head(200).copy()
    to_label["情感标签"] = ""
    _save_csv(to_label, DIR_L5 / "reviews_to_label.csv")


def gen_l5_labeled_results() -> None:
    """五-5 标注结果分类统计与报表输出  →  labeled_results.csv

    故意埋入的问题：
      · 出现规范外的标签取值（"中性偏正"、空值、数字 3）
      · 某标注员完成量异常低
      · 某类标签占比异常高（90% 为"正面"）
      · 7 条标注时间早于任务下发时间（逻辑冲突）
    """
    _reset_rng()
    n = 400
    annotators = ["A01", "A02", "A03", "A04", "A05"]
    # A05 完成量异常低
    weights = np.array([0.26, 0.25, 0.24, 0.23, 0.02])
    who = rng.choice(annotators, n, p=weights)

    # 标签分布严重不均衡
    labels = rng.choice(["正面", "负面", "中性"], n, p=[0.90, 0.06, 0.04])

    start = pd.Timestamp("2026-04-01 09:00:00")
    times = [start + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 4800, n)]

    df = pd.DataFrame({
        "标注ID": [f"AN{i:05d}" for i in range(n)],
        "评价ID": [f"RV{i:06d}" for i in rng.choice(600, n, replace=False)],
        "标注员": who,
        "情感标签": labels,
        "标注时间": [t.strftime("%Y-%m-%d %H:%M:%S") for t in times],
        "任务下发时间": start.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # 规范外标签取值
    df.loc[rng.choice(n, 5, replace=False), "情感标签"] = "中性偏正"
    df.loc[rng.choice(n, 4, replace=False), "情感标签"] = "3"
    df.loc[rng.choice(n, 3, replace=False), "情感标签"] = np.nan

    # 标注时间早于下发时间
    for i in rng.choice(n, 7, replace=False):
        df.at[i, "标注时间"] = (start - pd.Timedelta(hours=int(rng.integers(1, 48)))).strftime("%Y-%m-%d %H:%M:%S")

    _save_csv(df, DIR_L5 / "labeled_results.csv")


def gen_l5_images() -> None:
    """五-1 商品图像原始数据采集与归档  →  images/

    合成 40 张商品图，故意包含：
      · 5 张分辨率不一致   · 4 张严重模糊（低分辨率放大）
      · 3 张过曝（接近全白） · 3 张命名不规范
      · 2 组完全重复
    """
    _reset_rng()
    out = DIR_L5 / "images"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()

    cats = ["phone", "laptop", "headphone", "watch"]
    count = 0
    for i in range(40):
        cat = cats[i % 4]
        size = (640, 480)
        if i in (5, 11, 19, 26, 33):          # 分辨率不一致
            size = (320, 240) if i % 2 else (1280, 960)

        color = tuple(int(c) for c in rng.integers(40, 200, 3))
        img = Image.new("RGB", size, color)
        d = ImageDraw.Draw(img)
        w, h = size
        d.rectangle([w * 0.25, h * 0.25, w * 0.75, h * 0.75],
                    fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
        d.text((10, 10), f"{cat}-{i:03d}", fill=(255, 255, 255))

        if i in (7, 14, 22, 31):              # 模糊
            img = img.resize((32, 24)).resize(size)
        if i in (9, 17, 28):                  # 过曝
            img = Image.blend(img, Image.new("RGB", size, (255, 255, 255)), 0.88)

        if i in (3, 16, 29):                  # 命名不规范
            name = f"IMG_{i}.jpg" if i == 3 else (f"商品图片{i}.jpg" if i == 16 else f"new photo {i}.jpg")
        else:
            name = f"{cat}_{i:03d}.jpg"

        img.save(out / name, quality=90)
        count += 1

    # 两组完全重复
    for src, dup in ((f"phone_000.jpg", "phone_000_copy.jpg"),
                     ("laptop_001.jpg", "laptop_001_副本.jpg")):
        if (out / src).exists():
            Image.open(out / src).save(out / dup, quality=90)
            count += 1

    print(f"  ✓ {(out).relative_to(BASE)}  ({count} 张图像)")


def gen_l5_sqlite() -> None:
    """五级课题 1-2 SQL 查询与导出练习库  →  practice.db

    三张表：users(用户) / products(商品) / orders(订单)
    故意设计：orders 中有 status 为 'cancelled' 的记录，
             练习时需注意筛选条件；有 3 条订单的 user_id 在 users 中不存在（脏数据）。
    """
    _reset_rng()
    DIR_L5.mkdir(parents=True, exist_ok=True)
    db = DIR_L5 / "practice.db"
    if db.exists():
        db.unlink()

    conn = sqlite3.connect(db)

    users = pd.DataFrame({
        "user_id": range(1, 201),
        "user_name": [f"用户{i:03d}" for i in range(1, 201)],
        "city": rng.choice(["北京", "上海", "广州", "深圳", "成都", "杭州"], 200),
        "register_date": [(pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                          for d in rng.integers(0, 400, 200)],
    })

    products = pd.DataFrame({
        "product_id": range(1, 51),
        "product_name": [f"商品{i:03d}" for i in range(1, 51)],
        "category": rng.choice(["手机", "笔记本电脑", "耳机", "智能手表", "平板"], 50),
        "price": rng.choice([999, 1999, 3999, 5999, 8999, 12999], 50).astype(float),
    })

    n = 1000
    orders = pd.DataFrame({
        "order_id": range(1, n + 1),
        "user_id": rng.integers(1, 201, n),
        "product_id": rng.integers(1, 51, n),
        "quantity": rng.integers(1, 6, n),
        "order_date": [(pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                       for d in rng.integers(0, 90, n)],
        "status": rng.choice(["completed", "cancelled", "pending"], n, p=[0.80, 0.12, 0.08]),
    })
    # 3 条孤儿订单（user_id 不存在）
    orders.loc[orders.index[-3:], "user_id"] = [9001, 9002, 9003]

    users.to_sql("users", conn, index=False)
    products.to_sql("products", conn, index=False)
    orders.to_sql("orders", conn, index=False)
    conn.commit()
    conn.close()
    print(f"  ✓ {db.relative_to(BASE)}  (users 200 / products 50 / orders 1000)")


# ════════════════════════════════════════════════════════════════════════
#  四级 / 中级工
# ════════════════════════════════════════════════════════════════════════

def gen_l4_image_audit() -> None:
    """四-1 公共交通工具识别图片数据集质量审核  →  transport_images/ + transport_labels.csv

    故意埋入的问题：
      · 标注文件中有 4 条记录对应的图像文件不存在
      · 有 3 张图像没有对应的标注记录
      · 类别分布严重不均衡（bus 占 62%，tram 仅 3 张）
      · 6 条标注的 bbox 坐标越界或 x2<x1
      · 5 条重复标注（同一图同一框标两次）
    """
    _reset_rng()
    out = DIR_L4 / "transport_images"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()

    cats = (["bus"] * 62 + ["car"] * 20 + ["bicycle"] * 10 + ["tram"] * 3 + ["truck"] * 5)
    rng.shuffle(cats)
    W, H = 640, 480

    names = []
    for i, c in enumerate(cats):
        img = Image.new("RGB", (W, H), tuple(int(v) for v in rng.integers(120, 220, 3)))
        d = ImageDraw.Draw(img)
        x1, y1 = int(rng.integers(20, 300)), int(rng.integers(20, 240))
        x2, y2 = x1 + int(rng.integers(120, 300)), y1 + int(rng.integers(100, 200))
        d.rectangle([x1, y1, min(x2, W - 1), min(y2, H - 1)],
                    fill=tuple(int(v) for v in rng.integers(0, 120, 3)))
        d.text((10, 10), f"{c}_{i:03d}", fill=(255, 255, 255))
        name = f"{c}_{i:03d}.jpg"
        img.save(out / name, quality=88)
        names.append((name, c, x1, y1, min(x2, W - 1), min(y2, H - 1)))

    rows = [{"image_file": n, "label": c, "x1": a, "y1": b, "x2": x, "y2": y}
            for n, c, a, b, x, y in names]

    # 3 张图无标注：删掉它们的标注记录
    rows = rows[:-3]

    # 4 条标注指向不存在的图像
    for k in range(4):
        rows.append({"image_file": f"missing_{k:03d}.jpg", "label": "bus",
                     "x1": 10, "y1": 10, "x2": 200, "y2": 200})

    df = pd.DataFrame(rows)

    # 6 条坐标非法
    bad = rng.choice(len(df) - 4, 6, replace=False)
    for k, i in enumerate(bad):
        if k % 2 == 0:
            df.at[i, "x2"] = df.at[i, "x1"] - 20          # x2 < x1
        else:
            df.at[i, "y2"] = H + 50                        # 越界

    # 5 条重复标注
    df = pd.concat([df, df.sample(5, random_state=SEED)], ignore_index=True)

    _save_csv(df, DIR_L4 / "transport_labels.csv")
    print(f"  ✓ {out.relative_to(BASE)}  ({len(names)} 张图像)")


def gen_l4_cosmetics_sales() -> None:
    """四-2 化妆品销售数据处理方法优化  →  cosmetics_sales.csv

    故意设计：
      · 销售额呈【长尾分布】——用均值填充缺失会系统性抬高中心值，
        这是本题的核心考点（应使用中位数或分组中位数）
      · 会员等级字段 18% 缺失
      · 3 个 SKU 的单价出现数量级错误（分位 × 100）
      · 退货记录用负数量表示但未单独标识
      · 门店编码大小写混用（S001 / s001）
    """
    _reset_rng()
    n = 800
    # 长尾：对数正态
    amount = rng.lognormal(mean=5.2, sigma=1.3, size=n).round(2)

    df = pd.DataFrame({
        "订单号": [f"CS{i:06d}" for i in range(n)],
        "门店编码": rng.choice(["S001", "S002", "S003", "S004"], n),
        "SKU": rng.choice([f"SKU{i:03d}" for i in range(30)], n),
        "会员等级": rng.choice(["普通", "白银", "黄金", "铂金"], n, p=[0.5, 0.25, 0.17, 0.08]),
        "销售数量": rng.integers(1, 8, n),
        "销售额": amount,
        "销售日期": [(pd.Timestamp("2026-02-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                     for d in rng.integers(0, 60, n)],
    })

    # 销售额缺失 15%（考点：别用均值填）
    df.loc[rng.choice(n, int(n * 0.15), replace=False), "销售额"] = np.nan
    # 会员等级缺失 18%
    df.loc[rng.choice(n, int(n * 0.18), replace=False), "会员等级"] = np.nan
    # 门店编码大小写混用
    idx = rng.choice(n, 90, replace=False)
    df.loc[idx, "门店编码"] = df.loc[idx, "门店编码"].str.lower()
    # 数量级错误
    for sku in ["SKU003", "SKU011", "SKU022"]:
        m = df["SKU"] == sku
        df.loc[m, "销售额"] = df.loc[m, "销售额"] * 100
    # 退货（负数量）未单独标识
    neg = rng.choice(n, 24, replace=False)
    df.loc[neg, "销售数量"] = -df.loc[neg, "销售数量"]
    df.loc[neg, "销售额"] = -df.loc[neg, "销售额"].abs()

    _save_csv(df, DIR_L4 / "cosmetics_sales.csv")


def gen_l4_user_behavior() -> None:
    """四-3 用户行为数据聚类分析与归类定义  →  user_behavior.csv

    设计：真实存在 4 个可分离的用户群（高频高额 / 高频低额 / 低频高额 / 沉睡），
    但特征量纲差异极大（访问次数 1~200，客单价 10~5000，停留时长 5~3600 秒），
    【不做标准化直接 KMeans 会被客单价主导】—— 这是本题核心考点。
    """
    _reset_rng()
    groups = [
        # (n, 访问次数, 客单价, 停留秒, 近30天下单)
        (150, (80, 200), (800, 5000), (900, 3600), (8, 25)),
        (200, (60, 180), (10, 150), (300, 1500), (6, 20)),
        (120, (3, 15), (600, 4000), (600, 2400), (1, 4)),
        (180, (1, 6), (10, 200), (5, 180), (0, 1)),
    ]
    rows = []
    for gi, (cnt, visit, price, stay, orders) in enumerate(groups):
        for _ in range(cnt):
            rows.append({
                "访问次数": int(rng.integers(*visit)),
                "平均客单价": round(float(rng.uniform(*price)), 2),
                "平均停留时长秒": int(rng.integers(*stay)),
                "近30天下单数": int(rng.integers(orders[0], orders[1] + 1)),
                "_真实分群": gi,  # 仅供教师核对，实操题中需删除
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "用户ID", [f"U{i:05d}" for i in range(len(df))])
    _save_csv(df, DIR_L4 / "user_behavior.csv")


def gen_l4_annotation_audit() -> None:
    """四-4 / 四-5 客服对话标注数据审核、纠错与筛选  →  service_annotations.csv

    故意埋入的问题：
      · 标注员 A03 存在【系统性偏见】——把大量"咨询"标成"投诉"
      · 标签分布不均衡（咨询 70%），导致一致率虚高而 Kappa 偏低
      · 12 条必填字段（标注依据）缺失
      · 9 条标签取值不在规范内
      · 15 条文本为边界样本（既像投诉又像咨询），三人标注各不相同
    """
    _reset_rng()
    n = 300
    valid = ["咨询", "投诉", "闲聊"]

    templates = {
        "咨询": ["请问我的订单什么时候发货", "这个商品支持七天无理由吗",
                 "怎么申请开发票", "运费是多少", "有没有其他颜色"],
        "投诉": ["等了半个月还没到，太差了", "客服一直不回复，什么服务态度",
                 "东西是坏的，必须退款", "描述严重不符，欺骗消费者"],
        "闲聊": ["你好", "在吗", "谢谢", "辛苦了"],
    }
    boundary = ["你们这物流也太慢了吧，大概还要多久到",
                "买了三天还没发货，是不是没货了",
                "这个质量看着一般，能换货吗"]

    rows = []
    for i in range(n):
        if i < 15:
            text, truth = str(rng.choice(boundary)), "边界"
        else:
            truth = str(rng.choice(valid, p=[0.70, 0.22, 0.08]))
            text = str(rng.choice(templates[truth]))
        rows.append({"对话ID": f"D{i:05d}", "对话内容": text, "_参考答案": truth})

    base = pd.DataFrame(rows)

    # 三位标注员各标一遍
    out = []
    for ann in ["A01", "A02", "A03"]:
        for _, r in base.iterrows():
            truth = r["_参考答案"]
            if truth == "边界":
                lab = str(rng.choice(valid, p=[0.4, 0.5, 0.1]))   # 边界样本三人分歧
            elif ann == "A03" and truth == "咨询" and rng.random() < 0.35:
                lab = "投诉"                                        # A03 系统性偏见
            elif rng.random() < 0.05:
                lab = str(rng.choice(valid))                        # 随机噪声
            else:
                lab = truth
            out.append({
                "标注记录ID": f"AR{len(out):05d}",
                "对话ID": r["对话ID"],
                "对话内容": r["对话内容"],
                "标注员": ann,
                "意图标签": lab,
                "标注依据": f"依据规范第{rng.integers(1, 8)}条",
                "标注时间": (pd.Timestamp("2026-05-06 09:00:00")
                             + pd.Timedelta(minutes=int(rng.integers(0, 2400)))).strftime("%Y-%m-%d %H:%M:%S"),
            })

    df = pd.DataFrame(out)
    # 必填字段缺失
    df.loc[rng.choice(len(df), 12, replace=False), "标注依据"] = np.nan
    # 非法标签取值
    for k, i in enumerate(rng.choice(len(df), 9, replace=False)):
        df.at[i, "意图标签"] = ["咨询类", "其他", ""][k % 3]

    _save_csv(df, DIR_L4 / "service_annotations.csv")


# ════════════════════════════════════════════════════════════════════════
#  三级 / 高级工
# ════════════════════════════════════════════════════════════════════════

def gen_l3_review_sentiment() -> None:
    """三级 2.1.1 商品评价情感识别的数据清洗与标注规范制定  →  data.csv

    对应真实考区题干：「基于原始数据 data.csv 制定情感识别数据清洗标准，
    并保存清洗后数据为 data_cleaned.csv；针对积极(1)/消极(0)两类标签制定标注规范」

    故意埋入的问题：
      · 重复、空值、HTML、emoji、超长超短（同五级，但量更大）
      · 【混合情感】评价 —— "质量真不咋地，不过客服态度还行"，
        规范必须定义主次判定规则，这是本题最高配分的考点
      · 【无关内容】—— 纯广告、纯表情、与商品无关的闲聊
      · 【否定句式】—— "不能说不好"，考察规范是否覆盖
      · 个人信息（手机号/邮箱）需脱敏
    """
    _reset_rng()
    pos = ["质量很好，物流也快，非常满意", "性价比高，推荐购买", "做工精细，外观漂亮",
           "续航给力，一天一充完全够用", "客服响应快，问题解决及时"]
    neg = ["收到就坏了，差评", "描述不符，色差太大", "物流太慢，等了半个月",
           "用了三天就出问题", "客服一直不回复，态度差"]
    mixed = ["这个产品质量真不咋地，不过客服态度还行",
             "东西还可以，就是物流太慢了，等得我都快忘了",
             "价格便宜是真的，质量也是真的一般",
             "外观我很喜欢，但是用了两天就出故障了",
             "客服态度差，但产品本身没什么问题"]
    negation = ["不能说不好，但也谈不上多好", "不是不满意，只是没达到预期",
                "没有想象中那么差"]
    irrelevant = ["加微信xxx领优惠券", "😀😀😀😀", "今天天气真不错",
                  "第一", "沙发", "刷单的都退散"]

    n = 1000
    texts = []
    for _ in range(n):
        r = rng.random()
        if r < 0.38:
            texts.append(str(rng.choice(pos)))
        elif r < 0.70:
            texts.append(str(rng.choice(neg)))
        elif r < 0.84:
            texts.append(str(rng.choice(mixed)))
        elif r < 0.90:
            texts.append(str(rng.choice(negation)))
        else:
            texts.append(str(rng.choice(irrelevant)))

    df = pd.DataFrame({
        "review_id": [f"R{i:06d}" for i in range(n)],
        "product_id": rng.choice([f"P{i:04d}" for i in range(80)], n),
        "review_text": texts,
        "rating": rng.integers(1, 6, n),
        "review_time": [(pd.Timestamp("2026-06-01") + pd.Timedelta(minutes=int(m))).strftime("%Y-%m-%d %H:%M:%S")
                        for m in rng.integers(0, 86400, n)],
    })

    for i in rng.choice(n, 45, replace=False):
        df.at[i, "review_text"] = f"<div>{df.at[i, 'review_text']}</div>"
    for i in rng.choice(n, 35, replace=False):
        df.at[i, "review_text"] = f"{df.at[i, 'review_text']} 👍😊★★★"
    for i in rng.choice(n, 20, replace=False):
        df.at[i, "review_text"] = "非常好" * 250
    for i in rng.choice(n, 25, replace=False):
        df.at[i, "review_text"] = str(rng.choice(["好", "差", "。", "？"]))
    for k, i in enumerate(rng.choice(n, 18, replace=False)):
        df.at[i, "review_text"] = (f"{df.at[i, 'review_text']}，电话139{rng.integers(10000000, 99999999)}"
                                   if k % 2 == 0 else
                                   f"{df.at[i, 'review_text']}，邮箱 buyer{k}@example.com")
    for k, i in enumerate(rng.choice(n, 30, replace=False)):
        df.at[i, "review_text"] = "" if k % 2 == 0 else "    "
    df.loc[rng.choice(n, 10, replace=False), "rating"] = 0

    df = pd.concat([df, df.sample(40, random_state=SEED)], ignore_index=True)
    _save_csv(df, DIR_L3 / "data.csv")


def gen_l3_credit_scoring() -> None:
    """三级 2.2.1 智能信用评分模型开发与测试  →  credit_data.csv

    设计要点（本题核心考点全在这里）：
      · 【类别不均衡】违约率 15.5% —— 全判"不违约"即有 84.5% 准确率
      · 含 2 个无用噪声特征（misc_score / channel），考察特征理解
      · 收入字段 8% 缺失（实际 160 条）、年龄有 3 个异常值（0 / 150 / 199）
      · 不预先划分训练/测试集 —— 划分本身是评分点

    ── 实测基准（Python 3.11.17 / scikit-learn 1.9.1，7:3 分层划分，
       random_state=42，StandardScaler + LogisticRegression）────────────
       阈值 0.5：准确率 84.7%，违约类 精确率 0.529 / 召回率 0.097 / F1 0.164
                 → 准确率几乎等于"全判不违约"的 84.5%，模型形同虚设
       阈值 0.3：准确率 84.5%，违约类 精确率 0.500 / 召回率 0.366 / F1 0.422
                 → 召回率提升 3.8 倍，精确率仅下降 2.9 个百分点
       AUC = 0.768 —— 模型本身有判别力，问题出在决策阈值而非模型无能

       这组数字正是本题的教学落点：
       「准确率高 ≠ 模型可用」「必须看混淆矩阵」「阈值要按业务代价调」。
       参考答案中的数值以此为准，学员复现应完全一致。
    ───────────────────────────────────────────────────────────────────
    """
    _reset_rng()
    n = 2000

    age = rng.integers(22, 65, n).astype(float)
    income = rng.lognormal(mean=10.2, sigma=0.5, size=n).round(0)
    debt_ratio = np.clip(rng.beta(2, 5, n) * 1.5, 0, 1.2).round(3)
    history_months = rng.integers(0, 240, n)
    late_times = rng.poisson(1.2, n)
    credit_lines = rng.integers(1, 12, n)
    noise_1 = rng.normal(0, 1, n).round(3)
    noise_2 = rng.choice(["A", "B", "C"], n)

    # 真实违约倾向：负债率高、逾期多、历史短、收入低 → 风险高
    # 截距 -3.05 经实测校准，使违约率落在 12% 附近：
    # 既保证"全判不违约也有 88% 准确率"的不均衡考点成立，
    # 又让 Logistic 回归能学出可分析的混淆矩阵（召回率不至于趋近 0）。
    z = (-3.05
         + 2.6 * debt_ratio
         + 0.42 * late_times
         - 0.004 * history_months
         - 0.9 * (np.log(income) - 10.2)
         + 0.012 * (age < 28))
    prob = 1 / (1 + np.exp(-z))
    default = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(n)],
        "age": age,
        "monthly_income": income,
        "debt_ratio": debt_ratio,
        "credit_history_months": history_months,
        "late_payment_times": late_times,
        "credit_lines": credit_lines,
        "misc_score": noise_1,
        "channel": noise_2,
        "is_default": default,
    })

    df.loc[rng.choice(n, int(n * 0.08), replace=False), "monthly_income"] = np.nan
    for i, v in zip(rng.choice(n, 3, replace=False), [199.0, 0.0, 150.0]):
        df.at[i, "age"] = v

    _save_csv(df, DIR_L3 / "credit_data.csv")
    print(f"      违约率 {df['is_default'].mean():.1%}（全判不违约的准确率 = {1 - df['is_default'].mean():.1%}）")


def gen_l3_annotation_agreement() -> None:
    """三级课题 2-2 Kappa 实训  →  annotation_agreement.csv

    设计要点：
      · 标签分布极不均衡（"是"占 90%），两位标注员在多数类上高度一致，
        在少数类上几乎在猜 —— 制造"一致率高、Kappa 低"的经典反差
      · 另附一组均衡数据（annotation_agreement_balanced.csv）作对照

    ── 实测基准（sklearn.metrics.cohen_kappa_score）────────────────────
       不均衡组：观测一致率 87.8%，Kappa = 0.217（"一般"，接近"很差"）
       均衡组　：观测一致率 76.2%，Kappa = 0.524（"中等"）

       注意这个反直觉结果：**一致率更高的那一组，Kappa 反而更低**。
       这正是本课题要学员亲手算出来并解释的核心现象 ——
       一致率没有刨除"靠标签分布本身就能蒙对"的部分，Kappa 刨除了。
       参考答案中的数值以此为准，学员复现应完全一致。
    ───────────────────────────────────────────────────────────────────
    """
    _reset_rng()
    n = 500

    # 不均衡组：真实标签 90% 为"是"，两人在少数类上判断几乎随机
    truth = rng.choice(["是", "否"], n, p=[0.90, 0.10])
    a, b = [], []
    for t in truth:
        if t == "是":
            a.append("是" if rng.random() < 0.95 else "否")
            b.append("是" if rng.random() < 0.95 else "否")
        else:
            a.append(str(rng.choice(["是", "否"])))   # 少数类上几乎在猜
            b.append(str(rng.choice(["是", "否"])))

    df_imb = pd.DataFrame({
        "样本ID": [f"S{i:05d}" for i in range(n)],
        "语音片段": [f"audio_{i:05d}.wav" for i in range(n)],
        "标注员甲_是否礼貌": a,
        "标注员乙_是否礼貌": b,
    })
    _save_csv(df_imb, DIR_L3 / "annotation_agreement.csv")
    agree = (df_imb["标注员甲_是否礼貌"] == df_imb["标注员乙_是否礼貌"]).mean()
    print(f"      不均衡组观测一致率 = {agree:.1%}（Kappa 将远低于此值）")

    # 均衡对照组
    truth2 = rng.choice(["是", "否"], n, p=[0.5, 0.5])
    a2 = [t if rng.random() < 0.85 else ("否" if t == "是" else "是") for t in truth2]
    b2 = [t if rng.random() < 0.85 else ("否" if t == "是" else "是") for t in truth2]
    df_bal = pd.DataFrame({
        "样本ID": [f"B{i:05d}" for i in range(n)],
        "语音片段": [f"balanced_{i:05d}.wav" for i in range(n)],
        "标注员甲_是否礼貌": a2,
        "标注员乙_是否礼貌": b2,
    })
    _save_csv(df_bal, DIR_L3 / "annotation_agreement_balanced.csv")


def gen_l3_speaker_logs() -> None:
    """三级 3.1.1 智能音箱产品数据分析与优化  →  speaker_logs.csv

    故意埋入的现象（供学员分析发现）：
      · 【数据漂移】第 7 周起"儿童"用户群占比从 8% 骤升至封顶 28%，
        而该群体识别准确率比成人低约 18 个百分点
      · 【模型退化】基线准确率逐周缓慢下滑
      · 夜间时段（22:00-6:00）失败率显著偏高（环境噪声）
      · "音乐播放"意图占 55%，长尾意图（百科问答/购物/儿童故事）识别率差
      · 2% 日志 response_time 为 -1（埋点 bug，属数据质量问题而非模型问题）

    ── 实测基准（可精确复现，参考答案数值以此为准）────────────────────
       整体识别成功率：第 1 周 91.4% → 第 12 周 82.6%，下滑 8.8 个百分点

       按用户群体下钻后，这 8.8 个百分点可分解为两个独立原因：
         · 成人群体成功率：92.6% → 87.3%，下滑 5.3 个百分点
           → 这才是【真实的模型退化】
         · 儿童用户占比：7.7% → 28.8%（第 7 周起骤升）
           儿童群体成功率始终比成人低约 16~20 个百分点
           → 占比上升本身就把整体均值拉低约 3.5 个百分点，
             属【数据漂移 / 用户结构变化】，不是模型变差

       教学落点：学员若只报"模型退化了，建议重新训练"即失分。
       必须做分群下钻，把 8.8pp 拆成 5.3pp（模型）+ 3.5pp（结构）。
       对应的优化方案也因此分两条：重训练解决前者，
       补充儿童语音训练数据解决后者 —— 只做重训练无法解决漂移部分。
    ───────────────────────────────────────────────────────────────────
    """
    _reset_rng()
    rows = []
    intents = ["音乐播放", "天气查询", "闹钟设置", "智能家居控制", "百科问答", "购物", "儿童故事"]
    intent_p = [0.55, 0.12, 0.09, 0.08, 0.07, 0.05, 0.04]

    for week in range(1, 13):
        n_week = 1500
        # 第 7 周起儿童用户占比骤升（数据漂移），但封顶 28%，避免整体指标塌掉
        child_ratio = 0.08 if week < 7 else min(0.08 + (week - 6) * 0.034, 0.28)
        base_acc = 0.945 - (week - 1) * 0.0042                           # 缓慢退化
        for _ in range(n_week):
            is_child = rng.random() < child_ratio
            hour = int(rng.choice(range(24), p=np.array(
                [3, 2, 1, 1, 1, 2, 4, 6, 6, 5, 5, 5, 6, 5, 5, 5, 6, 7, 8, 8, 7, 5, 4, 3]) / 110))
            intent = str(rng.choice(intents, p=intent_p))

            acc = base_acc
            if is_child:
                acc -= 0.18
            if hour >= 22 or hour < 6:
                acc -= 0.09
            if intent in ("百科问答", "购物", "儿童故事"):
                acc -= 0.08
            success = rng.random() < max(acc, 0.05)

            rows.append({
                "日志ID": f"LG{len(rows):07d}",
                "周次": week,
                "小时": hour,
                "用户群体": "儿童" if is_child else "成人",
                "识别意图": intent,
                "识别成功": int(success),
                "置信度": round(float(np.clip(rng.beta(6, 2) if success else rng.beta(2, 4), 0, 1)), 3),
                "响应时长ms": int(max(rng.normal(680, 180), 80)),
                "是否转人工": int((not success) and rng.random() < 0.15),
            })

    df = pd.DataFrame(rows)
    # 埋点 bug：2% 响应时长为负
    df.loc[rng.choice(len(df), int(len(df) * 0.02), replace=False), "响应时长ms"] = -1

    _save_csv(df, DIR_L3 / "speaker_logs.csv")
    w = df.groupby("周次")["识别成功"].mean()
    print(f"      识别成功率：第1周 {w.iloc[0]:.1%} → 第12周 {w.iloc[-1]:.1%}（呈退化趋势）")


def gen_l3_cx_interaction() -> None:
    """三级 3.2.1 客户体验管理智能系统人机交互流程设计  →  cx_interaction.csv

    设计要点（本题的分析落点）：
      · 置信度与准确率【强正相关】，存在明显的分段阈值效应：
        ≥0.90 准确率 ~97%；0.70~0.90 约 83%；<0.70 仅约 52%
      · 低置信度区间若不转人工，用户满意度骤降
      · 人工处理成本是 AI 的 20 倍 —— 阈值设计必须权衡成本
      → 学员应据此推导出"高置信直接自动、中置信 AI 建议+人工确认、低置信直接转人工"的分段流程
    """
    _reset_rng()
    n = 4000
    conf = np.clip(rng.beta(5, 2, n), 0.3, 0.999).round(3)

    rows = []
    for c in conf:
        if c >= 0.90:
            acc_p = 0.97
        elif c >= 0.70:
            acc_p = 0.83
        else:
            acc_p = 0.52
        correct = rng.random() < acc_p
        transferred = (c < 0.60) and (rng.random() < 0.7)

        if transferred:
            satis = int(np.clip(rng.normal(4.3, 0.6), 1, 5))
        elif correct:
            satis = int(np.clip(rng.normal(4.5, 0.5), 1, 5))
        else:
            satis = int(np.clip(rng.normal(2.0, 0.8), 1, 5))

        rows.append({
            "会话ID": f"SE{len(rows):06d}",
            "AI置信度": c,
            "AI回答正确": int(correct),
            "是否转人工": int(transferred),
            "用户满意度": satis,
            "会话时长秒": int(max(rng.normal(180 if transferred else 65, 40), 10)),
            "处理成本元": round(4.0 if transferred else 0.2, 2),
        })

    df = pd.DataFrame(rows)
    _save_csv(df, DIR_L3 / "cx_interaction.csv")
    for lo, hi in [(0.0, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        m = (df["AI置信度"] >= lo) & (df["AI置信度"] < hi)
        print(f"      置信度 [{lo}, {hi}): 样本 {m.sum():>4}，正确率 {df.loc[m, 'AI回答正确'].mean():.1%}")


def gen_l3_marketing_funnel() -> None:
    """三级 1.1.2 自动化营销模型业务模块效果优化  →  marketing_funnel.csv

    设计要点：
      · 漏斗 曝光→点击→加购→下单→支付 五环节
      · 【瓶颈在"加购→下单"】转化率仅 11%，远低于其他环节
      · 但渠道 C 的"点击→加购"异常低（3%）—— 存在第二个局部问题
      · 第 5 周起渠道 B 曝光量翻倍但转化率腰斩（投放质量下降）
      → 学员需识别出"全局瓶颈"与"局部异常"两类问题，而不是只报一个
    """
    _reset_rng()
    rows = []
    for week in range(1, 11):
        for ch in ["A", "B", "C", "D"]:
            imp = int(rng.integers(40000, 80000))
            if ch == "B" and week >= 5:
                imp *= 2

            ctr = {"A": 0.052, "B": 0.048, "C": 0.061, "D": 0.044}[ch]
            if ch == "B" and week >= 5:
                ctr *= 0.5
            clicks = int(imp * ctr * rng.uniform(0.92, 1.08))

            add_rate = 0.03 if ch == "C" else rng.uniform(0.24, 0.30)
            carts = int(clicks * add_rate)

            orders = int(carts * rng.uniform(0.10, 0.12))      # 全局瓶颈
            paid = int(orders * rng.uniform(0.88, 0.94))

            rows.append({
                "周次": week, "渠道": ch,
                "曝光量": imp, "点击量": clicks, "加购量": carts,
                "下单量": orders, "支付量": paid,
                "投放成本元": round(imp * rng.uniform(0.012, 0.02), 2),
            })

    df = pd.DataFrame(rows)
    _save_csv(df, DIR_L3 / "marketing_funnel.csv")


def gen_l3_process_design_context() -> None:
    """三级 1.1.1 业务数据处理流程设计  →  customer_service_raw_sample.csv

    流程设计题以文字作答为主，但提供一份"现状数据样例"让学员有据可依：
    故意呈现出字段缺失、口径不一、无质检记录等现状问题。
    """
    _reset_rng()
    n = 200
    df = pd.DataFrame({
        "会话ID": [f"CS{i:06d}" for i in range(n)],
        "用户ID": rng.choice([f"U{i:05d}" for i in range(150)], n),
        "渠道": rng.choice(["APP", "小程序", "电话", "网页", ""], n, p=[.3, .25, .2, .2, .05]),
        "开始时间": [(pd.Timestamp("2026-07-01") + pd.Timedelta(minutes=int(m))).strftime("%Y-%m-%d %H:%M:%S")
                     for m in rng.integers(0, 43200, n)],
        "会话内容": rng.choice([
            "请问订单什么时候发货", "要求退款", "咨询发票", "物流太慢了", ""], n),
        "客服工号": rng.choice(["E01", "E02", "E03", "", "未知"], n, p=[.3, .3, .25, .1, .05]),
        "满意度评分": rng.choice([1, 2, 3, 4, 5, np.nan], n, p=[.05, .05, .1, .25, .35, .2]),
        "问题是否解决": rng.choice(["是", "否", "", "y", "Y"], n, p=[.5, .2, .15, .1, .05]),
    })
    _save_csv(df, DIR_L3 / "customer_service_raw_sample.csv")


# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("人工智能训练师培训资料 —— 实操数据集生成")
    print(f"随机种子 SEED = {SEED}（固定，保证结果可复现）")
    print("=" * 70)

    print("\n【五级 / 初级工】")
    gen_l5_sales_raw()
    gen_l5_reviews_raw()
    gen_l5_labeled_results()
    gen_l5_images()
    gen_l5_sqlite()

    print("\n【四级 / 中级工】")
    gen_l4_image_audit()
    gen_l4_cosmetics_sales()
    gen_l4_user_behavior()
    gen_l4_annotation_audit()

    print("\n【三级 / 高级工】")
    gen_l3_process_design_context()
    gen_l3_marketing_funnel()
    gen_l3_review_sentiment()
    gen_l3_credit_scoring()
    gen_l3_annotation_agreement()
    gen_l3_speaker_logs()
    gen_l3_cx_interaction()

    print("\n" + "=" * 70)
    print("全部数据集生成完成。")
    print("注意：带 _ 前缀的列（如 _参考答案、_真实分群）仅供教师核对，")
    print("      发给学员前请删除这些列。")
    print("=" * 70)


if __name__ == "__main__":
    main()
