from __future__ import annotations

import json
import random
import string
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
ABUSE_TARGET = 420
NORMAL_TARGET = 360
SEXUAL_TARGET = 300
AD_TARGET = 300

DATA_DIR = Path("data")
COLD_BASE_URL = "https://raw.githubusercontent.com/thu-coai/COLDataset/main/COLDataset"


MANUAL_NORMAL_SEEDS = [
    "今晚的电影节奏很好，演员演得很自然。",
    "这个耳机降噪不错，通勤用很方便。",
    "客服回复很快，已经帮我处理退款了。",
    "周末准备去爬山，有没有推荐的路线？",
    "这家面包店新出的芝士包很好吃。",
    "今天开会把需求重新梳理了一遍。",
    "孩子这次考试进步很大，继续加油。",
    "物流速度可以，包装也没有破损。",
    "这个教程写得很清楚，适合新手入门。",
    "昨晚直播讲的内容挺实用的。",
    "我觉得这个产品配色比上一代更好看。",
    "新版本修复了闪退问题，体验好多了。",
    "今天把周报写完了，准备下班回家。",
    "这次更新后加载速度提升明显。",
    "天气不错，下午出去跑步。",
    "这个回答很专业，学到了不少。",
    "售后态度很好，没有推诿。",
    "这个键盘手感不错，打字很舒服。",
    "明天要交周报，我先整理下数据。",
    "朋友推荐的餐厅味道还可以。",
    "终于把 bug 修好了，准备发版。",
]

MANUAL_AD_SEEDS = [
    "加微信领取优惠券，今天最后一天。",
    "兼职刷单日结，宝妈学生都能做。",
    "想做代理的私聊我，利润很高。",
    "低价出售账号，有需要的联系我。",
    "课程限时优惠，扫码进群了解。",
    "引流合作看这里，精准粉丝当天到。",
    "需要办证的加我，速度快价格低。",
    "买粉涨赞加vx咨询，稳定靠谱。",
    "二手手机清仓，批发价处理。",
    "代写作业代做PPT，接急单。",
    "返利活动开始了，拉人就有奖励。",
    "招加盟代理，全国都可以做。",
]

MANUAL_SEXUAL_SEEDS = [
    "私聊我看成人资源，晚上更精彩。",
    "这个群里全是擦边福利视频。",
    "想约的话加我，发你私密照片。",
    "今晚有人一起暧昧聊天吗？",
    "这里有成人内容链接，懂的来。",
    "想看更刺激的内容就私信。",
    "她主页里都是擦边写真。",
    "发点私密照让我看看。",
    "这个账号老是在评论区发约会暗示。",
    "群里有人卖成人视频资源。",
    "这种低俗挑逗内容应该处理一下。",
    "私信聊点不能公开说的话题。",
]


def make_contact_handle(index: int) -> str:
    random.seed(SEED + index)
    letters = "".join(random.choices(string.ascii_lowercase, k=3))
    digits = "".join(random.choices(string.digits, k=4))
    return f"{letters}{digits}"


def build_manual_normal_texts() -> list[str]:
    subjects = [
        "这个版本",
        "这次更新",
        "这家店的服务",
        "这款耳机",
        "这篇教程",
        "今天的会议",
        "这份方案",
        "这个页面",
        "这套键盘",
        "这次直播",
        "这个功能",
        "这个课程",
    ]
    aspects = [
        "的加载速度",
        "的稳定性",
        "的操作流程",
        "的讲解节奏",
        "的售后体验",
        "的整体设计",
        "的响应效率",
        "的包装细节",
        "的音质表现",
        "的交互反馈",
    ]
    evaluations = [
        "比之前顺畅很多",
        "明显更稳定了",
        "用起来挺舒服",
        "细节处理得很到位",
        "整体体验比较满意",
        "没有让我失望",
        "适合继续复用",
        "已经达到预期",
        "对新手很友好",
        "值得继续优化",
    ]
    suffixes = [
        "，准备继续用一段时间。",
        "，今天终于顺利跑通了。",
        "，后面可以继续跟进。",
        "，比上次版本好不少。",
        "，适合日常使用。",
    ]

    texts = list(MANUAL_NORMAL_SEEDS)
    for subject in subjects:
        for aspect in aspects:
            for evaluation in evaluations[:3]:
                texts.append(f"{subject}{aspect}{evaluation}{suffixes[(len(texts) + len(subject)) % len(suffixes)]}")
    for subject in subjects[:8]:
        for evaluation in evaluations:
            texts.append(f"{subject}{evaluation}，整体反馈比较正面。")

    return deduplicate_texts(texts)[:NORMAL_TARGET]


def build_manual_ad_texts() -> list[str]:
    channels = ["加微信", "vx联系", "私信留联系方式", "扫码进群", "QQ咨询", "评论区留v", "加v", "加微"]
    offers = [
        "领取优惠券",
        "了解代理项目",
        "咨询兼职日结",
        "代写作业接急单",
        "购买涨粉引流服务",
        "加入返利活动",
        "低价拿货",
        "咨询清仓处理",
    ]
    urgencies = ["今天截止", "名额有限", "当天结算", "量大优惠", "稳定回本", "新手也能做", "手把手带", "长期可做"]
    openings = ["想赚钱的", "需要副业的", "做校园代理的", "想接单的", "想做拉新的", "需要推广的", "想做兼职的", "要找渠道的"]
    closings = ["有兴趣直接来问", "细节私聊再说", "当天就能开始", "适合学生和宝妈", "懂行的都知道", "不方便公开展开说"]

    texts = list(MANUAL_AD_SEEDS)
    for index, channel in enumerate(channels):
        for offer in offers:
            handle = make_contact_handle(index + len(offer))
            urgency = urgencies[(index + len(offer)) % len(urgencies)]
            texts.append(f"{channel}{handle}，{offer}，{urgency}。")
            texts.append(f"{offer}的可以{channel} {handle}，{urgency}。")
            texts.append(f"{channel}:{handle}，{offer}，{closings[(index + len(handle)) % len(closings)]}。")
    for opening in openings:
        for offer in offers:
            urgency = urgencies[(len(opening) + len(offer)) % len(urgencies)]
            closing = closings[(len(opening) + len(offer)) % len(closings)]
            texts.append(f"{opening}看我主页，{offer}，{urgency}。")
            texts.append(f"{opening}私聊我，{offer}，支持远程操作，{urgency}。")
            texts.append(f"{opening}可以来咨询，{offer}，{closing}。")
            texts.append(f"{offer}这边长期有，{opening}可做，{closing}。")

    return deduplicate_texts(texts)[:AD_TARGET]


def build_manual_sexual_texts() -> list[str]:
    invitations = [
        "私信聊更刺激的话题",
        "想约的话晚上来找我",
        "懂的来主页看图",
        "想看福利视频的私聊",
        "今晚可以单独约",
        "想看私密内容的来找我",
        "不方便公开聊的内容私下说",
        "想继续看图就私信我",
    ]
    hooks = ["只发熟人", "不方便公开说", "评论区别问太细", "看完记得删", "晚上更精彩", "主页还有更多", "懂的自己来", "私下再发完整的"]
    contents = ["擦边写真", "私密照片", "成人资源", "暧昧聊天", "福利视频", "成人视频", "私房图", "约会暗示"]
    openings = ["这个账号一直在发", "她主页全是", "评论区有人卖", "私聊里会继续发", "有人在群里推", "这个直播间总在暗示", "主页动态里都是", "最近老有人在私发"]

    texts = list(MANUAL_SEXUAL_SEEDS)
    for opening in openings:
        for content in contents:
            hook = hooks[(len(opening) + len(content)) % len(hooks)]
            texts.append(f"{opening}{content}，{hook}。")
            texts.append(f"{opening}{content}，想看的自己私聊，{hook}。")
    for invitation in invitations:
        for content in contents:
            hook = hooks[(len(invitation) + len(content)) % len(hooks)]
            texts.append(f"{invitation}，还能继续发{content}，{hook}。")
            texts.append(f"{invitation}，想看{content}就私聊，{hook}。")
            texts.append(f"{invitation}，这边有{content}，{hook}。")

    return deduplicate_texts(texts)[:SEXUAL_TARGET]


def load_cold_abuse_texts() -> list[str]:
    train_frame = pd.read_csv(f"{COLD_BASE_URL}/train.csv")
    dev_frame = pd.read_csv(f"{COLD_BASE_URL}/dev.csv")
    frame = pd.concat([train_frame, dev_frame], ignore_index=True)
    offensive = frame[frame["label"] == 1]["TEXT"].astype(str).tolist()

    cleaned = []
    for text in offensive:
        clean_text = normalize_text(text)
        if 6 <= len(clean_text) <= 80:
            cleaned.append(clean_text)

    random.Random(SEED).shuffle(cleaned)
    return deduplicate_texts(cleaned)[:ABUSE_TARGET]


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").replace("\n", " ").split()).strip()


def deduplicate_texts(texts: list[str]) -> list[str]:
    unique_texts: list[str] = []
    seen: set[str] = set()
    for text in texts:
        clean_text = normalize_text(text)
        if not clean_text or clean_text in seen:
            continue
        seen.add(clean_text)
        unique_texts.append(clean_text)
    return unique_texts


def build_dataset() -> pd.DataFrame:
    abuse_texts = load_cold_abuse_texts()
    normal_texts = build_manual_normal_texts()
    sexual_texts = build_manual_sexual_texts()
    ad_texts = build_manual_ad_texts()

    rows = (
        [{"text": text, "label": "abuse"} for text in abuse_texts]
        + [{"text": text, "label": "normal"} for text in normal_texts]
        + [{"text": text, "label": "sexual"} for text in sexual_texts]
        + [{"text": text, "label": "ad"} for text in ad_texts]
    )

    frame = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    return frame.sample(frac=1, random_state=SEED).reset_index(drop=True)


def split_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frame, temp_frame = train_test_split(
        frame,
        test_size=0.30,
        random_state=SEED,
        stratify=frame["label"],
    )
    dev_frame, test_frame = train_test_split(
        temp_frame,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_frame["label"],
    )
    return train_frame.reset_index(drop=True), dev_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def write_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_dataset()
    train_frame, dev_frame, test_frame = split_dataset(frame)

    train_frame.to_csv(DATA_DIR / "train.csv", index=False, encoding="utf-8")
    dev_frame.to_csv(DATA_DIR / "dev.csv", index=False, encoding="utf-8")
    test_frame.to_csv(DATA_DIR / "test.csv", index=False, encoding="utf-8")

    summary = {
        "total": int(len(frame)),
        "train": int(len(train_frame)),
        "dev": int(len(dev_frame)),
        "test": int(len(test_frame)),
        "label_distribution": frame["label"].value_counts().sort_index().to_dict(),
        "sources": {
            "abuse": "COLDataset offensive samples",
            "normal": "manual self-labeled interview examples",
            "sexual": "manual self-labeled interview examples",
            "ad": "manual self-labeled interview examples",
        },
    }

    with (DATA_DIR / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    write_dataset()
