# WeChat-Dialogue-History-Analysis-Agent-
memotrace的毛病是语音通话的时长这个记录没保留
为了工程实践，原材料需要重新映射和分类
flash-attn选用，显卡版本低的话，高的话就注意显卡驱动匹配。如果只需要适配某一个系列的显卡，可以限定在# 假设你只需要适配 RTX 50 系列 (sm_120)          export TORCH_CUDA_ARCH_LIST="12.0" 
huggingface toekn注册后
router和01run的det和rec问题
解决out of memory爆显存的问题
显存的碎片化处理，内存和显存的管理
撤回态

对了了多个模型的效果
openSMILE eGeMAPS
YAMNet 或 PANNs
中文 HuBERT + CASIA

加入社会动力学

本地载入abliterated时遇到阻碍，失败了，试了几种顺序和方案
换了 Q4_K_M.gguf

## 数据清洗和映射
### 语音模态的参数有
seq_in_html	msg_uid	MsgSvrID	token	ts	time_local	speaker	type	sub_type	modality	text_raw	media_path	voice_length	voice_to_text
先跑通五条语音，再跑全量数据
#### 
0) 在终端准备项目目录（一次性）

在哪个终端/路径： 随便，你只要能执行 shell（推荐直接 cd /data/wechatDHA/lwy）

cd /data/wechatDHA/lwy
mkdir -p scripts asr_out


之后所有脚本我都放在：/data/wechatDHA/lwy/scripts/
输出我都放在：/data/wechatDHA/lwy/asr_out/

1) 创建并激活 conda 环境（一次性）

在哪个终端/路径： 任意目录都行

conda create -n wechatDHA python=3.10 -y
conda activate wechatDHA

python -m pip install -U pip
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130

在哪个终端/路径： 任意目录都行（仍在 wechatDHA 环境中）

3.1 安装 faster-whisper（Whisper large-v3 推理用）
pip install -U faster-whisper

3.2 安装 FunASR + modelscope（用 ct-punc / paraformer）

FunASR 官方安装文档写了：pip3 install -U funasr，并且如果要用 ModelScope 的预训练模型，建议安装 modelscope。

pip install -U funasr modelscope

管线 A：faster-whisper（large-v3）→ ct-punc（加标点）
4A) 创建脚本：转写 5 条样本并输出 JSONL

在哪个终端/路径： cd /data/wechatDHA/lwy（推荐在项目根目录）

cd /data/wechatDHA/lwy
cat > scripts/run_5_whisper_ctpunc.py << 'PY'
import json
from pathlib import Path
from faster_whisper import WhisperModel
from funasr import AutoModel
from text_normalize import strip_punc, dedup_punc


VOICE_DIR = Path("/data/wechatDHA/lwy/voice")
OUT = Path("/data/wechatDHA/lwy/asr_out/asr_5_whisper_ctpunc.jsonl")

SAMPLES = [
  "20250624-000413-249048-1.mp3",
  "20250624-000452-183278-1.mp3",
  "20250707-010920-167012-1.mp3",
  "20250707-014930-883422-1.mp3",
  "20250708-000111-478053-1.mp3",
]

# ASR: faster-whisper 是 pip 包（CTranslate2 推理实现），适合本地批量跑:contentReference[oaicite:7]{index=7}
asr = WhisperModel("/data/models/faster-whisper-large-v3", device="cuda", compute_type="float16")

# 标点：ct-punc 模型可用于“纯文本加标点”或“ASR后处理”:contentReference[oaicite:8]{index=8}
punc = AutoModel(model="ct-punc")  # FunASR 教程中给出 AutoModel/命令行推理方式:contentReference[oaicite:9]{index=9}

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", encoding="utf-8") as f:
    for fn in SAMPLES:
        path = VOICE_DIR / fn
        if not path.exists():
            f.write(json.dumps({"file": fn, "error": "file_not_found", "path": str(path)}, ensure_ascii=False) + "\n")
            continue

        segments, info = asr.transcribe(
            str(path),
            beam_size=5,
            language="zh",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        raw_text = "".join(s["text"] for s in segs).strip()
        
        # 1) 繁->简（证据层保留 raw_text，不覆盖）
        raw_text_s = to_simplified(raw_text)
        
        # 2) 给 ct-punc 前清洗（去标点 + 简体）
        raw_for_punc, prep_meta = prepare_for_punc(raw_text, simplify=True)
        
        pres = punc.generate(input=raw_for_punc)  # ct-punc 作为 ASR 后处理标点模型 :contentReference[oaicite:5]{index=5}
        punct_text = pres[0].get("text") if isinstance(pres, list) and isinstance(pres[0], dict) and "text" in pres[0] else str(pres)
        punct_text = dedup_punc(punct_text)
        
        # 3) 补丁（专名/同音纠错），并记日志
        punct_text, patches = apply_patches(punct_text)
        
        # 4) 极保守的“假问号”修正（你 FunASR 第24条那种）
        punct_text, qfix = fix_false_question(punct_text)
        patches.extend(qfix)

        f.write(json.dumps({
            "file": fn,
            "engine": "faster-whisper large-v3 + ct-punc",
            "raw_text": raw_text,
            "punct_text": punct_text,
            "segments": segs,
        }, ensure_ascii=False) + "\n")

print("Wrote:", OUT)
PY


说明：ct-punc 的定位（文本/ASR 后处理）来自它的模型卡说明。
FunASR 教程明确给了 AutoModel 与 paraformer-zh + fsmn-vad + ct-punc 的组合方式（我们在管线 B 会用到一体化）。

5A) 运行脚本

在哪个终端/路径： /data/wechatDHA/lwy

cd /data/wechatDHA/lwy
python scripts/run_5_whisper_ctpunc.py


运行成功后你会得到：

输出文件：/data/wechatDHA/lwy/asr_out/asr_5_whisper_ctpunc.jsonl

遇到网络问题，连接魔法或者下载好以后放到指定的路径

本地下载
sudo apt update
sudo apt install git-lfs -y
git lfs install
在目录的/data/models/的cmd下
git clone https://huggingface.co/Systran/faster-whisper-large-v3



conda install -c conda-forge nvitop

管线 B：FunASR 一体化（paraformer-zh + fsmn-vad + ct-punc）

FunASR 教程明确给出了命令行写法：
funasr ++model=paraformer-zh ++vad_model="fsmn-vad" ++punc_model="ct-punc" ++input=...

我更推荐用 Python AutoModel 批量输出 JSONL（更容易与你的消息时间线做 join）。

4B) 创建脚本：一体化跑 5 条并输出 JSONL

在哪个终端/路径： /data/wechatDHA/lwy

cd /data/wechatDHA/lwy
cat > scripts/run_5_funasr_paraformer.py << 'PY'
import json
from pathlib import Path
from funasr import AutoModel

VOICE_DIR = Path("/data/wechatDHA/lwy/voice")
OUT = Path("/data/wechatDHA/lwy/asr_out/asr_5_funasr_paraformer.jsonl")

SAMPLES = [
  "20250624-000413-249048-1.mp3",
  "20250624-000452-183278-1.mp3",
  "20250707-010920-167012-1.mp3",
  "20250707-014930-883422-1.mp3",
  "20250708-000111-478053-1.mp3",
]

# FunASR 教程给出可自由组合 ASR/VAD/标点的方式:contentReference[oaicite:13]{index=13}
model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
)

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", encoding="utf-8") as f:
    for fn in SAMPLES:
        path = VOICE_DIR / fn
        if not path.exists():
            f.write(json.dumps({"file": fn, "error": "file_not_found", "path": str(path)}, ensure_ascii=False) + "\n")
            continue

        HOTWORD = "云顶之弈 金铲铲"
        res = model.generate(input=str(path), hotword=HOTWORD)
        f.write(json.dumps({
            "file": fn,
            "engine": "faster-whisper large-v3 + ct-punc",
            "raw_text": raw_text,
            "raw_text_s": raw_text_s,
            "raw_for_punc": raw_for_punc,
            "prep_meta": prep_meta,
            "punct_text": punct_text,
            "patches": patches,
            "segments": segs,
        }, ensure_ascii=False) + "\n")

print("Wrote:", OUT)
PY

5B) 运行脚本

在哪个终端/路径： /data/wechatDHA/lwy

cd /data/wechatDHA/lwy
python scripts/run_5_funasr_paraformer.py


你会得到：

输出文件：/data/wechatDHA/lwy/asr_out/asr_5_funasr_paraformer.jsonl

1) 修复管线A的“双标点”：给 ct-punc 前先做“去标点归一化”，之后再做“去重标点”
1.1 创建一个通用的文本清洗器脚本

在 /data/wechatDHA/lwy 终端执行：

cat > /data/wechatDHA/lwy/scripts/text_normalize.py << 'PY'
import re
from typing import Dict, List, Tuple, Optional

# --------
# OpenCC (繁->简)
# --------
_cc = None
def to_simplified(text: str) -> str:
    """
    Convert Traditional Chinese to Simplified Chinese using OpenCC if installed.
    OpenCC supports phrase-level conversion and regional idioms.  (Use OpenCC('t2s'))
    """
    global _cc
    t = (text or "").strip()
    if not t:
        return t
    try:
        if _cc is None:
            from opencc import OpenCC
            _cc = OpenCC("t2s")
        return _cc.convert(t)
    except Exception:
        # 若未安装 OpenCC 或运行异常，直接返回原文（不阻塞主流程）
        return t

# --------
# Punctuation normalize (给 ct-punc 前清洗 & 后去重)
# --------
PUNC_RE = re.compile(r'[，。？！；：、,.!?;:"“”‘’（）()【】\[\]{}<>《》…]+')

def strip_punc(text: str) -> str:
    """Remove existing punctuation BEFORE ct-punc to avoid double punctuation."""
    t = (text or "").strip()
    t = PUNC_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def dedup_punc(text: str) -> str:
    """Deduplicate repeated punctuation AFTER ct-punc."""
    t = (text or "").strip()
    t = re.sub(r'([，。？！；：、])\1+', r'\1', t)
    t = re.sub(r'([,.!?;:])\1+', r'\1', t)
    t = re.sub(r'[，,]{2,}', '，', t)
    t = re.sub(r'[。\.]{2,}', '。', t)
    return t

# --------
# Patch rules (可控纠错 + 记录日志)
# --------
DEFAULT_PATCH_MAP: Dict[str, str] = {
    # 你这轮确认的 FunASR/Whisper 专名错
    "云顶之翼": "云顶之弈",
    "金灿铲": "金铲铲",

    # 你前面样本里出现过的同音错（保留在这里，方便全局复用）
    "醒久了": "醒酒了",
    "很适很适": "很湿很湿",
}

def apply_patches(text: str, patch_map: Optional[Dict[str, str]] = None) -> Tuple[str, List[dict]]:
    """
    Apply string replacement patches and return (new_text, patches_log).
    patches_log: [{"from": "...", "to": "...", "count": n}, ...]
    """
    patch_map = patch_map or DEFAULT_PATCH_MAP
    t = text or ""
    logs: List[dict] = []
    for a, b in patch_map.items():
        if a in t:
            n = t.count(a)
            t = t.replace(a, b)
            logs.append({"from": a, "to": b, "count": n})
    return t, logs

# --------
# Conservative punctuation fix (专治“什么的。”被打成“什么的？”这类)
# --------
_SHENME_DE_PATTERNS = [
    re.compile(r"(什么的)\?$"),
    re.compile(r"(什么之类的)\?$"),
    re.compile(r"(之类的)\?$"),
]

def fix_false_question(text: str) -> Tuple[str, List[dict]]:
    """
    Very conservative fixer: only changes trailing '？' to '。' for patterns like '什么的？'
    We keep it narrow to avoid harming real questions.
    """
    t = (text or "").strip()
    logs: List[dict] = []
    if not t:
        return t, logs

    # 只处理句末的问号，且命中特定“非疑问语气”短语
    if t.endswith("?") or t.endswith("？"):
        # 统一用中文问号判断
        t_norm = t[:-1] + "？"
        for pat in _SHENME_DE_PATTERNS:
            if pat.search(t_norm):
                fixed = t_norm[:-1] + "。"
                logs.append({"rule": "fix_false_question_shenme_de", "from": t, "to": fixed})
                return fixed, logs

    return t, logs

# --------
# Pipeline helper
# --------
def prepare_for_punc(raw_text: str, simplify: bool = True) -> Tuple[str, dict]:
    """
    raw_text -> (clean_text_for_ct_punc, meta)
    meta includes whether simplified was applied.
    """
    t = raw_text or ""
    meta = {"simplified": False}
    if simplify:
        t2 = to_simplified(t)
        meta["simplified"] = (t2 != t)
        t = t2
    t = strip_punc(t)
    return t, meta
PY



解释：ct-punc 适合做“无标点文本 → 加标点”的后处理。
所以我们强制先 strip_punc()，再喂给 ct-punc，最后 dedup_punc() 收尾。

0) 进入目录并激活环境

终端路径：

cd /data/wechatDHA/lwy
conda activate wechatDHA
mkdir -p scripts picks

1) 创建脚本：遍历 voice 文件夹并输出“Python 列表文件”

目标：扫描 /data/wechatDHA/lwy/voice，把文件名排序后输出到：

/data/wechatDHA/lwy/picks/voice_all_names.py（里面就是 SAMPLES = [...]）

在 /data/wechatDHA/lwy 终端执行：

cat > /data/wechatDHA/lwy/scripts/list_voice_filenames.py << 'PY'
from pathlib import Path
import argparse

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}

def main():
    ap = argparse.ArgumentParser(description="List audio filenames in a folder and write a Python list file.")
    ap.add_argument("--voice_dir", default="/data/wechatDHA/lwy/voice", help="Voice folder path")
    ap.add_argument("--out", default="/data/wechatDHA/lwy/picks/voice_all_names.py", help="Output .py file path")
    ap.add_argument("--recursive", action="store_true", help="Use rglob (recursive)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = all)")
    args = ap.parse_args()

    voice_dir = Path(args.voice_dir)
    if not voice_dir.exists():
        raise SystemExit(f"voice_dir not found: {voice_dir}")

    it = voice_dir.rglob("*") if args.recursive else voice_dir.glob("*")
    files = [p for p in it if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

    # 只写 basename，方便你和 HTML 的 ./voice/xxx.mp3 对齐
    names = sorted({p.name for p in files})

    if args.limit and args.limit > 0:
        names = names[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated. Paste SAMPLES into your run_*.py if you want.\n")
        f.write("SAMPLES = [\n")
        for n in names:
            f.write(f'  "{n}",\n')
        f.write("]\n")

    print(f"Found {len(names)} audio files.")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
PY

2) 运行它：生成“全量文件名列表”

在 /data/wechatDHA/lwy 终端执行：

python scripts/list_voice_filenames.py \
  --voice_dir /data/wechatDHA/lwy/voice \
  --out /data/wechatDHA/lwy/picks/voice_all_names.py


如果你的 voice 目录里有子目录（例如按月份分文件夹），就加 --recursive：

python scripts/list_voice_filenames.py \
  --voice_dir /data/wechatDHA/lwy/voice \
  --out /data/wechatDHA/lwy/picks/voice_all_names.py \
  --recursive

3) 你如何把其中 30 条粘贴进 run_5_whisper_ctpunc.py
3.1 打开生成的列表文件看内容

终端路径：

cd /data/wechatDHA/lwy
sed -n '1,120p' picks/voice_all_names.py


你会看到类似：

SAMPLES = [
  "20250624-000413-249048-1.mp3",
  ...
]

3.2 你手工挑 30 条

最简单粗暴：直接复制其中 30 行（或你挑的 30 行）粘到 run_5_whisper_ctpunc.py 里的 SAMPLES = [...]。

如果你想先自动取“前 30 条作为临时样本”，你可以运行：

python scripts/list_voice_filenames.py \
  --voice_dir /data/wechatDHA/lwy/voice \
  --out /data/wechatDHA/lwy/picks/voice_30_names.py \
  --limit 30


1) 先把“繁体输出”工程化成稳定简体（保留证据、不覆盖原文）
为什么会出现繁体

Whisper 有时会输出繁体/简繁混杂（尤其在某些短句、某些词形上），这属于模型输出分布的一部分，不是你脚本问题。

最稳的处理方式

在 JSONL 里同时保存：

raw_text：Whisper 原始输出（证据层）

raw_text_s：把 raw_text 转成简体（给后续分析用）

punct_text / punct_text_s 同理

用 OpenCC 做简繁转换（支持字级/词组级转换）。

1.1 在哪个终端执行
cd /data/wechatDHA/lwy
conda activate wechatDHA

1.2 安装 OpenCC（一次）
pip install OpenCC


（PyPI 上的 OpenCC 包就是简繁转换库。）

### 表情模态
下载的分为可以直接显示的是webq
而浏览器叫下载的是gif文件
自动化下载表情成图片然后做映射

### 撤回消息的也得提取出来（英文是recalled，中文是撤回就行）
recall message 前面的speaker字段是other的话是对方撤回信息
而speaker是me的话则是我方撤回消息

### 引用信息
注意让agent提取引用信息时要记住索引引用的信息

### 匿名化
pip install pyyaml -q

### 存在的问题
中文的他/她和它的问题

### 视频模态
都是短视频
2. 使用 4-bit 量化 (4-bit Quantization)
这是目前在消费级显卡上跑大模型的核心黑科技。
原始状态 (FP16/BF16)：模型里的每一个参数（权重）原本占用 2 字节（16位）空间。对于 7B（70亿参数）模型，仅权重本身就需要约 14GB 显存，再加上视觉编码器（VL模型特有）和对话产生的上下文（KV Cache），16GB 显存会瞬间爆满。
量化后 (4-bit)：通过特殊算法（如 AWQ 或 BitsAndBytes），将原本 16 位的参数压缩到只有 4 位。
体积骤减：权重占用的显存从 14GB 压缩到约 4GB - 5GB。
性能权衡：虽然精度会有极其微小的损失，但在 2026 年的量化技术下，这种损失在实际对话和识图中几乎察觉不到。


### 原始数据转写
总体思路（以后换任何联系人都能复用）
HTML 是主时间线（唯一真相）：因为它带 timestamp/is_send/type/text(媒体路径)/voice_to_text 等字段，能把多模态稳定绑定回时间轴。  
CSV 主要用来做统计、校验（这个 CSV 没有 MsgSvrID 列名，所以更适合校验/统计，不做主键）。
Step 1：从 HTML 里抓出 chatMessages 数组（JSON）
导出 HTML 里有类似：
const chatMessages = [ {...}, {...}, ... ];
所以做法是：
读入 HTML 文本
用正则把 chatMessages = [ ... ]; 这段抠出来
json.loads() 解析成 Python list[dict]
按原顺序 enumerate() 保留 seq_in_html（同一秒多条消息也不乱序）
示例代码（最核心部分）：
```
import re, json, pathlib
html_text = pathlib.Path("李维彦.html").read_text(encoding="utf-8", errors="ignore")
m = re.search(r'const\s+chatMessages\s*=\s*(\[\s*.*?\s*\]);', html_text, re.S)
arr_text = m.group(1)
# 防止出现结尾多余逗号导致 JSON 解析失败
arr_clean = re.sub(r',(\s*[\]\}])', r'\1', arr_text)
messages = json.loads(arr_clean)  # list of dict
```
Step 2：逐条消息映射成统一 schema（并推断模态）
对每条 msg：
```
ts = msg["timestamp"]
speaker = "ME" if msg["is_send"]==1 else "OTHER"
msg_uid = "P1:"+MsgSvrID（稳定主键）
modality 由 type 推断：
1=text
3=image
34=voice
49=link_or_file
47=sticker
```
其它保留为 type_xx（不丢数据）
并把 text_raw / media_path 原样保存（媒体就是 ./voice/...mp3 这种相对路径）。
输出 JSONL（每行一条消息）：
```
import json, datetime
with open("P1_messages_raw.jsonl","w",encoding="utf-8") as f:
    for i, msg in enumerate(messages):
        out = {
            "seq_in_html": i,
            "MsgSvrID": msg.get("MsgSvrID",""),
            "msg_uid": f'P1:{msg.get("MsgSvrID","")}' or f'P1:seq:{i}',
            "ts": msg.get("timestamp"),
            "time_local": datetime.datetime.fromtimestamp(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            "speaker": "ME" if msg.get("is_send")==1 else "OTHER",
            "type": msg.get("type"),
            "text_raw": msg.get("text",""),
            "media_path": msg.get("text","") if str(msg.get("text","")).startswith("./") else None,
            "voice_length": msg.get("voice_length"),
            "voice_to_text": msg.get("voice_to_text"),
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
```
### 经过ai处理后的html文件中的msg uid是全的，如html文件中"MsgSvrID": "3360438420036727851"；但是如果双击用excel打开csv文件时处理时会把后面的数字截断变成 "3360438420036720000"
正确打开 CSV 的姿势（避免 Excel 自动吞精度）
不要双击打开，用导入：
Excel → 数据 → 从文本/CSV（Power Query）
在导入界面把“数据类型检测”设为：不检测 / 全部作为文本（或者导入后把该列改成 Text）
这样 19 位的 MsgSvrID 才不会被当数字吞掉
要保留长数字，应该用导入流程并把列格式设为 Text。

### 跑通voice转写
先抽出几个录音单独放一个文件夹
