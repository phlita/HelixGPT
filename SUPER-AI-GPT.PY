import math
import json
import sqlite3
import os
import re
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Iterable


# =====================================================================
# SUPER AI GPT - V1
# 身份离散 + 关系连续 + 阈值事实
# 支持外部 TXT 语料 / 分句 / 增量训练 / 持久化
# =====================================================================

PHASE_MAX = 360.0
LEGAL_HALF_WIDTH = 5.0
R_THRESHOLD = 0.97
R_ISOLATION = 0.80
STABLE_THRESHOLD = 1.0
LR_ALIGN = 0.5
LR_REPEL = 0.3

N_SLOTS_PER_CLASS = 10
SLOT_STEP = 36.0
N_PORTS = 256
CELL_SIZE = 5.0


def normalize_phase(a: float) -> float:
    return a % PHASE_MAX


def phase_diff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def resonance(phase_a: float, phase_b: float) -> float:
    delta = math.radians(phase_diff(phase_a, phase_b))
    return (1.0 + math.cos(delta)) / 2.0


def is_legal(phase: float, center: float) -> bool:
    return abs(phase_diff(phase, center)) <= LEGAL_HALF_WIDTH


# ---------------------------------------------------------------------
# 0. 语料加载器（新增）
# ---------------------------------------------------------------------
class CorpusLoader:
    """
    外部 TXT 语料加载器。
    支持：
    - 中文按字
    - 英文按词
    - 分句
    - 增量读取
    """

    # 中文句末标点
    CN_SENT_END = "。！？；…"
    # 英文句末标点
    EN_SENT_END = ".!?;"

    def __init__(self, min_len: int = 4, max_len: int = 200,
                 tokenize_mode: str = "char"):
        """
        min_len: 最短句长（字符数）
        max_len: 最长句长（超过则截断）
        tokenize_mode: "char" 按字 | "word" 按词 | "auto" 中英混合
        """
        self.min_len = min_len
        self.max_len = max_len
        self.tokenize_mode = tokenize_mode

    @staticmethod
    def read_txt(path: str, encoding: str = "utf-8") -> str:
        """读整个 txt 文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"语料文件不存在: {path}")
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            return f.read()

    @staticmethod
    def iter_txt(path: str, encoding: str = "utf-8") -> Iterable[str]:
        """按行迭代读取（适合大文件）"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"语料文件不存在: {path}")
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            for line in f:
                yield line

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """把长文本切成句子"""
        # 统一换行
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 按换行 + 句末标点切
        parts = re.split(r"[\n]+", text)
        sentences = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # 再用句末标点切
            buf = ""
            for ch in p:
                buf += ch
                if ch in cls.CN_SENT_END or ch in cls.EN_SENT_END:
                    s = buf.strip()
                    if s:
                        sentences.append(s)
                    buf = ""
            if buf.strip():
                sentences.append(buf.strip())
        return sentences

    def tokenize(self, sentence: str) -> List[str]:
        """把句子切成 token 序列"""
        if self.tokenize_mode == "char":
            return [c for c in sentence if not c.isspace()]
        elif self.tokenize_mode == "word":
            return sentence.split()
        else:  # auto
            tokens = []
            buf = ""
            for ch in sentence:
                if "\u4e00" <= ch <= "\u9fff":
                    # 中文：先 flush 英文缓冲，再单字成 token
                    if buf:
                        tokens.append(buf)
                        buf = ""
                    tokens.append(ch)
                elif ch.isalnum() or ch in "-_":
                    buf += ch
                else:
                    if buf:
                        tokens.append(buf)
                        buf = ""
                    if not ch.isspace():
                        tokens.append(ch)
            if buf:
                tokens.append(buf)
            return tokens

    def clean(self, sentence: str) -> Optional[str]:
        """清洗：去空白、过滤过短过长"""
        s = sentence.strip()
        if len(s) < self.min_len:
            return None
        if len(s) > self.max_len:
            s = s[:self.max_len]
        return s

    def load(self, path: str, encoding: str = "utf-8",
             max_sentences: int = None) -> List[str]:
        """加载并返回句子列表"""
        raw = self.read_txt(path, encoding)
        sentences = self.split_sentences(raw)
        result = []
        for s in sentences:
            c = self.clean(s)
            if c:
                result.append(c)
                if max_sentences and len(result) >= max_sentences:
                    break
        return result

    def load_tokens(self, path: str, encoding: str = "utf-8",
                    max_sentences: int = None) -> List[List[str]]:
        """加载并返回 token 列表的列表"""
        sentences = self.load(path, encoding, max_sentences)
        return [self.tokenize(s) for s in sentences]


# ---------------------------------------------------------------------
# 1. 空间索引
# ---------------------------------------------------------------------
class PhaseIndex:
    def __init__(self, cell_size: float = CELL_SIZE):
        self.cell_size = cell_size
        self.n_cells = int(PHASE_MAX / cell_size)
        self.cells: Dict[Tuple[str, int], List[Tuple[str, float]]] = defaultdict(list)

    def _cell(self, alpha: float) -> int:
        return int(normalize_phase(alpha) / self.cell_size) % self.n_cells

    def add(self, word: str, alpha: float, f: str):
        cell = self._cell(alpha)
        self.cells[(f, cell)].append((word, alpha))

    def query(self, alpha: float, f: str, radius: int = 2) -> List[Tuple[str, float]]:
        center_cell = self._cell(alpha)
        results = []
        for d in range(-radius, radius + 1):
            cell = (center_cell + d) % self.n_cells
            results.extend(self.cells.get((f, cell), []))
        return results


# ---------------------------------------------------------------------
# 2. 身份层
# ---------------------------------------------------------------------
class IdentityLayer:

    F_PRONOUN = "pronoun"
    F_NOUN = "noun"
    F_VERB = "verb"
    F_PUNCT = "punct"
    F_CONCEPT = "concept"
    F_ADJ = "adj"
    F_NUM = "num"
    F_EN = "en"          # 英文词
    F_OTHER = "other"

    def __init__(self):
        self.word_identity: Dict[str, Tuple[str, float, int]] = {}
        self.class_next_slot: Dict[str, int] = defaultdict(int)
        self.slot_next_port: Dict[Tuple[str, int], int] = defaultdict(int)
        self.index = PhaseIndex()

        self.PAD = "<PAD>"
        self.EOS = "<EOS>"
        self.UNK = "<UNK>"
        self._register(self.PAD, self.F_PUNCT)
        self._register(self.EOS, self.F_PUNCT)
        self._register(self.UNK, self.F_PUNCT)

    def _register(self, word: str, f: str):
        if word in self.word_identity:
            return
        slot = self.class_next_slot[f] % N_SLOTS_PER_CLASS
        alpha = normalize_phase(SLOT_STEP * slot)
        port = self.slot_next_port[(f, slot)] % N_PORTS
        self.word_identity[word] = (f, alpha, port)
        self.class_next_slot[f] += 1
        self.slot_next_port[(f, slot)] += 1
        self.index.add(word, alpha, f)

    def register(self, word: str, f: str = None):
        if word in self.word_identity:
            return
        if f is None:
            f = self._infer_class(word)
        self._register(word, f)

    def _infer_class(self, word: str) -> str:
        if word in "，。！？、；：,.!?;:":
            return self.F_PUNCT
        if word in "我你他她它":
            return self.F_PRONOUN
        if word in "是的有在会能可":
            return self.F_VERB
        if word.isdigit():
            return self.F_NUM
        if word in "的很大小高快慢":
            return self.F_ADJ
        # 英文词
        if re.match(r"^[A-Za-z][A-Za-z\-_]*$", word):
            return self.F_EN
        return self.F_NOUN

    def get_identity(self, word: str) -> Tuple[str, float, int]:
        if word not in self.word_identity:
            self.register(word)
        return self.word_identity[word]

    def decode_with_legal(self, phase: float, f: str) -> Optional[str]:
        candidates = self.index.query(phase, f)
        if not candidates:
            return None
        best_word, best_r = None, -1.0
        for word, center in candidates:
            r = resonance(phase, center)
            if r > best_r:
                best_r, best_word = r, word
        if best_r < R_THRESHOLD or best_word is None:
            return None
        center = self.word_identity[best_word][1]
        if not is_legal(phase, center):
            return None
        return best_word

    def all_classes(self) -> List[str]:
        return list(self.class_next_slot.keys())

    def save(self, path: str):
        data = {
            "word_identity": {
                w: [f, a, p] for w, (f, a, p) in self.word_identity.items()
            },
            "class_next_slot": dict(self.class_next_slot),
            "slot_next_port": {f"{k[0]}::{k[1]}": v
                               for k, v in self.slot_next_port.items()},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.word_identity.clear()
        self.index = PhaseIndex()
        for w, (f, a, p) in data["word_identity"].items():
            self.word_identity[w] = (f, a, p)
            self.index.add(w, a, f)
        self.class_next_slot = defaultdict(int, data["class_next_slot"])
        self.slot_next_port = defaultdict(int)
        for k, v in data["slot_next_port"].items():
            f, slot = k.split("::")
            self.slot_next_port[(f, int(slot))] = v


# ---------------------------------------------------------------------
# 3. 关系层
# ---------------------------------------------------------------------
class RelationState:
    def __init__(self):
        self.relation: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)

    def set_relation(self, word: str, f: str, alpha: float, A: float = 1.0):
        self.relation[word][f] = (normalize_phase(alpha), A)

    def get_relation(self, word: str, f: str) -> Tuple[float, float]:
        return self.relation[word].get(f, (0.0, 0.0))

    def project(self, word_a: str, word_b: str, f: str) -> float:
        a_alpha, a_A = self.get_relation(word_a, f)
        b_alpha, b_A = self.get_relation(word_b, f)
        return a_A * b_A * resonance(a_alpha, b_alpha)


# ---------------------------------------------------------------------
# 4. 多关系轴注意力
# ---------------------------------------------------------------------
class MultiAxisAttention:
    def __init__(self, ctx_len: int = 16, n_axes: int = 4):
        self.ctx_len = ctx_len
        self.n_axes = n_axes

    def compute_axes(self, past_words: List[str],
                     identity: IdentityLayer) -> List[Tuple[str, float]]:
        ctx = past_words[-self.ctx_len:]
        by_class: Dict[str, List[float]] = defaultdict(list)
        for w in ctx:
            if w not in identity.word_identity:
                identity.register(w)
            f, alpha, _ = identity.get_identity(w)
            by_class[f].append(alpha)

        axes = []
        for f, alphas in by_class.items():
            sum_x, sum_y = 0.0, 0.0
            for i, a in enumerate(alphas):
                weight = math.exp(i - len(alphas))
                rad = math.radians(a)
                sum_x += weight * math.cos(rad)
                sum_y += weight * math.sin(rad)
            if abs(sum_x) > 1e-9 or abs(sum_y) > 1e-9:
                axis = normalize_phase(math.degrees(math.atan2(sum_y, sum_x)))
                axes.append((f, axis))
        axes.sort(key=lambda x: -len(by_class[x[0]]))
        return axes[:self.n_axes]

    def compute_context_phase(self, past_words: List[str],
                              target_f: str,
                              identity: IdentityLayer,
                              relation: RelationState) -> float:
        ctx = past_words[-self.ctx_len:]
        sum_x, sum_y = 0.0, 0.0
        for i, w in enumerate(ctx):
            alpha, A = relation.get_relation(w, target_f)
            if A == 0.0:
                _, alpha, _ = identity.get_identity(w)
                A = 1.0
            weight = math.exp(i - len(ctx)) * A
            rad = math.radians(alpha)
            sum_x += weight * math.cos(rad)
            sum_y += weight * math.sin(rad)
        if abs(sum_x) < 1e-9 and abs(sum_y) < 1e-9:
            return 0.0
        return normalize_phase(math.degrees(math.atan2(sum_y, sum_x)))


# ---------------------------------------------------------------------
# 5. 学习层
# ---------------------------------------------------------------------
class LearningRule:
    @staticmethod
    def align(current: float, target: float, lr: float = LR_ALIGN) -> float:
        error = phase_diff(target, current)
        if abs(error) < STABLE_THRESHOLD:
            return current
        return normalize_phase(current + lr * error)

    @staticmethod
    def repel(current: float, wrong: float, lr: float = LR_REPEL) -> float:
        error = phase_diff(current, wrong)
        if abs(error) > 90.0:
            return current
        return normalize_phase(current + lr * error)


# ---------------------------------------------------------------------
# 6. OSD
# ---------------------------------------------------------------------
class OSD:
    def __init__(self, f: str, alpha: float, A: float = 1.0, p: int = 0,
                 type: str = "fact", route: str = None):
        self.f = f
        self.alpha = normalize_phase(alpha)
        self.A = A
        self.p = p
        self.type = type
        self.route = route

    def __repr__(self):
        return (f"OSD(f={self.f}, α={self.alpha:.2f}°, "
                f"A={self.A:.2f}, p={self.p}, type={self.type})")

    def to_tuple(self):
        return (self.f, self.alpha, self.A, self.p, self.type, self.route)

    @classmethod
    def from_tuple(cls, t):
        return cls(*t)


# ---------------------------------------------------------------------
# 7. OSD 持久化
# ---------------------------------------------------------------------
class OSDStore:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS osd (
                key TEXT PRIMARY KEY,
                f TEXT, alpha REAL, A REAL,
                p INTEGER, type TEXT, route TEXT
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_f_alpha ON osd (f, alpha)"
        )
        self.conn.commit()

    def save(self, key: str, osd: OSD):
        self.conn.execute(
            "INSERT OR REPLACE INTO osd VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key,) + osd.to_tuple()
        )

    def load(self, key: str) -> Optional[OSD]:
        row = self.conn.execute(
            "SELECT f, alpha, A, p, type, route FROM osd WHERE key=?",
            (key,)
        ).fetchone()
        if row:
            return OSD(*row)
        return None

    def bulk_save(self, items: Iterable[Tuple[str, OSD]]):
        self.conn.executemany(
            "INSERT OR REPLACE INTO osd VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(k,) + osd.to_tuple() for k, osd in items]
        )
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM osd").fetchone()[0]

    def save_ars(self, ars: "ActiveRelationSet"):
        self.conn.execute("DROP TABLE IF EXISTS ars")
        self.conn.execute("""
            CREATE TABLE ars (
                f TEXT, ctx_phase REAL, twist REAL, weight REAL
            )
        """)
        rows = []
        for f, items in ars.store.items():
            for ctx_phase, twist, w in items:
                rows.append((f, ctx_phase, twist, w))
        self.conn.executemany("INSERT INTO ars VALUES (?, ?, ?, ?)", rows)
        self.conn.commit()

    def load_ars(self, ars: "ActiveRelationSet"):
        ars.store.clear()
        for f, ctx_phase, twist, w in self.conn.execute(
                "SELECT f, ctx_phase, twist, weight FROM ars"):
            ars.store[f].append((ctx_phase, twist, w))


# ---------------------------------------------------------------------
# 8. ARS
# ---------------------------------------------------------------------
class ActiveRelationSet:
    def __init__(self):
        self.store: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
        self.index: Dict[str, PhaseIndex] = defaultdict(PhaseIndex)

    def add(self, f: str, ctx_phase: float, twist: float, weight: float = 1.0):
        self.store[f].append((ctx_phase, twist, weight))
        self.index[f].add(str(len(self.store[f]) - 1), ctx_phase, f)

    def query(self, f: str, ctx_phase: float) -> Optional[float]:
        candidates = self.index[f].query(ctx_phase, f, radius=2)
        if not candidates:
            return None
        best_twist, best_r = None, -1.0
        for idx_str, ctx_p in candidates:
            idx = int(idx_str)
            if idx < len(self.store[f]):
                _, twist, w = self.store[f][idx]
                r = resonance(ctx_phase, ctx_p) * w
                if r > best_r:
                    best_r, best_twist = r, twist
        if best_r < R_ISOLATION:
            return None
        return best_twist

    def stats(self):
        return {f: len(v) for f, v in self.store.items()}

    def rebuild_index(self):
        """从 store 重建索引（加载后需要调用）"""
        self.index = defaultdict(PhaseIndex)
        for f, items in self.store.items():
            for i, (ctx_phase, _, _) in enumerate(items):
                self.index[f].add(str(i), ctx_phase, f)


# ---------------------------------------------------------------------
# 9. 主模型
# ---------------------------------------------------------------------
class SuperAIGPTV4:

    def __init__(self, ctx_len: int = 16, n_axes: int = 4,
                 osd_path: str = ":memory:"):
        self.identity = IdentityLayer()
        self.relation = RelationState()
        self.attention = MultiAxisAttention(ctx_len, n_axes)
        self.ars = ActiveRelationSet()
        self.osd_store = OSDStore(osd_path)
        self.ctx_len = ctx_len

    # -------- 训练单条 token 序列 --------
    def train_tokens(self, tokens: List[str],
                     lr_align: float = LR_ALIGN,
                     lr_repel: float = LR_REPEL):
        if len(tokens) < 2:
            return
        words = tokens + [self.identity.EOS]

        for i in range(1, len(words)):
            context_words = words[:i]
            target_word = words[i]

            self.identity.register(target_word)
            f, target_center, target_port = self.identity.get_identity(target_word)

            ctx_phase = self.attention.compute_context_phase(
                context_words, f, self.identity, self.relation
            )

            if f not in self.relation.relation.get(target_word, {}):
                self.relation.set_relation(target_word, f, target_center, 1.0)

            twist = phase_diff(target_center, ctx_phase)

            old_twist = self.ars.query(f, ctx_phase)
            if old_twist is None:
                new_twist = twist
            else:
                new_twist = LearningRule.align(old_twist, twist, lr_align)

                pred_phase = normalize_phase(ctx_phase + old_twist)
                pred_word = self.identity.decode_with_legal(pred_phase, f)
                if pred_word is not None and pred_word != target_word:
                    wrong_center = self.identity.get_identity(pred_word)[1]
                    new_twist = LearningRule.repel(new_twist, wrong_center, lr_repel)

            self.ars.add(f, ctx_phase, new_twist)
            self.relation.set_relation(
                target_word, f,
                normalize_phase(ctx_phase + new_twist),
                1.0
            )

            self.osd_store.save(
                f"{f}:{target_word}",
                OSD(f=f, alpha=target_center, A=1.0,
                    p=target_port, type="fact", route=target_word)
            )

    # -------- 训练单条文本 --------
    def train_step(self, text: str, lr_align: float = LR_ALIGN,
                   lr_repel: float = LR_REPEL):
        self.train_tokens(list(text), lr_align, lr_repel)

    # -------- 批量训练 --------
    def train_corpus(self, corpus: List[str], epochs: int = 10,
                     verbose: bool = True, save_every: int = 0,
                     save_dir: str = None):
        for epoch in range(epochs):
            random.shuffle(corpus)
            for text in corpus:
                self.train_step(text)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch+1}/{epochs} | "
                      f"词数={len(self.identity.word_identity)} | "
                      f"ARS={sum(self.ars.stats().values())} | "
                      f"OSD={self.osd_store.count()}")
            if save_every and save_dir and (epoch + 1) % save_every == 0:
                self.save(save_dir)

    # -------- 增量训练：从 TXT 文件 --------
    def train_from_file(self, txt_path: str, epochs: int = 10,
                        tokenize_mode: str = "char",
                        max_sentences: int = None,
                        batch_size: int = 1000,
                        verbose: bool = True,
                        save_every: int = 0,
                        save_dir: str = None):
        """
        从外部 TXT 文件训练。
        支持大文件分批加载。
        """
        loader = CorpusLoader(tokenize_mode=tokenize_mode)
        print(f"[CorpusLoader] 读取 {txt_path} ...")
        sentences = loader.load(txt_path, max_sentences=max_sentences)
        print(f"[CorpusLoader] 有效句子数: {len(sentences)}")

        if not sentences:
            print("[CorpusLoader] 警告：没有有效句子")
            return

        token_seqs = [loader.tokenize(s) for s in sentences]
        token_seqs = [t for t in token_seqs if len(t) >= 2]
        print(f"[CorpusLoader] 有效 token 序列数: {len(token_seqs)}")

        for epoch in range(epochs):
            random.shuffle(token_seqs)
            for i in range(0, len(token_seqs), batch_size):
                batch = token_seqs[i:i + batch_size]
                for tokens in batch:
                    self.train_tokens(tokens)
                if verbose:
                    print(f"  Epoch {epoch+1}/{epochs} | "
                          f"batch {i//batch_size + 1} | "
                          f"词数={len(self.identity.word_identity)} | "
                          f"ARS={sum(self.ars.stats().values())}")

            if save_every and save_dir and (epoch + 1) % save_every == 0:
                self.save(save_dir)

    # -------- 前向 --------
    def forward(self, context_words: List[str], f: str) -> Tuple[float, float]:
        ctx_phase = self.attention.compute_context_phase(
            context_words, f, self.identity, self.relation
        )
        twist = self.ars.query(f, ctx_phase)
        if twist is None:
            return ctx_phase, ctx_phase
        pred_phase = normalize_phase(ctx_phase + twist)
        return pred_phase, ctx_phase

    # -------- 生成 --------
    def generate(self, prompt: str, max_length: int = 50,
                 tokenize_mode: str = "char") -> str:
        loader = CorpusLoader(tokenize_mode=tokenize_mode)
        words = loader.tokenize(prompt)
        output = ""

        for _ in range(max_length):
            if words:
                last = words[-1]
                if last not in self.identity.word_identity:
                    self.identity.register(last)
                f, _, _ = self.identity.get_identity(last)
            else:
                f = self.identity.F_NOUN

            pred_phase, _ = self.forward(words, f)
            next_word = self.identity.decode_with_legal(pred_phase, f)

            if next_word is None:
                for alt_f in self.identity.all_classes():
                    if alt_f == f:
                        continue
                    alt_word = self.identity.decode_with_legal(pred_phase, alt_f)
                    if alt_word is not None:
                        next_word = alt_word
                        f = alt_f
                        break

            if next_word is None or next_word == self.identity.EOS:
                break

            output += next_word
            words.append(next_word)

        return output

    # -------- 持久化 --------
    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        self.identity.save(os.path.join(dir_path, "identity.json"))
        self.osd_store.save_ars(self.ars)
        print(f"[V4] 已保存到 {dir_path}")

    def load(self, dir_path: str):
        self.identity.load(os.path.join(dir_path, "identity.json"))
        self.osd_store.load_ars(self.ars)
        self.ars.rebuild_index()   # 重建索引
        print(f"[V4] 已从 {dir_path} 加载")


# =====================================================================
# 实验：从外部 TXT 训练
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SUPER AI GPT - V4.1")
    print("外部 TXT 语料训练")
    print("=" * 70)

    # ---------- 配置 ----------
    TXT_PATH = "corpus.txt"       # 你的训练素材
    TOKENIZE_MODE = "char"        # "char" / "word" / "auto"
    EPOCHS = 10
    CTX_LEN = 16
    N_AXES = 4
    MAX_SENTENCES = None          # None = 全部
    SAVE_DIR = "./v4_checkpoint"

    # ---------- 初始化 ----------
    model = SuperAIGPTV4(
        ctx_len=CTX_LEN,
        n_axes=N_AXES,
        osd_path="v4_osd.db"
    )

    # ---------- 训练 ----------
    if os.path.exists(TXT_PATH):
        model.train_from_file(
            txt_path=TXT_PATH,
            epochs=EPOCHS,
            tokenize_mode=TOKENIZE_MODE,
            max_sentences=MAX_SENTENCES,
            batch_size=1000,
            verbose=True,
            save_every=5,
            save_dir=SAVE_DIR,
        )
    else:
        print(f"[警告] 找不到 {TXT_PATH}，使用内置小语料演示。")
        demo = [
            "你好我是基于统一场论驱动的智能系统",
            "世界的本质是空间以光速螺旋运动",
            "万有引力是空间在二维平面旋转的向心力",
            "电荷是一维空间流体的拓扑漏损",
        ]
        model.train_corpus(demo, epochs=20, verbose=True)

    # ---------- 保存 ----------
    model.save(SAVE_DIR)

    # ---------- 统计 ----------
    print(f"\n[系统] 身份层词数: {len(model.identity.word_identity)}")
    print(f"[系统] OSD 存储条数: {model.osd_store.count()}")
    print(f"[系统] ARS 统计: {model.ars.stats()}")

    # ---------- 对话 ----------
    print("\n" + "-" * 50)
    print("对话测试")
    print("-" * 50)

    questions = [
        "你好",
        "世界的本质",
        "万有引力是",
        "电荷是",
    ]

    for q in questions:
        print(f"\n[用户]: {q}")
        print(f"[AI  ]: {model.generate(q, max_length=30,
                                       tokenize_mode=TOKENIZE_MODE)}")
