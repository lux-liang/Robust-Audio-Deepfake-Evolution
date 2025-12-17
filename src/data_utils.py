import numpy as np
import soundfile as sf
import torch
import random
import torchaudio
import torchaudio.transforms as T
try:
    import librosa
except ImportError:
    librosa = None
    print("⚠️  Warning: librosa not found. Resampling will be skipped.")
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset

# 尝试导入 RawBoost，如果没有则给出警告
try:
    from rawboost import RawBoost
except ImportError:
    print("⚠️  Warning: rawboost.py not found. RawBoost will be disabled.")
    RawBoost = None

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


# --- Codec Augmentation (Simulated) ---
# Since we might not have ffmpeg installed, we simulate codec artifacts
# via downsampling/upsampling which mimics high-frequency loss (like MP3/AAC).
# This is a standard "poor man's codec" augmentation.
def apply_codec_aug(waveform_np, sample_rate=16000):
    # Input: numpy array (Time,)
    # Output: numpy array (Time,)
    
    if random.random() < 0.5: # default 50% probability (controlled by dataset flag)
        # Convert to tensor for torchaudio transforms
        if isinstance(waveform_np, np.ndarray):
            sig = torch.from_numpy(waveform_np).float().unsqueeze(0) # (1, T)
        else:
            sig = waveform_np
            
        # Randomly choose a lower sample rate to simulate bandwidth loss
        # MP3/AAC often cutoff > 16kHz content, or even lower for low bitrates.
        # 16k input -> downsample to 8k/6k/4k -> upsample back
        target_sr = random.choice([8000, 6000, 4000]) 
        
        # Note: T.Resample might be slow if re-instantiated every time.
        # But for randomness we need to change params.
        # Optimization: Pre-define resamplers if too slow.
        resampler_down = T.Resample(sample_rate, target_sr)
        resampler_up = T.Resample(target_sr, sample_rate)
        
        try:
            aug_sig = resampler_up(resampler_down(sig))
            return aug_sig.squeeze(0).numpy()
        except Exception:
            return waveform_np # Fallback if error
            
    return waveform_np
# --------------------------------------

def genSpoof_list(dir_meta, is_train=False, is_eval=False, is_2021=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    # ASVspoof 2021 格式：协议文件有多列，文件名在第1列（索引1）
    # 格式示例: "LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof ..."
    if is_2021:
        for line in l_meta:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            # 按空格分割，文件名在第1列（索引1）
            parts = line.split()
            if len(parts) >= 2:
                key = parts[1]  # 第1列是文件名（如 DF_E_2000011）
                file_list.append(key)
            else:
                # 兼容格式：如果只有一列，则直接使用（向后兼容）
                file_list.append(parts[0])
        return file_list

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, algo=0, use_codec=False, codec_p: float = 0.5, rawboost_p: float = 1.0):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.algo = algo
        self.use_codec = use_codec
        self.codec_p = float(codec_p)
        self.rawboost_p = float(rawboost_p)
        self.rawboost = None
        
        # 初始化 RawBoost 处理器
        if self.algo != 0:
            if RawBoost is None:
                print(f"⚠️  Warning: RawBoost not available, but algo={self.algo} was requested. RawBoost will be disabled.")
            else:
                # RawBoost algo codes: 
                # 0=None, 1=LNL, 2=ISD, 3=SSI, 4=LNL+ISD, 5=随机选择(1,2,3,4)
                # algo=5 是压缩编码增强，专治 A19 (Unknown VC attacks)
                if self.algo == 5:
                    # 随机选择一种增强方法，增加多样性
                    # RawBoost 的 process 方法会在每次调用时从 algo_id 列表中随机选择
                    self.rawboost = RawBoost(algo_id=[1, 2, 3, 4], fs=16000)
                else:
                    # 使用指定的增强方法
                    self.rawboost = RawBoost(algo_id=[self.algo], fs=16000)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        
        # 1. Apply RawBoost (Data Augmentation)
        # RawBoost 是主要的数据增强方法，对 A19 攻击特别有效
        if self.algo != 0 and self.rawboost is not None and random.random() < self.rawboost_p:
            try:
                X = self.rawboost.process(X)
            except Exception as e:
                print(f"⚠️  Warning: RawBoost processing failed: {e}. Using original audio.")
                # 如果 RawBoost 处理失败，使用原始音频
            
        # 2. Apply Codec Augmentation (Additional Robustness)
        # Codec 增强可以作为锦上添花，模拟 MP3/AAC 压缩伪影
        if self.use_codec and random.random() < self.codec_p:
            X = apply_codec_aug(X)
            
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key


class Dataset_ASVspoof2021_eval(Dataset):
    """Dataset loader for ASVspoof 2021 evaluation set (DF track)"""
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs: list of strings (each string: utt key)"""
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # ASVspoof 2021 文件格式：直接使用key作为文件名
        try:
            X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        except Exception as e:
            # 如果读取失败，返回零填充的音频
            print(f"⚠️  Warning: Failed to read {key}.flac: {e}. Using zero-padded audio.")
            X = np.zeros(self.cut, dtype=np.float32)
        
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key


# In-the-Wild evaluation dataset (uses meta.csv with filename/label)
class Dataset_InTheWild(Dataset):
    """
    In-the-Wild dataset loader.
    Expects a meta.csv with columns: file,speaker,label
    Audio files are stored directly under base_dir (e.g., 0.wav, 1.wav, ...).
    Labels: bona-fide / spoof
    """
    def __init__(self, meta_csv, base_dir, sample_rate: int = 16000):
        import pandas as pd
        self.base_dir = Path(base_dir)
        self.cut = 64600  # ~4 seconds at 16 kHz
        self.sample_rate = sample_rate
        df = pd.read_csv(meta_csv)
        required_cols = {"file", "label"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"meta.csv 必须包含列: {required_cols}, 当前列: {df.columns}")
        # store needed columns only
        self.files = df["file"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fname = self.files[index]
        label_str = self.labels[index]
        label = 0 if label_str.lower() == "bona-fide" else 1
        wav_path = self.base_dir / fname
        try:
            X, sr = sf.read(str(wav_path))
            # resample if needed (only if librosa is available)
            if sr != self.sample_rate and librosa is not None:
                X = librosa.resample(X, orig_sr=sr, target_sr=self.sample_rate)
        except Exception as e:
            print(f"⚠️  Warning: Failed to read {wav_path}: {e}. Using zero-padded audio.")
            X = np.zeros(self.cut, dtype=np.float32)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, label, fname
