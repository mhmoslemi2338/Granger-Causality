# model.py
# ---------------------------------------------------------
# Training with your ETT-style data factory + loaders.
# Uses:
#   - data_provider(args, flag) from your data factory
#   - Dataset_ETT_* shapes: (seq_x, seq_y, seq_x_mark, seq_y_mark)
#   - MLP forecaster (feed-forward) for simplicity
#   - Differentiable ridge-VAR extractor E on FUTURE window
#   - GC-skeleton loss (time-domain) + optional CIG loss (frequency-domain)
# ---------------------------------------------------------


# =========================
# ---- import your data factory
# =========================


import argparse
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.fft as fft
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,

}

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


# =========================
# ---- Model: simple feed-forward MLP
# =========================
class MLPForecaster(nn.Module):
    """
    Maps flattened context (B, C*d) -> flattened future (B, H*d).
    """
    def __init__(self, d: int, context_len: int, horizon: int,
                 hidden: int = 512, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d = d
        self.C = context_len
        self.H = horizon

        in_dim = self.C * d
        out_dim = self.H * d
        dims = [in_dim] + [hidden] * max(0, depth - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        # ctx: (B, C, d) -> pred: (B, H, d)
        B, C, d = ctx.shape
        y = self.net(ctx.reshape(B, C * d))
        return y.view(B, -1, d)


# =========================
# ---- Differentiable ridge-VAR extractor E
# =========================
def ridge_var(
    Y: torch.Tensor,
    p: int = 2,
    alpha: float = 1e-2,
    standardize: bool = True,
    add_intercept: bool = True,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Ridge-regularized VAR(p) fit on a future window Y (H, d). Differentiable.
    Returns A (p,d,d), Sigma (d,d), residuals, etc.
    """
    assert Y.dim() == 2, "Y must be (H, d)"
    H, d = Y.shape
    assert H > p, "Need H > p to fit a VAR(p)"
    device, dtype = Y.device, Y.dtype

    if standardize:
        mu = Y.mean(dim=0, keepdim=True)
        std = Y.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
        Yz = (Y - mu) / std
    else:
        mu = std = None
        Yz = Y

    Y_resp = Yz[p:, :]  # (H-p, d)
    X_lags = [Yz[p - k: H - k, :] for k in range(1, p + 1)]
    X_reg = torch.cat(X_lags, dim=1)  # (H-p, p*d)

    if add_intercept:
        ones = torch.ones((H - p, 1), device=device, dtype=dtype)
        X_reg = torch.cat([X_reg, ones], dim=1)  # (H-p, p*d + 1)

    G = X_reg.T @ X_reg + alpha * torch.eye(X_reg.shape[1], device=device, dtype=dtype)
    B = X_reg.T @ Y_resp

    L = torch.linalg.cholesky(G)
    A_full = torch.cholesky_solve(B, L)  # solves G A = B

    if add_intercept:
        A_flat = A_full[:-1, :]
        c = A_full[-1, :]
    else:
        A_flat = A_full
        c = torch.zeros(d, device=device, dtype=dtype)

    A = A_flat.reshape(p, d, d)  # (p, d, d)

    Y_hat = X_reg @ A_full
    R = Y_resp - Y_hat
    Sigma = (R.T @ R) / (H - p)

    return {"A": A, "Sigma": Sigma, "R": R, "X_reg": X_reg, "Y_resp": Y_resp, "c": c,
            "mu": None if not standardize else mu.squeeze(0),
            "std": None if not standardize else std.squeeze(0)}


# =========================
# ---- GC-skeleton helper functions
# =========================
def gc_strengths(A: torch.Tensor) -> torch.Tensor:
    g = torch.sqrt((A ** 2).sum(dim=0))  # (d,d)
    mask = 1.0 - torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
    g_masked = g * mask
    return g_masked



def soft_mask(g: torch.Tensor, tau: float, beta: float) -> torch.Tensor:
    return torch.sigmoid(beta * (g - tau))

def choose_tau(g_real: torch.Tensor, quantile: float = 0.3) -> float:
    d = g_real.shape[0]
    off = g_real[~torch.eye(d, dtype=torch.bool, device=g_real.device)]
    q = torch.tensor(quantile, device=g_real.device, dtype=g_real.dtype)
    return torch.quantile(off, q).item()

def gc_mask_l2_loss(S_pred: torch.Tensor, S_real: torch.Tensor) -> torch.Tensor:
    return ((S_pred - S_real) ** 2).sum()


# =========================
# ---- Spectral/CIG helpers (batched)
# =========================
def batched_spectrum_inverse_partial_coherence(
    A: torch.Tensor,
    Sigma: torch.Tensor,
    M: int = 256,
    eps_s: float = 1e-8,
    eps_inv: float = 1e-5,
):
    """
    A: (B,p,d,d), Sigma: (B,d,d)
    Returns: S_w (B,M,d,d), Theta_w (B,M,d,d), gamma2 (B,M,d,d) with γ^2 in [0,1].
    """
    if A.dim() == 3:   A = A.unsqueeze(0)
    if Sigma.dim() == 2: Sigma = Sigma.unsqueeze(0)

    B, p, d, _ = A.shape
    device = A.device
    ctype = torch.complex128 if A.dtype == torch.float64 else torch.complex64

    A_seq = torch.zeros((B, M, d, d), dtype=ctype, device=device)
    eye_c = torch.eye(d, dtype=ctype, device=device).expand(B, d, d)
    A_seq[:, 0] = eye_c
    A_c = A.to(dtype=ctype)
    for k in range(p):
        A_seq[:, k + 1] = -A_c[:, k]


    H_w = fft.fft(A_seq.to('cpu'), dim=1).to(A.device)                                # (B,M,d,d)
    I_w = torch.eye(d, dtype=ctype, device=device).expand(B, M, d, d)
    H_inv = torch.linalg.solve(H_w, I_w)                       # (B,M,d,d)

    Sigma_c = Sigma.to(dtype=ctype)
    S_w = H_inv @ Sigma_c.unsqueeze(1) @ H_inv.conj().transpose(-1, -2)
    if eps_s > 0:
        S_w = S_w + eps_s * I_w
    Theta_w = torch.linalg.inv(S_w + eps_inv * I_w)

    num = (Theta_w.abs() ** 2)
    diag = Theta_w.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-12)
    denom = diag.unsqueeze(-1) * diag.unsqueeze(-2)
    gamma2 = (num / denom).real.clamp_(0.0, 1.0)
    return S_w, Theta_w, gamma2


# =========================
# ---- Combined GC + CIG loss on FUTURE windows
# =========================
def gc_and_cig_loss_from_windows(
    Y_real: torch.Tensor,        # (B, H, d)
    Y_pred: torch.Tensor,        # (B, H, d)
    p: int = 2,
    alpha: float = 1e-2,
    beta: float = 4.0,
    tau_quantile: float = 0.4,
    use_cig: bool = True,
    cig_lambda: float = 0.1,
    M: int = 256,
    eps_s: float = 1e-8,
    eps_inv: float = 1e-5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    End-to-end loss combining:
      - GC-skeleton soft-mask L2 (time-domain) averaged over batch
      - optional CIG partial-coherence L1 (frequency-domain), batched
    """
    B, H, d = Y_real.shape
    device = Y_real.device

    gc_total = 0.0
    diag = {}
    A_real_list, A_pred_list, Sig_real_list, Sig_pred_list = [], [], [], []

    for b in range(B):
        Er = ridge_var(Y_real[b], p=p, alpha=alpha, standardize=True, add_intercept=True)
        Ep = ridge_var(Y_pred[b], p=p, alpha=alpha, standardize=True, add_intercept=True)

        A_r, A_p = Er["A"], Ep["A"]
        g_r = gc_strengths(A_r)
        g_p = gc_strengths(A_p)

        tau = choose_tau(g_r, quantile=tau_quantile)
        S_r = soft_mask(g_r, tau=tau, beta=beta)
        S_p = soft_mask(g_p, tau=tau, beta=beta)

        gc_total = gc_total + gc_mask_l2_loss(S_p, S_r)

        A_real_list.append(A_r)
        A_pred_list.append(A_p)
        Sig_real_list.append(Er["Sigma"])
        Sig_pred_list.append(Ep["Sigma"])

        if b == 0:
            diag.update({
                "g_real": g_r.detach(),
                "g_pred": g_p.detach(),
                "S_real": S_r.detach(),
                "S_pred": S_p.detach(),
                "tau": torch.tensor(tau, device=device),
            })

    gc_loss = gc_total / B

    if use_cig:
        A1 = torch.stack(A_real_list, dim=0)   # (B,p,d,d)
        A2 = torch.stack(A_pred_list, dim=0)   # (B,p,d,d)
        S1 = torch.stack(Sig_real_list, dim=0) # (B,d,d)
        S2 = torch.stack(Sig_pred_list, dim=0) # (B,d,d)

        _, _, gamma1 = batched_spectrum_inverse_partial_coherence(A1, S1, M, eps_s, eps_inv)
        _, _, gamma2 = batched_spectrum_inverse_partial_coherence(A2, S2, M, eps_s, eps_inv)

        off = ~torch.eye(d, dtype=torch.bool, device=device)
        cig_loss = (gamma1 - gamma2).abs()[..., off].mean()
        diag["cig_loss"] = cig_loss.detach()
    else:
        cig_loss = torch.tensor(0.0, device=device)

    total_loss = gc_loss + cig_lambda * cig_loss
    diag.update({"gc_loss": gc_loss.detach(), "total_loss": total_loss.detach()})
    return total_loss, diag


# =========================
# ---- Training loop integrated with your data_provider
# =========================
def train_with_factory(
    args,
    hidden: int = 512,
    depth: int = 2,
    dropout: float = 0.0,
    ridge_p: int = 3,
    ridge_alpha: float = 1e-2,
    gc_lambda: float = 0.1,
    gc_beta: float = 4.0,
    gc_tau_q: float = 0.3,
    use_cig: bool = True,
    cig_lambda: float = 0.05,
    cig_M: int = 256,
    cig_eps_s: float = 1e-8,
    cig_eps_inv: float = 1e-5,
    device: Optional[str] = None,
):
    """
    Uses your data_provider(args, flag) to get loaders for train/val/test.
    Expects args to contain fields used by data_provider (root_path, data_path, etc.)
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # get loaders
    train_data, train_loader = data_provider(args, flag='train')
    vali_data,  vali_loader  = data_provider(args, flag='val')
    test_data,  test_loader  = data_provider(args, flag='test')

    # infer dims from one batch
    sample = next(iter(train_loader))
    seq_x, seq_y, seq_x_mark, seq_y_mark = sample
    B, C, d = seq_x.shape
    H = args.pred_len
    assert seq_y.shape[1] == args.label_len + args.pred_len, "seq_y layout mismatch."

    model = MLPForecaster(d=d, context_len=C, horizon=H,
                          hidden=hidden, depth=depth, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = None  # (you can wrap amp if you want)

    def run_epoch(loader, training: bool):
        if training:
            model.train()
        else:
            model.eval()
        mse_running, gc_running, cig_running = 0.0, 0.0, 0.0
        n_batches = 0

        for seq_x, seq_y, seq_x_mark, seq_y_mark in loader:
            n_batches += 1
            ctx = seq_x.to(dev).float()                   # (B,C,d)
            fut_full = seq_y.to(dev).float()              # (B,label_len+H,d)
            fut = fut_full[:, -H:, :]                     # (B,H,d) -> ground truth future

            pred = model(ctx)                             # (B,H,d)

            mse = ((pred - fut) ** 2).mean()
            struct_loss, diag = gc_and_cig_loss_from_windows(
                Y_real=fut, Y_pred=pred,
                p=ridge_p, alpha=ridge_alpha,
                beta=gc_beta, tau_quantile=gc_tau_q,
                use_cig=use_cig, cig_lambda=cig_lambda,
                M=cig_M, eps_s=cig_eps_s, eps_inv=cig_eps_inv
            )
            # struct_loss already = gc + λ*cig; keep separate logs by reading diag
            total = mse + struct_loss

            if training:
                opt.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            mse_running += mse.item()
            gc_running  += diag["gc_loss"].item()
            cig_running += diag.get("cig_loss", torch.tensor(0.0)).item()

        return mse_running / n_batches, gc_running / n_batches, cig_running / n_batches

    for epoch in range(1, args.train_epochs + 1):
        tr_mse, tr_gc, tr_cig = run_epoch(train_loader, training=True)
        va_mse, va_gc, va_cig = run_epoch(vali_loader,  training=False)
        print(f"[epoch {epoch:03d}] "
              f"train MSE {tr_mse:.6f} | GC {tr_gc:.6f} | CIG {tr_cig:.6f}  ||  "
              f"val MSE {va_mse:.6f} | GC {va_gc:.6f} | CIG {va_cig:.6f}")

    # optional: evaluate on test set
    te_mse, te_gc, te_cig = run_epoch(test_loader, training=False)
    print(f"[test] MSE {te_mse:.6f} | GC {te_gc:.6f} | CIG {te_cig:.6f}")

    return model


# =========================
# ---- CLI wiring to your args
# =========================
def build_arg_parser():
    p = argparse.ArgumentParser()
    # ---- dataset args (must match your data_provider expectations)
    p.add_argument('--root_path', type=str, default='./data/ETT-small/')
    p.add_argument('--data_path', type=str, default='ETTh1.csv')
    p.add_argument('--data', type=str, default='ETTh1', choices=['ETTh1','ETTh2','ETTm1','ETTm2','custom'])
    p.add_argument('--features', type=str, default='M', choices=['M','S','MS'])
    p.add_argument('--target', type=str, default='OT')
    p.add_argument('--freq', type=str, default='h')
    p.add_argument('--embed', type=str, default='timeF')  # controls timeenc in data_provider
    p.add_argument('--seq_len', type=int, default=336)     # context length (C)
    p.add_argument('--label_len', type=int, default=48)   # decoder warmup, unused by MLP but part of loader
    p.add_argument('--pred_len', type=int, default=96)    # horizon (H)
    p.add_argument('--batch_size', type=int, default= 128)
    p.add_argument('--num_workers', type=int, default=0)
    # ---- training args
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--train_epochs', type=int, default=1)
    # ---- model/extractor/loss args
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--ridge_p', type=int, default=2)
    p.add_argument('--ridge_alpha', type=float, default=1e-2)
    p.add_argument('--gc_lambda', type=float, default=0.1)   # (informational; struct has own λ for CIG)
    p.add_argument('--gc_beta', type=float, default=4.0)
    p.add_argument('--gc_tau_q', type=float, default=0.3)
    p.add_argument('--use_cig', action='store_true', default=True)
    p.add_argument('--cig_lambda', type=float, default=0.05)
    p.add_argument('--cig_M', type=int, default=256)
    p.add_argument('--cig_eps_s', type=float, default=1e-8)
    p.add_argument('--cig_eps_inv', type=float, default=1e-5)
    p.add_argument('--device', type=str, default=None)
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    _ = train_with_factory(
        args=args,
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        ridge_p=args.ridge_p,
        ridge_alpha=args.ridge_alpha,
        gc_lambda=args.gc_lambda,
        gc_beta=args.gc_beta,
        gc_tau_q=args.gc_tau_q,
        use_cig=args.use_cig,
        cig_lambda=args.cig_lambda,
        cig_M=args.cig_M,
        cig_eps_s=args.cig_eps_s,
        cig_eps_inv=args.cig_eps_inv,
        device=args.device,
    )


if __name__ == "__main__":
    main()
