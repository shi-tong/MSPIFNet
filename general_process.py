import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataPreprocess:
    def __init__(
        self,
        data_paths: Sequence[str],
        data_indexs:Sequence[str],
        labels: Optional[Sequence[int]] = None,
        output_root: str = "./",
        repeat_id:int = 0,
        window_size: int = 512,
        jump_step: int = 512,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.2,
        max_length: Optional[int] = None,
        scaler_type: str = "minmax", # "minmax", "standard", or "none"
        channel_index: Optional[int] = 0,
        small_sample_shots: Sequence[int] = (1, 2, 3, 4, 5),
        small_sample_levels: Optional[Sequence[str]] = None,
        snr_list: Sequence[float] = (-6,),
        noise_to: Sequence[str] = ("train", "valid", "test"),
        save_clean_full: bool = True,
        random_state: int = 42,
        pad_tail: bool = True,
        shuffle_train: bool = True,
    ):
        self.data_paths = list(data_paths)
        self.data_indexs = list(data_indexs)
        self.labels = list(labels) if labels is not None else list(range(len(data_paths)))
        self.output_root = Path(output_root).joinpath(f"repeat_{repeat_id:02d}")

        self.window_size = int(window_size)
        self.jump_step = int(jump_step)

        self.train_ratio = float(train_ratio)
        self.valid_ratio = float(valid_ratio)
        self.test_ratio = float(test_ratio)

        self.max_length = max_length
        self.scaler_type = scaler_type.lower()
        self.channel_index = channel_index

        self.small_sample_shots = list(small_sample_shots)
        self.small_sample_levels = (
            list(small_sample_levels)
            if small_sample_levels is not None
            else [f"L{i + 1}_shot{s}" for i, s in enumerate(self.small_sample_shots)]
        )

        self.snr_list = list(snr_list)
        self.noise_to = tuple(noise_to)

        self.save_clean_full = bool(save_clean_full)
        self.random_state = int(random_state)
        self.pad_tail = bool(pad_tail)
        self.shuffle_train = bool(shuffle_train)

    def _stable_seed(self, *items) -> int:
        key = "|".join([str(self.random_state)] + [str(item) for item in items])
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _rng_for(self, *items) -> np.random.Generator:
        return np.random.default_rng(self._stable_seed(*items))

    @staticmethod
    def _ensure_2d(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        elif data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        return data

    def _get_scaler(self):
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        if self.scaler_type == "standard":
            return StandardScaler()
        return None

    def _load_one_class(self, path: str, name_index) -> np.ndarray:
        data = scipy.io.loadmat(path)
        data = data[name_index]
        data = np.array(data, dtype=np.float32)

        if self.channel_index is not None:
            if data.ndim == 1:
                pass
            elif data.ndim >= 2:
                data = data[:, self.channel_index] # channel 0 for vibration
            else:
                raise ValueError(f"Don't support such data shape: {data.shape}")

        data = self._ensure_2d(data)

        if self.max_length is not None:
            data = data[:self.max_length]

        return data

    def add_gaussian_noise_to_raw(
        self,
        raw: np.ndarray,
        snr_db: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        raw = self._ensure_2d(raw).astype(np.float32, copy=False)
        rng = rng if rng is not None else self._rng_for("raw_noise", snr_db)

        signal_power = np.mean(np.square(raw), axis=0, keepdims=True)
        signal_power = np.maximum(signal_power, 1e-12)

        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=raw.shape)

        return (raw + noise).astype(np.float32)

    def add_gaussian_noise(self, X: np.ndarray, snr_db: float) -> np.ndarray:
        X = X.astype(np.float32, copy=False)

        if X.ndim == 1:
            axes = None
        else:
            axes = tuple(range(1, X.ndim))

        signal_power = np.mean(np.square(X), axis=axes, keepdims=True)
        signal_power = np.maximum(signal_power, 1e-12)

        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        rng = self._rng_for("window_noise", snr_db)
        noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=X.shape)

        return (X + noise).astype(np.float32)

    def _split_raw_signal(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        T = data.shape[0]

        n_train = int(T * self.train_ratio)
        n_valid = int(T * self.valid_ratio)
        n_test = T - n_train - n_valid

        if min(n_train, n_valid, n_test) <= 0:
            raise ValueError(
                f"The data length {T} is too short to be proportionally divided into train/valid/test."
            )

        train_data = data[:n_train]
        valid_data = data[n_train:n_train + n_valid]
        test_data = data[n_train + n_valid:]

        return train_data, valid_data, test_data

    def _fit_transform_split(
        self,
        train_data: np.ndarray,
        valid_data: np.ndarray,
        test_data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scaler = self._get_scaler()

        if scaler is None:
            return train_data, valid_data, test_data

        scaler.fit(train_data)

        train_data = scaler.transform(train_data)
        valid_data = scaler.transform(valid_data)
        test_data = scaler.transform(test_data)

        return train_data, valid_data, test_data

    def slide_window_sample(self, data: np.ndarray, label: int) -> Tuple[np.ndarray, np.ndarray]:
        data = self._ensure_2d(data)
        T, C = data.shape

        if T < self.window_size:
            if not self.pad_tail:
                return (
                    np.empty((0, self.window_size, C), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                )

            pad_len = self.window_size - T
            data = np.vstack([data, np.zeros((pad_len, C), dtype=data.dtype)])
            T = self.window_size

        starts = list(range(0, T - self.window_size + 1, self.jump_step))

        if len(starts) == 0:
            starts = [0]

        # Choose whether to retain the tail window of the last incomplete step.
        if self.pad_tail and starts[-1] + self.window_size < T:
            starts.append(starts[-1] + self.jump_step)

        windows = []

        for start in starts:
            end = start + self.window_size

            if end <= T:
                win = data[start:end]
            else:
                pad_len = end - T
                win = np.vstack([
                    data[start:T],
                    np.zeros((pad_len, C), dtype=data.dtype),
                ])

            windows.append(win.astype(np.float32))

        X = np.stack(windows, axis=0)
        y = np.full((X.shape[0],), label, dtype=np.int64)

        return X, y

    def build_full_dataset(
        self,
        snr_db: Optional[float] = None,
        shuffle_train: bool = False,
        train_perm: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        train_x_list, train_y_list = [], []
        valid_x_list, valid_y_list = [], []
        test_x_list, test_y_list = [], []

        for class_idx, (path, label, index) in enumerate(zip(self.data_paths, self.labels, self.data_indexs)):
            raw = self._load_one_class(path, index)

            if snr_db is not None:
                rng = self._rng_for("raw_noise", "class", class_idx, "label", label, "snr", snr_db)
                raw = self.add_gaussian_noise_to_raw(raw, snr_db, rng=rng)

            train_raw, valid_raw, test_raw = self._split_raw_signal(raw)

            train_raw, valid_raw, test_raw = self._fit_transform_split(
                train_raw, valid_raw, test_raw
            )

            train_x, train_y = self.slide_window_sample(train_raw, label)
            valid_x, valid_y = self.slide_window_sample(valid_raw, label)
            test_x, test_y = self.slide_window_sample(test_raw, label)

            train_x_list.append(train_x)
            train_y_list.append(train_y)
            valid_x_list.append(valid_x)
            valid_y_list.append(valid_y)
            test_x_list.append(test_x)
            test_y_list.append(test_y)

        dataset = {
            "train": np.concatenate(train_x_list, axis=0),
            "train_label": np.concatenate(train_y_list, axis=0),
            "valid": np.concatenate(valid_x_list, axis=0),
            "valid_label": np.concatenate(valid_y_list, axis=0),
            "test": np.concatenate(test_x_list, axis=0),
            "test_label": np.concatenate(test_y_list, axis=0),
        }

        if train_perm is not None:
            dataset = self.apply_train_permutation(dataset, train_perm)
        elif shuffle_train:
            rng = self._rng_for("train_shuffle", "snr", snr_db)
            train_perm = rng.permutation(len(dataset["train_label"]))
            dataset = self.apply_train_permutation(dataset, train_perm)

        return dataset

    def build_clean_full_dataset(self) -> Dict[str, np.ndarray]:
        return self.build_full_dataset(snr_db=None, shuffle_train=self.shuffle_train)

    @staticmethod
    def apply_train_permutation(
        dataset: Dict[str, np.ndarray],
        train_perm: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        dataset = {k: v.copy() for k, v in dataset.items()}
        dataset["train"] = dataset["train"][train_perm]
        dataset["train_label"] = dataset["train_label"][train_perm]

        return dataset

    def apply_noise_to_dataset_after_split(
        self,
        dataset: Dict[str, np.ndarray],
        snr_db: float,
    ) -> Dict[str, np.ndarray]:
        noisy = {k: v.copy() for k, v in dataset.items()}

        for split in self.noise_to:
            noisy[split] = self.add_gaussian_noise(noisy[split], snr_db)

        return noisy

    def select_n_shot_indices(self, y: np.ndarray, shot: int, level: str) -> np.ndarray:
        rng = self._rng_for("n_shot", level, shot)
        selected = []
        for label in self.labels:
            label_idx = np.where(y == label)[0]
            selected.append(rng.choice(label_idx, size=shot, replace=False))
        selected = np.concatenate(selected, axis=0)
        selected = rng.permutation(selected)
        return selected

    @staticmethod
    def subset_train(
        dataset: Dict[str, np.ndarray],
        train_indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        subset = {k: v.copy() for k, v in dataset.items()}
        subset["train"] = subset["train"][train_indices]
        subset["train_label"] = subset["train_label"][train_indices]
        return subset

    def save_dataset(
        self,
        dataset: Dict[str, np.ndarray],
        save_dir: Path,
        meta: Optional[dict] = None,
    ) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name in ["train", "train_label", "valid", "valid_label", "test", "test_label"]:
            np.save(save_dir / f"{name}.npy", dataset[name])
        if meta is not None:
            with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        self.print_summary(dataset, title=str(save_dir))

    @staticmethod
    def print_summary(dataset: Dict[str, np.ndarray], title: str = "") -> None:
        if title:
            print("\n" + "=" * 80)
            print(title)

        print("train:", dataset["train"].shape, "train_label:", dataset["train_label"].shape)
        print("valid:", dataset["valid"].shape, "valid_label:", dataset["valid_label"].shape)
        print("test :", dataset["test"].shape, "test_label :", dataset["test_label"].shape)

        for split in ["train", "valid", "test"]:
            y = dataset[f"{split}_label"]
            labels, counts = np.unique(y, return_counts=True)
            print(f"{split} label distribution:", dict(zip(labels.tolist(), counts.tolist())))

    def _common_meta(self) -> dict:
        return {
            "window_size": self.window_size,
            "jump_step": self.jump_step,
            "train_ratio": self.train_ratio,
            "valid_ratio": self.valid_ratio,
            "test_ratio": self.test_ratio,
            "max_length": self.max_length,
            "scaler_type": self.scaler_type,
            "channel_index": self.channel_index,
            "labels": self.labels,
            "random_state": self.random_state,
            "pad_tail": self.pad_tail,
            "shuffle_train": self.shuffle_train,
            "noise_position": "raw_signal_before_split",
        }

    def generate_all_conditions(self) -> Dict[str, Dict[str, np.ndarray]]:
        all_sets = {}
        base = self.build_full_dataset(snr_db=None, shuffle_train=False)
        train_perm = None
        if self.shuffle_train:
            rng = self._rng_for("common_train_shuffle")
            train_perm = rng.permutation(len(base["train_label"]))
            base = self.apply_train_permutation(base, train_perm)
        common_meta = self._common_meta()
        if self.save_clean_full:
            save_dir = self.output_root / "clean_full"
            self.save_dataset(
                base,
                save_dir,
                meta={
                    **common_meta,
                    "condition": "clean_full",
                    "snr_db": None,
                },
            )
            all_sets["clean_full"] = base
        small_indices = {
            (level, shot): self.select_n_shot_indices(base["train_label"], shot, level)
            for level, shot in zip(self.small_sample_levels, self.small_sample_shots)
        }
        for level, shot in zip(self.small_sample_levels, self.small_sample_shots):
            idx = small_indices[(level, shot)]
            small_set = self.subset_train(base, idx)

            save_dir = self.output_root / "small_sample" / level
            self.save_dataset(
                small_set,
                save_dir,
                meta={
                    **common_meta,
                    "condition": "small_sample",
                    "snr_db": None,
                    "shot_per_class": shot,
                },
            )
            all_sets[f"small_sample/{level}"] = small_set
        for snr in self.snr_list:
            snr_name = f"snr_{str(snr).replace('-', 'm').replace('.', 'p')}"
            noisy_full = self.build_full_dataset(
                snr_db=snr,
                shuffle_train=False,
                train_perm=train_perm,
            )
            save_dir = self.output_root / "noise" / snr_name
            self.save_dataset(
                noisy_full,
                save_dir,
                meta={
                    **common_meta,
                    "condition": "noise",
                    "snr_db": snr,
                },
            )
            all_sets[f"noise/{snr_name}"] = noisy_full

            for level, shot in zip(self.small_sample_levels, self.small_sample_shots):
                idx = small_indices[(level, shot)]
                small_noisy = self.subset_train(noisy_full, idx)
                save_dir = self.output_root / "small_sample_noise" / snr_name / level
                self.save_dataset(
                    small_noisy,
                    save_dir,
                    meta={
                        **common_meta,
                        "condition": "small_sample_noise",
                        "snr_db": snr,
                        "shot_per_class": shot,
                    },
                )
                all_sets[f"small_sample_noise/{snr_name}/{level}"] = small_noisy

        return all_sets

if __name__ == "__main__":
    data_paths, data_indexs = None, None
    n_repeats = 10 # Number of experimental repetitions
    for repeat_id in range(n_repeats):
        pre = DataPreprocess(
            data_paths=data_paths,
            data_indexs=data_indexs,
            labels=list(range(9)), # 9 for Ottawa, 10 for CWRU
            repeat_id=repeat_id,
            output_root=".../",
            window_size=512,
            jump_step=512,
            train_ratio=0.7,
            valid_ratio=0.1,
            test_ratio=0.2,
            max_length=None, # None for all
            scaler_type="minmax",
            channel_index=0,
            small_sample_shots=(1, 2, 3, 4),
            small_sample_levels=("L1_shot1", "L2_shot2", "L3_shot3", "L4_shot4"),
            snr_list=(-6, -4, -2, 0),
            noise_to=("train", "valid", "test"),
            save_clean_full=True,
            random_state=42,
            pad_tail=True,
            shuffle_train=True,
        )
        pre.generate_all_conditions()
