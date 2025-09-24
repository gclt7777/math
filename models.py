"""模型构建与训练模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from .utils import CalibrationCfg, Config

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - 依赖缺失时提示
    raise ImportError("请先安装 xgboost 以运行第二问训练流程。") from exc

try:
    from catboost import CatBoostClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 catboost 以运行第二问训练流程。") from exc


@dataclass
class SearchResult:
    """保存网格搜索与最佳模型信息。"""

    best_name: str
    best_params: Dict[str, any]
    best_pipeline: BaseEstimator
    cv_results: pd.DataFrame


class XGBLabelWrapper(XGBClassifier, ClassifierMixin):
    """在 XGBoost 之上封装标签编码逻辑，支持字符串标签。"""

    def __init__(self, random_state: int, **kwargs):
        params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
            "use_label_encoder": False,
            "verbosity": 0,
        }
        params.update(kwargs)
        super().__init__(**params)
        self._label_encoder: LabelEncoder | None = None
        self._classes = None

    def fit(self, X, y, **kwargs):
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.set_params(num_class=len(self._label_encoder.classes_))
        super().fit(X, y_encoded, **kwargs)
        self._classes = self._label_encoder.classes_
        return self

    def predict(self, X):
        encoded = super().predict(X)
        if self._label_encoder is None:
            return encoded
        return self._label_encoder.inverse_transform(encoded.astype(int))

    def predict_proba(self, X):
        proba = super().predict_proba(X)
        return proba

    @property
    def classes_(self):  # type: ignore[override]
        if self._classes is not None:
            return self._classes
        return super().classes_

def _build_classifier(candidate_type: str, random_state: int, use_class_weight: bool) -> BaseEstimator:
    """根据候选类型实例化分类器。"""

    if candidate_type == "svm":
        clf = SVC(probability=True, random_state=random_state)
        if use_class_weight:
            clf.class_weight = "balanced"
        return clf
    if candidate_type == "rf":
        params = {"random_state": random_state, "n_jobs": -1}
        if use_class_weight:
            params["class_weight"] = "balanced"
        return RandomForestClassifier(**params)
    if candidate_type == "xgb":
        return XGBLabelWrapper(random_state=random_state)
    if candidate_type == "hgb":
        return HistGradientBoostingClassifier(
            random_state=random_state,
            early_stopping=False,
        )
    if candidate_type == "catboost":
        params = {
            "loss_function": "MultiClass",
            "random_seed": random_state,
            "verbose": 0,
        }
        if use_class_weight:
            params["auto_class_weights"] = "Balanced"
        return CatBoostClassifier(**params)
    if candidate_type == "mlp":
        return MLPClassifier(random_state=random_state)
    raise ValueError(f"未知模型类型: {candidate_type}")


def _pipeline_for_candidate(
    candidate_type: str,
    random_state: int,
    use_scaler: bool,
    use_class_weight: bool,
    sampler: Optional[BaseEstimator],
) -> Pipeline:
    """构造包含可选标准化步骤的管线。"""

    steps: List[tuple[str, BaseEstimator]] = []
    steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0.0)))
    if use_scaler and candidate_type in {"svm", "mlp"}:
        steps.append(("scaler", StandardScaler()))
    if sampler is not None:
        steps.append(("sampler", clone(sampler)))
    clf = _build_classifier(candidate_type, random_state, use_class_weight)
    steps.append(("clf", clf))
    pipeline_cls = ImbPipeline if sampler is not None else Pipeline
    return pipeline_cls(steps)


def _normalize_param_grid(candidate_type: str, params: Dict[str, List]) -> Dict[str, List]:
    """对参数网格进行类型归一化，避免 YAML 字符串导致报错。"""

    normalized: Dict[str, List] = {}
    for key, values in params.items():
        new_values = []
        for val in values:
            if candidate_type == "mlp" and key.lower() == "alpha" and isinstance(val, str):
                try:
                    new_values.append(float(val))
                    continue
                except ValueError:
                    pass
            new_values.append(val)
        normalized[key] = new_values
    return normalized


def _prepare_sampler(
    cfg: Config,
    y_train: pd.Series,
    logger,
    cv_folds: int,
) -> Optional[BaseEstimator]:
    """根据配置与训练集分布动态构建采样器。"""

    sampler_cfg = cfg.imbalance.sampler
    if not sampler_cfg:
        return None

    name = sampler_cfg.get("name")
    if not name:
        return None

    if not sampler_cfg.get("enabled", True):
        logger.info("采样器已在配置中禁用，跳过类不平衡重采样。")
        return None

    counts = y_train.value_counts().sort_index()
    max_count = counts.max()
    min_count = counts.min()
    ratio = max_count / min_count if min_count else float("inf")
    logger.info("训练集标签统计: %s | max/min=%.2f", counts.to_dict(), ratio)

    if min_count == 0:
        logger.warning("存在标签样本数为 0 的类别，无法执行重采样。")
        return None

    if cv_folds < 2:
        logger.warning("CV 折数 < 2，跳过重采样以避免过拟合。")
        return None

    if name.lower() == "smote":
        threshold = float(sampler_cfg.get("threshold", 3.0))
        if ratio <= threshold:
            logger.info(
                "类别不平衡比例 %.2f 未超过阈值 %.2f，保持原始训练集。",
                ratio,
                threshold,
            )
            return None

        min_samples_default = int(sampler_cfg.get("min_samples_default", 6))
        min_samples_hard = int(sampler_cfg.get("min_samples_hard", 2))
        desired_k = sampler_cfg.get("k_neighbors")
        max_k_neighbors = int(sampler_cfg.get("max_k_neighbors", 5))

        train_min = min_count - int(np.ceil(min_count / cv_folds))
        if train_min < 2:
            logger.warning(
                "训练集中最小类别样本数为 %d，经 %d 折划分后单折仅余 %d 个样本，SMOTE 将被跳过。",
                min_count,
                cv_folds,
                train_min,
            )
            return None

        if desired_k is None or str(desired_k).lower() in {"auto", "adaptive"}:
            if min_count < min_samples_hard:
                logger.warning(
                    "最小类别样本仅 %d (<%d)，无法安全应用 SMOTE，跳过重采样。",
                    min_count,
                    min_samples_hard,
                )
                return None
            if min_count >= min_samples_default:
                k_neighbors = min(max_k_neighbors, min_count - 1)
            else:
                k_neighbors = max(1, min_count - 1)
        else:
            k_neighbors = int(desired_k)
            if k_neighbors >= min_count:
                k_neighbors = max(1, min_count - 1)

        # 进一步受限于交叉验证训练折的可用样本数
        k_neighbors = min(k_neighbors, max(1, train_min - 1))

        strategy = sampler_cfg.get("strategy", "auto")
        logger.info(
            "启用 SMOTE: strategy=%s, k_neighbors=%d, ratio=%.2f",
            strategy,
            k_neighbors,
            ratio,
        )
        return SMOTE(
            sampling_strategy=strategy,
            k_neighbors=k_neighbors,
            random_state=cfg.data.random_seed,
        )

    logger.warning("未识别采样器 %s，跳过类不平衡重采样。", name)
    return None


def perform_grid_search(
    splits,
    cfg: Config,
    logger,
) -> SearchResult:
    """针对所有候选模型执行网格搜索并选出最佳组合。"""

    scoring = {name: get_scorer(name) for name in cfg.models.cv.scoring}

    min_class_count = splits.y_train.value_counts().min()
    folds = cfg.models.cv.folds
    if min_class_count < folds:
        adjusted = max(2, min_class_count)
        logger.warning(
            "训练集中最小类别样本数为 %d，小于配置的折数 %d，已自动将 CV 折数调整为 %d",
            min_class_count,
            folds,
            adjusted,
        )
        folds = adjusted

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.data.random_seed)

    all_results: List[pd.DataFrame] = []
    best_records: List[Tuple[str, float, float, float, BaseEstimator, Dict[str, any]]] = []

    sampler = _prepare_sampler(cfg, splits.y_train, logger, folds)
    use_class_weight = cfg.imbalance.use_class_weight and sampler is None
    if sampler is not None and cfg.imbalance.use_class_weight:
        logger.info("检测到重采样，将自动关闭 class_weight 以避免重复平衡。")

    for candidate in cfg.models.candidates:
        logger.info("开始搜索模型: %s", candidate.name)
        if not splits.feature_cols:
            raise ValueError("特征列表为空，无法训练模型。")
        pipeline = _pipeline_for_candidate(
            candidate.type,
            cfg.data.random_seed,
            splits.use_scaler,
            use_class_weight,
            sampler,
        )
        normalized_grid = _normalize_param_grid(candidate.type, candidate.params_grid)
        param_grid = {f"clf__{k}": v for k, v in normalized_grid.items()}
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit="f1_macro",
            cv=cv,
            n_jobs=cfg.models.cv.n_jobs,
            verbose=0,
            error_score="raise",
        )
        grid.fit(splits.X_train, splits.y_train)
        result_df = pd.DataFrame(grid.cv_results_)
        result_df.insert(0, "model", candidate.name)
        all_results.append(result_df)

        idx = grid.best_index_
        record = result_df.iloc[idx]
        best_records.append(
            (
                candidate.name,
                record["mean_test_f1_macro"],
                record.get("mean_test_balanced_accuracy", float("nan")),
                record.get("mean_test_accuracy", float("nan")),
                grid.best_estimator_,
                grid.best_params_,
            )
        )
        logger.info("模型 %s 最佳 f1_macro: %.4f", candidate.name, record["mean_test_f1_macro"])

    cv_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    if not best_records:
        raise RuntimeError("网格搜索未找到可用模型。")

    # 按照 f1_macro -> balanced_accuracy -> accuracy 进行排序
    best_records.sort(key=lambda x: (
        -x[1],
        -(x[2] if pd.notna(x[2]) else -1),
        -(x[3] if pd.notna(x[3]) else -1),
    ))
    top = best_records[0]
    logger.info("最终选择模型: %s | f1_macro=%.4f", top[0], top[1])

    return SearchResult(
        best_name=top[0],
        best_params=top[5],
        best_pipeline=top[4],
        cv_results=cv_results,
    )


def fit_final_model(
    search_result: SearchResult,
    splits,
    calibration_cfg: CalibrationCfg,
    cv_folds: int,
) -> BaseEstimator:
    """基于最佳配置重新训练并可选执行概率校准。"""

    X_trainval, y_trainval = splits.trainval()
    base_estimator = clone(search_result.best_pipeline)
    if calibration_cfg.enabled:
        from sklearn.calibration import CalibratedClassifierCV

        min_class = y_trainval.value_counts().min()
        if min_class < 2:
            warnings.warn(
                "训练集中存在类别样本数少于 2，跳过概率校准。",
                RuntimeWarning,
            )
            base_estimator.fit(X_trainval, y_trainval)
            return base_estimator

        folds = max(2, min(cv_folds, min_class))
        try:
            calibrator = CalibratedClassifierCV(
                estimator=base_estimator,
                method=calibration_cfg.method,
                cv=folds,
            )
        except TypeError:  # 兼容旧版 sklearn 接口
            calibrator = CalibratedClassifierCV(
                base_estimator=base_estimator,
                method=calibration_cfg.method,
                cv=folds,
            )
        calibrator.fit(X_trainval, y_trainval)
        return calibrator

    base_estimator.fit(X_trainval, y_trainval)
    return base_estimator
