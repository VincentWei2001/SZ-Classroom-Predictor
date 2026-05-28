"""让旧版 scikit-learn 保存的 joblib 模型在新版 sklearn 下能 unpickle。"""
from __future__ import annotations

import sys


def register() -> None:
    """在首次 joblib.load 含 RandomForest / DecisionTree 的模型之前调用。"""
    if getattr(register, "_done", False):
        return

    # sklearn >= 0.22 将实现挪到私有模块，旧 pickle 仍引用旧公开路径
    if "sklearn.ensemble.forest" not in sys.modules:
        try:
            from sklearn.ensemble import _forest

            sys.modules["sklearn.ensemble.forest"] = _forest
        except Exception:
            pass

    if "sklearn.tree.tree" not in sys.modules:
        try:
            from sklearn.tree import _classes

            sys.modules["sklearn.tree.tree"] = _classes
        except Exception:
            pass

    if "sklearn.ensemble.base" not in sys.modules:
        try:
            from sklearn.ensemble import _base

            sys.modules["sklearn.ensemble.base"] = _base
        except Exception:
            pass

    setattr(register, "_done", True)
