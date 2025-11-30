# -*- coding: utf-8 -*-
"""
ml_utils.py

1) issue_classifier.pkl 로부터 분류 모델을 불러와
   텍스트에 대한 issue / non_issue 예측을 수행.
2) process_lookup.pkl 에 저장된 교육용 예시들을 이용해
   "가장 비슷한 예시"의 검수-피드백 설명을 찾아,
   프로세스 설명으로 사용할 수 있도록 지원.
"""

import os
import joblib
from typing import Optional, Dict, Any

from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "issue_classifier.pkl")
LOOKUP_PATH = os.path.join(BASE_DIR, "process_lookup.pkl")

_model = None    # 분류 모델 캐시
_lookup = None   # lookup 데이터 캐시


def load_model():
    """issue_classifier.pkl 모델을 한 번만 로드."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def load_lookup() -> Optional[Dict[str, Any]]:
    """process_lookup.pkl 데이터를 한 번만 로드."""
    global _lookup
    if _lookup is None:
        if not os.path.exists(LOOKUP_PATH):
            # lookup 파일이 없으면 None 반환 (프로세스 매칭 기능 비활성)
            return None
        _lookup = joblib.load(LOOKUP_PATH)
    return _lookup


def predict_issue(text: str) -> dict:
    """
    하나의 텍스트에 대해 예측을 수행하고,
    최종 라벨과 issue/non_issue 확률을 반환합니다.
    """
    model = load_model()

    if not isinstance(text, str):
        text = str(text)

    proba = model.predict_proba([text])[0]
    classes = model.classes_
    prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}

    label = model.predict([text])[0]

    issue_prob = prob_dict.get("issue", 0.0)
    non_issue_prob = prob_dict.get("non_issue", 0.0)

    return {
        "label": label,                 # "issue" 또는 "non_issue"
        "prob_issue": issue_prob,       # issue일 확률
        "prob_non_issue": non_issue_prob,
        "proba_by_class": prob_dict,
    }


def suggest_process_description(game: str, text: str,
                                min_sim: float = 0.20) -> Optional[str]:
    """
    교육용 예시(엑셀)에 기반하여,
    - 같은 게임(FM / NK 등)에 속한 예시 중
    - TF-IDF 코사인 유사도가 가장 높은 행을 찾아
    - 그 행의 '검수 - 피드백' 내용을 프로세스 설명으로 반환.

    min_sim: 이 값보다 유사도가 낮으면 "적당한 예시 없음"으로 보고 None 반환.
    """
    lookup = load_lookup()
    if lookup is None:
        return None

    model = load_model()
    tfidf = model.named_steps["tfidf"]

    if not isinstance(text, str):
        text = str(text)

    # 쿼리 벡터
    q_vec = tfidf.transform([text])

    games = [str(g).strip().upper() for g in lookup["games"]]
    feedback_list = lookup["feedback"]
    X_all = lookup["X_tfidf"]

    # 같은 게임에 해당하는 인덱스만 사용
    target_game = str(game).strip().upper()
    idxs = [i for i, g in enumerate(games) if g == target_game]

    if not idxs:
        return None

    X_game = X_all[idxs]
    sims = cosine_similarity(q_vec, X_game)[0]

    best_pos = sims.argmax()
    best_sim = sims[best_pos]

    if best_sim < min_sim:
        # 너무 유사하지 않으면 사용하지 않음
        return None

    original_idx = idxs[best_pos]
    feedback = str(feedback_list[original_idx])

    return feedback
