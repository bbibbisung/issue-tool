# -*- coding: utf-8 -*-
"""
ml_utils.py

1) issue_classifier.pkl 로부터 분류 모델을 불러와
   텍스트에 대한 issue / non_issue 예측을 수행.

이번 패치에서 추가된 점:
- 문맥 기반 non_issue 보정 로직 유지/보강
  (예: 이벤트/패스 기간 종료 + 이용자 개인 사유로 인한 미참여 등)
- '단순 문의/안내'에 가까운 문구를 simple_question_non_issue 플래그로 감지
  → 강한 부정어/이슈 키워드가 없는 경우에 한해 non_issue 쪽으로 보정
- 프로세스 설명(suggest_process_description) 관련 기능 및 lookup 파일 사용 제거
  → X/Y 차원 불일치(ValueError) 발생 원인 자체를 제거
"""

import os
import re
import joblib
from typing import Optional, Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "issue_classifier.pkl")

_model = None    # 분류 모델 캐시


# ----------------------------------------------------------------------
# 0) 로컬 유틸: 텍스트 정규화 & 문맥 패턴 검사
# ----------------------------------------------------------------------

def normalize_text_local(text: str) -> str:
    """
    공백/일부 특수문자를 제거한 버전.
    rules.py 의 normalize_text 와 충돌하지 않도록
    이 파일 안에서만 쓰는 경량 버전입니다.
    """
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[\s\.\,\!\?\~\-\_\+\=\(\)\[\]\{\}\\\/\|\"\'\:;]+", "", text)


def find_hits_local(keywords: List[str], text: str, text_compact: str) -> List[str]:
    """
    간단 키워드 매칭:
    - 원문에 그대로 포함되거나
    - 공백 제거 버전에 포함되면 히트로 판단
    """
    hits: List[str] = []
    for kw in keywords:
        if not kw:
            continue
        kw_comp = re.sub(r"\s+", "", kw)
        if (kw in text) or (kw_comp and kw_comp in text_compact):
            hits.append(kw)
    return hits


# ----------------------------------------------------------------------
# 0-1) non_issue 문맥 플래그들
# ----------------------------------------------------------------------

def detect_time_expired_non_issue(text: str) -> bool:
    """
    '이벤트/패스가 기간 종료라서 더 이상 적용이 안 되는 상황'을
    non_issue(정상 정책)으로 인식하기 위한 휴리스틱.

    대략적인 조건:
    1) 기간 종료/마감/이미 끝남 관련 단어가 있고
    2) 이용자 측 사유(늦게 봄, 병원, 잠듦, 외출 등)가 함께 언급되며
    3) 운영/버그 잘못을 직접적으로 지적하기보다는,
       '아쉽다/되냐/가능하냐' 류의 뉘앙스에 가까운 경우
    """
    if not text:
        return False

    t = text
    t_comp = normalize_text_local(text)

    # 1) 기간 종료/마감 계열 키워드
    expired_keywords = [
        "기간 종료", "기간이 종료", "기간이 끝", "기간 끝",
        "마감", "마감이", "이미 끝난", "이미 끝났", "끝났네요",
        "이벤트 끝났", "패스 끝났", "지난 이벤트", "12시까지", "자정까지",
        "어제까지만", "어제까진", "어제까지였", "어제까지였던",
        "기한이 지났", "기한이 지나",
    ]
    has_expired = bool(find_hits_local(expired_keywords, t, t_comp))
    if not has_expired:
        return False

    # 2) 이용자 개인 사유(병원, 늦게 봄, 잠듦 등)
    user_side_reason_keywords = [
        "늦게 봐서", "이제 봐서", "오늘 알았", "오늘 처음 봤",
        "병원 다녀", "병원에 다녀", "병원에 있어서", "응급실",
        "일하느라", "야근하느라", "출장 다녀와서",
        "잠들어", "잠들었", "자느라", "잤더니",
        "외출했다가", "집에 없어서", "접속을 못했",
        "못 들어왔", "못 들어가서", "못 눌렀", "못 했습니다",
    ]
    has_user_reason = bool(find_hits_local(user_side_reason_keywords, t, t_comp))

    # 3) 아주 강한 '운영 잘못' 지적이 있으면 제외
    strong_blame_keywords = [
        # 운영/정책 비난
        "운영 진짜", "운영 개판", "운영 잘못", "운영 탓", "운영 때문",
        # 버그/오류 의심
        "버그 아니냐", "버그냐", "버그인 거", "버그인것", "버그 맞냐",
        "오류 아니냐", "오류 맞냐",
        # 보상/환불 강한 요구
        "보상해라", "보상 해라", "보상해 줘라", "보상해줘라",
        "보상해 주세요", "보상 해 주세요", "보상 좀 해", "보상 좀 해라",
        "환불하라", "환불 해라", "환불해주세요", "환불 해주세요",
        "환불 요청", "환불 요구", "환불 좀",
        # 감정적인 강한 비난/사기 의심
        "이상하다", "왜 이렇게",
        "쓰레기 이벤트", "이벤트 쓰레기",
        "쓰레기 게임", "최악", "개판", "X같", "개같",
        "사기 아님", "사기 아니냐", "장난하냐", "뭐 하는 거냐", "진짜 뭐냐",
    ]
    has_strong_blame = bool(find_hits_local(strong_blame_keywords, t, t_comp))

    # 기본 휴리스틱:
    # - 기간 종료 키워드가 있고
    # - 이용자 사유 키워드가 하나 이상 있으며
    # - 운영/버그 탓을 강하게 하는 표현이 없으면
    #   → '기간 종료로 인한 정상 상황'으로 본다.
    return has_expired and has_user_reason and not has_strong_blame


def detect_simple_question_non_issue(text: str) -> bool:
    """
    전반적인 '단순 문의/안내 요청'에 해당하는지 판별하는 휴리스틱.

    조건(보수적으로 설정):
    1) '문의드립니다/알고 싶습니다/알려주실 수 있을까요' 등 정중한 질문 표현
    2) 아래의 강한 이슈 키워드(버그/오류/안되/보상 안/환불/튕김/접속 안됨 등)는 포함되지 않을 것
    """
    if not text:
        return False

    t = text
    t_comp = normalize_text_local(text)

    # 1) 정중한 질문/문의 표현
    question_keywords = [
        "문의드립니다", "문의 드립니다", "문의드려요", "문의 드려요",
        "질문드립니다", "질문 드립니다",
        "궁금해서", "궁금한 점", "궁금합니다",
        "알고 싶습니다", "알고싶습니다",
        "알려주실 수 있을까요", "알려 주실 수 있을까요",
        "알려주실 수 있나요", "알려 주실 수 있나요",
        "알려주세요", "알려 주세요", "알려주시면 감사",
        "혹시", "문의 남깁니다", "문의 남겨요",
    ]

    if not find_hits_local(question_keywords, t, t_comp):
        return False

    # 2) 강한 이슈/장애/보상 단어가 있으면 '단순 문의'로 보지 않음
    strong_issue_keywords = [
        "버그", "오류", "에러", "튕김", "튕겨", "렉", "멈춤", "프리징",
        "접속 안되", "접속 안 되", "접속이 안되", "접속이 안 되",
        "실행 안되", "실행 안 되", "실행이 안되", "실행이 안 되",
        "서버 에러", "서버 오류", "서버 터짐",
        "보상 안", "보상이 안", "보상도 안", "보상 왜",
        "환불", "환불해", "환불 요청",
        "지급 안됐", "지급 안 됐", "안 들어왔", "안들어왔",
        "로그인이 안되", "로그인이 안 되",
    ]

    if find_hits_local(strong_issue_keywords, t, t_comp):
        return False

    return True


def detect_non_issue_context_flags(text: str) -> Dict[str, bool]:
    """
    non_issue 로 보정할 수 있는 문맥 플래그들을 한 곳에서 계산하는 함수.

    - time_expired_non_issue : 기간 종료 + 개인 사유
    - simple_question_non_issue : 정중한 단순 문의/안내 요청
    """
    flags = {
        "time_expired_non_issue": detect_time_expired_non_issue(text),
        "simple_question_non_issue": detect_simple_question_non_issue(text),
    }
    return flags


# ----------------------------------------------------------------------
# 1) 모델 로딩
# ----------------------------------------------------------------------

def load_model():
    """issue_classifier.pkl 모델을 한 번만 로드."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


# ----------------------------------------------------------------------
# 2) issue / non_issue 예측 + 문맥 보정
# ----------------------------------------------------------------------

def predict_issue(text: str) -> dict:
    """
    하나의 텍스트에 대해 예측을 수행하고,
    최종 라벨과 issue/non_issue 확률을 반환합니다.

    - 기본적으로는 기존 ML 결과를 그대로 사용하되,
    - 문맥 플래그(예: time_expired_non_issue, simple_question_non_issue)가 켜져 있으면
      non_issue 쪽으로 보정합니다.
    """
    model = load_model()

    if not isinstance(text, str):
        text = str(text)

    # 1) 기본 ML 예측
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}

    label = model.predict([text])[0]

    issue_prob = prob_dict.get("issue", 0.0)
    non_issue_prob = prob_dict.get("non_issue", 0.0)

    # 2) 문맥 플래그 계산
    context_flags = detect_non_issue_context_flags(text)

    # 3) 기간 종료 정상 상황 → non_issue 보정
    if context_flags.get("time_expired_non_issue", False):
        if label == "non_issue":
            non_issue_prob = max(non_issue_prob, 0.85)
            issue_prob = min(issue_prob, 0.15)
        else:
            label = "non_issue"
            non_issue_prob = max(non_issue_prob, 0.80)
            issue_prob = min(issue_prob, 0.20)

        prob_dict["issue"] = issue_prob
        prob_dict["non_issue"] = non_issue_prob

    # 4) 정중한 단순 문의/안내 → 약하게 non_issue 쪽으로 보정
    #    (버그/오류/접속불가/보상문제 등 강한 패턴이 없는 경우만 해당)
    if context_flags.get("simple_question_non_issue", False):
        if label == "non_issue":
            # 이미 non_issue면 약하게만 강화
            non_issue_prob = max(non_issue_prob, 0.7)
            issue_prob = min(issue_prob, 0.3)
        else:
            # ML이 issue로 보더라도 확률이 애매할 때만 뒤집음
            if issue_prob < 0.6:
                label = "non_issue"
                non_issue_prob = max(non_issue_prob, 0.7)
                issue_prob = min(issue_prob, 0.3)

        prob_dict["issue"] = issue_prob
        prob_dict["non_issue"] = non_issue_prob

    return {
        "label": label,                 # "issue" 또는 "non_issue"
        "prob_issue": issue_prob,       # issue일 확률
        "prob_non_issue": non_issue_prob,
        "proba_by_class": prob_dict,
        "context_flags": context_flags,  # (추가 정보) 어떤 문맥 플래그가 켜졌는지
    }
