# -*- coding: utf-8 -*-
"""
app.py

Flask 기반 웹 서버.
- "/" 페이지에서 게임 / 텍스트 입력
- RULES + 머신러닝 결과 동시에 표시
- 교육용 요약만 제공
- 로그 파일로 저장
- (5순위 패치) 나딘 피드백 기록 & 다음번 자동 반영

※ Training Mode, 프로세스 설명 기능은 모두 제거한 버전입니다.
"""

import os
import csv
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any

from flask import Flask, render_template, request

from ml_utils import predict_issue
from rules import (
    rules_classify,
    normalize_text,
    find_keyword_hits,
    detect_character_game_mismatch,   # (4순위 패치) 게임-캐릭터 뒤틀림 감지용
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "prediction_log.csv")
FEEDBACK_LOG_PATH = os.path.join(LOG_DIR, "feedback_log.csv")

os.makedirs(LOG_DIR, exist_ok=True)

app = Flask(__name__)


# ---------------------------------------------------
# 1) 욕설/감정 표현 간단 검출
# ---------------------------------------------------

ABUSE_KEYWORDS = [
    "욕", "욕설", "비속어", "씨발", "ㅅㅂ", "개새", "개같", "개같네", "개같은",
    "패드립", "비하", "모욕", "인신공격", "막말", "쓰레기", "X같", "병신", "ㅄ",
]


def detect_abuse_hits(text: str):
    """
    요약 문구 생성을 위해 텍스트 내 욕설/비속어 여부만 간단히 검출.
    """
    text_compact = normalize_text(text)
    return find_keyword_hits(ABUSE_KEYWORDS, text, text_compact)


# ---------------------------------------------------
# 2) FEEDBACK 관련 유틸 (5순위 패치)
# ---------------------------------------------------

def get_text_hash(text: str) -> str:
    """
    텍스트(정규화 버전)에 대한 해시 값 생성.
    - 같은 텍스트는 항상 동일한 text_hash를 사용하게 하기 위함.
    """
    normalized = normalize_text(text or "")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_feedback_for_hash(text_hash: str) -> Optional[Dict[str, str]]:
    """
    feedback_log.csv에서 같은 text_hash를 가진 마지막 피드백을 찾아 반환.
    - 동일 텍스트에 대해 여러 번 피드백이 쌓일 수 있으므로, 가장 마지막 것을 사용.
    반환 예:
    {
        "correct_label": "issue" / "non_issue" / "",
        "correct_category": "...",
        "comment": "..."
    }
    """
    if not text_hash:
        return None

    if not os.path.exists(FEEDBACK_LOG_PATH):
        return None

    last_row: Optional[Dict[str, str]] = None

    with open(FEEDBACK_LOG_PATH, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("text_hash") == text_hash:
                last_row = row

    if not last_row:
        return None

    return {
        "correct_label": (last_row.get("correct_label") or "").strip(),
        "correct_category": (last_row.get("correct_category") or "").strip(),
        "comment": (last_row.get("comment") or "").strip(),
    }


def save_feedback(
    game: str,
    text: str,
    text_hash: str,
    correct_label: str,
    correct_category: str,
    comment: str,
) -> None:
    """
    feedback_log.csv에 피드백 한 줄 추가.
    컬럼:
    - timestamp, game, text_hash, text, correct_label, correct_category, comment
    """
    is_new_file = not os.path.exists(FEEDBACK_LOG_PATH)

    with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(
                [
                    "timestamp",
                    "game",
                    "text_hash",
                    "text",
                    "correct_label",
                    "correct_category",
                    "comment",
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                game,
                text.replace("\n", " ").strip(),
                correct_label,
                correct_category,
                comment,
            ]
        )


# ---------------------------------------------------
# 3) RULES + ML 종합 라벨 결정 (맥락 플래그 반영)
#    + 진단 정보 계산용 보조 함수
# ---------------------------------------------------

def decide_final_label(rule_result: Dict, ml_result: Optional[Dict]) -> str:
    """
    RULES 결과와 ML 결과를 종합해서 최종 라벨을 결정.

    정책 우선순위(보수적으로 issue 쪽을 우선하는 방향):
    1) ML 문맥 플래그가 '기간 종료 정상 상황'이면 → 무조건 non_issue
    2) RULES가 non_issue + 카테고리 '기타' 이면 기본적으로 non_issue
    3) RULES와 ML 라벨이 같으면 그대로
    4) RULES가 issue이면 RULES를 최우선으로 신뢰
    5) RULES가 non_issue, ML이 issue인 경우
       - ML issue 확률이 0.8 이상일 때만 예외적으로 issue 인정
         (기존 0.9 → 0.8로 완화하여 '이슈인데 non_issue로 빠지는' 케이스 줄임)
    6) 나머지는 non_issue
    """
    rule_label = rule_result.get("label", "non_issue")
    rule_category = rule_result.get("category", "")
    ml_label = (ml_result or {}).get("label", "non_issue")
    issue_prob = (ml_result or {}).get("prob_issue", 0.0)

    context_flags = (ml_result or {}).get("context_flags", {})

    # 0) ML 문맥 플래그에서 '기간 종료 정상 상황'을 강하게 non_issue 로 처리
    if context_flags.get("time_expired_non_issue", False):
        return "non_issue"

    # 1) RULES가 non_issue + 기타면 기본적으로 non_issue
    if rule_label == "non_issue" and rule_category == "기타":
        # 다만, ML이 issue 쪽으로 매우 강하게 치우친 경우는 아래 로직에서 다시 한 번 검토
        pass
    else:
        # 2) RULES와 ML이 같으면 그대로
        if rule_label == ml_label:
            return rule_label

        # 3) RULES가 issue면 RULES 우선
        if rule_label == "issue":
            return "issue"

    # 4) RULES는 non_issue, ML은 issue인데 확률이 꽤 높은 경우만 issue로 인정
    if rule_label == "non_issue" and ml_label == "issue" and issue_prob >= 0.8:
        return "issue"

    # 5) 나머지는 non_issue
    return "non_issue"


def build_diagnostics(
    rule_result: Optional[Dict],
    ml_result: Optional[Dict],
    final_label: Optional[str],
) -> Optional[Dict]:
    """
    교육생이 '어디를 의심해야 하는지'를 한 번에 볼 수 있도록
    RULES / ML / 최종 라벨 간의 관계를 요약한 진단 정보 생성.

    주요 플래그:
    - label_conflict: RULES vs ML 라벨 충돌
    - low_confidence: ML이 issue / non_issue를 애매하게 보고 있는 케이스
      (issue 확률이 40~60% 사이일 때)
    - category_other: RULES 카테고리가 '기타'인 케이스
    """
    if not rule_result or not ml_result or not final_label:
        return None

    rule_label = rule_result.get("label", "non_issue")
    rule_category = rule_result.get("category", "")
    ml_label = ml_result.get("label", "non_issue")
    prob_issue = float(ml_result.get("prob_issue", 0.0))
    prob_non = float(ml_result.get("prob_non_issue", 0.0))

    messages = []

    # 1) RULES vs ML 라벨 충돌
    label_conflict = rule_label != ml_label
    if label_conflict:
        messages.append(
            f"RULES와 ML 라벨이 서로 다릅니다. (RULES: {rule_label}, ML: {ml_label})"
        )

    # 2) ML 신뢰도 낮음 (issue 확률이 40~60% 구간이면 애매한 케이스로 간주)
    low_confidence = 0.4 <= prob_issue <= 0.6
    if low_confidence:
        messages.append(
            "ML이 issue / non_issue 확률을 비슷하게 보고 있어 신뢰도가 낮은 케이스입니다."
            f" (issue {prob_issue * 100:.1f}%, non_issue {prob_non * 100:.1f}%)"
        )

    # 3) RULES 카테고리가 '기타'인 경우
    category_other = "기타" == rule_category or rule_category.startswith("기타 ")
    if category_other:
        messages.append(
            "RULES 카테고리가 '기타'로 분류되었습니다. "
            "체크리스트를 기준으로 직접 카테고리를 다시 한 번 확인해 주세요."
        )

    needs_attention = label_conflict or low_confidence or category_other

    if not messages:
        messages.append(
            "현재 기준으로는 RULES와 ML이 큰 충돌 없이 일치하는 케이스입니다. "
            "그래도 체크리스트와 함께 교차 확인해 주세요."
        )

    return {
        "rule_label": rule_label,
        "rule_category": rule_category,
        "ml_label": ml_label,
        "prob_issue": prob_issue,
        "prob_non_issue": prob_non,
        "label_conflict": label_conflict,
        "low_confidence": low_confidence,
        "category_other": category_other,
        "needs_attention": needs_attention,
        "messages": messages,
    }


# ---------------------------------------------------
# 4) 교육자용 요약 문구 생성
# ---------------------------------------------------

def build_summary_text(
    game: str,
    text: str,
    rule_result: Dict,
    ml_result: Optional[Dict],
) -> Optional[str]:
    """
    교육자가 바로 피드백을 줄 수 있도록,
    - 핵심 이슈가 무엇인지
    - 욕설/감정 표현은 부가 요소인지
    를 간단히 정리한 요약 문구를 생성.
    """
    if not rule_result:
        return None

    label = rule_result.get("label", "non_issue")
    category = rule_result.get("category", "")

    parts = []

    # 1) 핵심 이슈 문장
    if label == "issue":
        main_head = "핵심 이슈"
    else:
        main_head = "핵심 판단"

    main_type: Optional[str] = None
    if "접속/로그인 불가" in category:
        main_type = "접속/로그인 불가(게임 접속 불가 / 서버 연결 실패)"
    elif "버그/오류 제보" in category:
        main_type = "버그/오류 제보"
    elif (
        "보상/재화" in category
        or "재화/보상" in category
        or "보상/이벤트" in category
        or "결제/재화/보상" in category
    ):
        main_type = "결제/보상/재화 관련 이슈"
    elif "커뮤니티(" in category:
        main_type = "커뮤니티 게시글/댓글 관련 이슈"
    elif "기간 종료" in category:
        main_type = "이벤트/패스 기간 종료로 인한 정상 동작"

    if main_type:
        parts.append(f"{main_head}: {main_type} (카테고리: {category})")
    else:
        parts.append(f"{main_head}: {category}")

    # 2) 욕설/감정 표현 여부
    abuse_hits = detect_abuse_hits(text)
    if abuse_hits:
        if (
            "접속/로그인 불가" in category
            or "버그/오류 제보" in category
            or "보상" in category
        ):
            parts.append(
                "부가 요소: 욕설/감정 표현이 포함되어 있으나, "
                "우선 처리해야 할 대상은 위의 핵심 이슈입니다."
            )
        elif "커뮤니티(" in category:
            parts.append(
                "부가 요소: 욕설/비매너 표현이 포함되어 있으며, "
                "커뮤니티 운영 관점에서 모니터링이 필요한 게시글입니다."
            )
        else:
            parts.append("부가 요소: 욕설/감정 표현이 포함되어 있습니다.")

    # 3) ML 결과는 참고용으로만 간단히
    if ml_result:
        prob_issue = float(ml_result.get("prob_issue", 0.0)) * 100.0
        prob_non = float(ml_result.get("prob_non_issue", 0.0)) * 100.0
        parts.append(
            f"참고용 ML 판단: issue {prob_issue:.1f}%, "
            f"non_issue {prob_non:.1f}% (교육용 참고값)"
        )

    return "\n".join(parts)


# ---------------------------------------------------
# 5) 공유 기준 안내 문구 생성 (최종 판단 상단 박스용)
# ---------------------------------------------------

def build_share_guideline(rule_result: Optional[Dict], final_label: Optional[str]) -> str:
    """
    최종 판단 박스 안에 들어가는 '현재로서는 즉시 공유가 필요한 이슈...' 같은
    안내 문구를, 최종 라벨/카테고리에 맞춰 조금 더 일관성 있게 생성.
    - 프로세스 기반 카테고리([FM]/[NK] 등)가 잡힌 경우에는
      '구체적인 건수 기준은 체크리스트 표를 참고'하도록 유도
    - 카테고리 '기타' 또는 완전 non_issue 인 경우에는
      '즉시 공유 필요 가능성은 낮으나, 반복 여부를 모니터링' 쪽으로 통일
    실제 3건/5건 등의 숫자는 체크리스트 이미지에만 두고,
    코드 상 문구에는 구체 숫자를 넣지 않는다. (혼선 방지)
    """
    if not rule_result or not final_label:
        return ""

    category = (rule_result.get("category") or "").strip()

    is_process_based = category.startswith("[FM]") or category.startswith("[NK]")
    is_misc = (category == "기타") or category.startswith("기타 ")

    # ISSUE 케이스
    if final_label == "issue":
        if is_process_based:
            return (
                "이 케이스는 RULES/ML 기준으로 '이슈'에 가까운 사례입니다. "
                "구체적인 공유 기준(즉시 공유 / 누적 건수 기준 등)은 "
                "게임별 프로세스 체크리스트 표를 참고해 판단해 주세요."
            )
        else:
            return (
                "이 케이스는 RULES/ML 기준으로 '이슈'로 분류되었습니다. "
                "동일 유형 문의가 반복될 경우에는 체크리스트 기준에 따라 "
                "담당자와 공유해 주세요."
            )

    # NON ISSUE 케이스
    if is_misc:
        return (
            "현재로서는 RULES/ML 기준으로 '즉시 공유가 필요한 이슈'로 보기는 어렵습니다. "
            "다만 동일 유형 문의가 반복되는지 가볍게 모니터링해 주세요."
        )

    return (
        "현재로서는 RULES/ML 기준으로 '즉시 공유가 필요한 이슈' 가능성은 높지 않습니다. "
        "그래도 아래 결과 진단과 업무 체크리스트를 함께 보면서 "
        "공유 여부를 한 번 더 점검해 주세요."
    )


# ---------------------------------------------------
# 6) 로그 저장 (프로세스 설명 컬럼 제거)
# ---------------------------------------------------

def append_log(
    game: str,
    text: str,
    rule_result: Dict,
    ml_result: Dict,
    final_label: str,
    suspect_game_mismatch: bool,
) -> None:
    """
    결과를 CSV 로그 파일에 한 줄씩 추가.
    프로세스 설명 컬럼은 더 이상 사용하지 않음.

    (4순위 패치)
    - 게임 선택과 캐릭터 감지 결과가 어긋난 경우를
      suspect_game_mismatch 컬럼으로 함께 기록.
    """
    is_new_file = not os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(
                [
                    "timestamp",
                    "game",
                    "text",
                    "rule_label",
                    "rule_category",
                    "rule_characters",
                    "ml_label",
                    "ml_prob_issue",
                    "ml_prob_non_issue",
                    "final_label",
                    "suspect_game_mismatch",
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                game,
                text.replace("\n", " ").strip(),
                rule_result.get("label"),
                rule_result.get("category"),
                ",".join(rule_result.get("characters", [])),
                ml_result.get("label"),
                f"{ml_result.get('prob_issue', 0.0):.4f}",
                f"{ml_result.get('prob_non_issue', 0.0):.4f}",
                final_label,
                "1" if suspect_game_mismatch else "0",
            ]
        )


# ---------------------------------------------------
# 7) 한 번의 분류 요청을 처리하는 공통 함수
#    (5순위 패치: 피드백 덮어쓰기 포함)
# ---------------------------------------------------

def run_classification(game: str, content: str) -> Dict[str, Any]:
    """
    하나의 텍스트에 대해 RULES + ML + 피드백까지 모두 반영한
    최종 결과를 계산하고, 템플릿에서 바로 쓸 수 있는 dict로 반환.
    """
    # 1) RULES 기반 결과
    rule_result = rules_classify(game, content)

    # 2) ML 결과
    ml_result = predict_issue(content)

    # 3) 최종 라벨 결정 (맥락 플래그 포함)
    final_label = decide_final_label(rule_result, ml_result)

    # 4) 게임 선택(FM/NK) ↔ 캐릭터 검출 뒤틀림 체크
    mismatch_info = detect_character_game_mismatch(game, content)
    suspect_game_mismatch = bool(mismatch_info.get("mismatch", False))

    game_warning: Optional[str] = None
    if suspect_game_mismatch:
        game_warning = (
            f"주의: 선택한 게임({game})과 감지된 캐릭터명이 어긋난 것 같습니다. "
            "게임 선택 또는 텍스트 복사 구간을 다시 확인해 주세요."
        )

    # 5) 텍스트 해시 계산
    text_hash = get_text_hash(content)

    # 6) (5순위 패치) 피드백 여부 확인 및 최종 결과 덮어쓰기
    feedback_applied = False
    feedback_correct_label: Optional[str] = None
    feedback_correct_category: Optional[str] = None

    fb_info = load_feedback_for_hash(text_hash)
    if fb_info:
        # 정답 라벨
        fb_label = fb_info.get("correct_label", "").lower()
        if fb_label in ("issue", "non_issue"):
            final_label = fb_label
            feedback_applied = True
            feedback_correct_label = fb_label

        # 정답 카테고리
        fb_cat = fb_info.get("correct_category", "")
        if fb_cat:
            rule_result["category"] = fb_cat
            feedback_applied = True
            feedback_correct_category = fb_cat

    # 7) 교육용 요약/진단/공유 기준 문구 생성
    summary_text = build_summary_text(game, content, rule_result, ml_result)
    diagnostics = build_diagnostics(rule_result, ml_result, final_label)
    share_guideline = build_share_guideline(rule_result, final_label)

    # 8) 로그 저장 (피드백으로 덮어쓴 final_label 기준)
    append_log(
        game,
        content,
        rule_result,
        ml_result,
        final_label,
        suspect_game_mismatch,
    )

    return {
        "rule_result": rule_result,
        "ml_result": ml_result,
        "final_label": final_label,
        "summary_text": summary_text,
        "diagnostics": diagnostics,
        "game_warning": game_warning,
        "text_hash": text_hash,
        "feedback_applied": feedback_applied,
        "feedback_correct_label": feedback_correct_label,
        "feedback_correct_category": feedback_correct_category,
        "share_guideline": share_guideline,
    }


# ---------------------------------------------------
# 8) Flask 라우트
# ---------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    game = "FM"
    content = ""

    rule_result: Optional[Dict] = None
    ml_result: Optional[Dict] = None
    final_label: Optional[str] = None
    summary_text: Optional[str] = None
    diagnostics: Optional[Dict] = None
    game_warning: Optional[str] = None
    share_guideline: Optional[str] = None

    text_hash: Optional[str] = None
    feedback_applied: bool = False
    feedback_correct_label: Optional[str] = None
    feedback_correct_category: Optional[str] = None
    feedback_saved: bool = False

    if request.method == "POST":
        action = request.form.get("action", "classify")

        # -----------------------------------------------
        # A) 피드백 저장 처리
        # -----------------------------------------------
        if action == "feedback":
            game = request.form.get("game", "FM")
            content = request.form.get("content", "").strip()
            text_hash = request.form.get("text_hash", "").strip()

            if content:
                if not text_hash:
                    text_hash = get_text_hash(content)

                # 정답 라벨 (ISSUE / NON ISSUE → 내부적으로 issue / non_issue)
                fb_label_raw = request.form.get("feedback_label", "").strip().upper()
                if fb_label_raw == "ISSUE":
                    correct_label = "issue"
                elif fb_label_raw in ("NON ISSUE", "NON_ISSUE", "NONISSUE"):
                    correct_label = "non_issue"
                else:
                    correct_label = ""

                # 정답 카테고리 + 코멘트
                correct_category = request.form.get("feedback_category", "").strip()
                comment = request.form.get("feedback_comment", "").strip()

                # CSV에 저장
                save_feedback(
                    game=game,
                    text=content,
                    text_hash=text_hash,
                    correct_label=correct_label,
                    correct_category=correct_category,
                    comment=comment,
                )
                feedback_saved = True

                # 저장 직후, 동일 텍스트로 다시 분류 실행 → 피드백이 곧바로 반영된 상태를 보여줌
                class_data = run_classification(game, content)
                rule_result = class_data["rule_result"]
                ml_result = class_data["ml_result"]
                final_label = class_data["final_label"]
                summary_text = class_data["summary_text"]
                diagnostics = class_data["diagnostics"]
                game_warning = class_data["game_warning"]
                text_hash = class_data["text_hash"]
                feedback_applied = class_data["feedback_applied"]
                feedback_correct_label = class_data["feedback_correct_label"]
                feedback_correct_category = class_data["feedback_correct_category"]
                share_guideline = class_data["share_guideline"]

        # -----------------------------------------------
        # B) 일반 분류 실행
        # -----------------------------------------------
        else:
            game = request.form.get("game", "FM")
            content = request.form.get("content", "").strip()

            if content:
                class_data = run_classification(game, content)
                rule_result = class_data["rule_result"]
                ml_result = class_data["ml_result"]
                final_label = class_data["final_label"]
                summary_text = class_data["summary_text"]
                diagnostics = class_data["diagnostics"]
                game_warning = class_data["game_warning"]
                text_hash = class_data["text_hash"]
                feedback_applied = class_data["feedback_applied"]
                feedback_correct_label = class_data["feedback_correct_label"]
                feedback_correct_category = class_data["feedback_correct_category"]
                share_guideline = class_data["share_guideline"]

    return render_template(
        "index.html",
        game=game,
        content=content,
        rule_result=rule_result,
        ml_result=ml_result,
        final_label=final_label,
        summary_text=summary_text,
        diagnostics=diagnostics,
        game_warning=game_warning,
        text_hash=text_hash,
        feedback_applied=feedback_applied,
        feedback_correct_label=feedback_correct_label,
        feedback_correct_category=feedback_correct_category,
        feedback_saved=feedback_saved,
        share_guideline=share_guideline,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
