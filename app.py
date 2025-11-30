# -*- coding: utf-8 -*-
"""
app.py

Flask 기반 웹 서버.
- "/" 페이지에서 게임 / 텍스트 입력
- RULES + 머신러닝 결과 동시에 표시
- 교육용 예시 기반 프로세스 설명도 함께 표시
- 로그 파일로 저장
- Training Mode(헷갈리는 예제 / 퀴즈) 제공
"""

import os
import csv
import random
from datetime import datetime

from flask import Flask, render_template, request

from ml_utils import predict_issue, suggest_process_description
from rules import (
    rules_classify,
    PROCESS_RULES,
    normalize_text,
    find_keyword_hits,
    make_process_category_name,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "prediction_log.csv")

os.makedirs(LOG_DIR, exist_ok=True)

app = Flask(__name__)


# ---------------------------------------------------
# 0) 교육용 헷갈리는 예제 데이터
# ---------------------------------------------------

CONFUSING_EXAMPLES = [
    {
        "id": 1,
        "game": "FM",
        "text": "씨 발 게임 접속이 안 돼서 하나도 못 하겠습니다.",
        "note": "욕설 + 접속 불가가 함께 있는 전형적인 사례. 접속/로그인 불가가 핵심 이슈.",
    },
    {
        "id": 2,
        "game": "FM",
        "text": "보상도 안 주고 씨발 버그도 안 고치냐.",
        "note": "보상 불만 + 버그 제보 + 욕설이 함께 있는 사례. 버그/오류 제보가 우선.",
    },
    {
        "id": 3,
        "game": "FM",
        "text": "이벤트 보상 구성이 너무 쓰레기네요. 욕은 안 하겠지만 이건 선 넘었습니다.",
        "note": "감정 표현은 있지만 핵심은 이벤트 보상 구성에 대한 불만.",
    },
    {
        "id": 4,
        "game": "NK",
        "text": "접속은 되는데 전투 들어가면 계속 튕기고 버그 걸려요. 진짜 욕 나올 지경.",
        "note": "전투 중 튕김/버그가 핵심. 접속 자체보다는 버그/오류 제보로 분류되는지 확인하는 사례.",
    },
    {
        "id": 5,
        "game": "NK",
        "text": "정치 얘기 하면서 특정 국가 비하까지 하고 있네요. 이런 글 좀 제재해 주세요.",
        "note": "정치/외교 갈등 + 비하 표현. 커뮤니티 내 S급 위험 요소 사례.",
    },
    {
        "id": 6,
        "game": "FM",
        "text": "그냥 접속이 조금 느린 것 같아요. 버그인지는 잘 모르겠습니다.",
        "note": "모호한 표현. 접속/네트워크 vs 버그/오류 경계에 있는 사례.",
    },
    {
        "id": 7,
        "game": "NK",
        "text": "결제는 됐는데 보상이 안 들어왔습니다. 욕 나오네요.",
        "note": "결제/보상 이슈 + 욕설. 결제/보상 관련 불만이 핵심 이슈.",
    },
    {
        "id": 8,
        "game": "FM",
        "text": "캐릭터가 안 움직이고 스킬도 안 나가요. 게임이 이게 뭐냐 진짜.",
        "note": "전형적인 전투 버그/오류 사례. 감정 표현보다 버그가 핵심.",
    },
]


def build_quiz_options_for_game(game: str) -> list[str]:
    """
    퀴즈용 드롭다운 옵션을 해당 게임(FM/NK)에 맞게 생성.
    RULES에서 쓰는 카테고리 문자열과 100% 동일하게 맞춘다.
    """
    options = {
        make_process_category_name(
            r["game"], r["process_name"], r["importance"], r["detail_name"]
        )
        for r in PROCESS_RULES
        if r["game"] == game
    }
    return sorted(options)


ABUSE_KEYWORDS = [
    "욕", "욕설", "비속어", "씨발", "ㅅㅂ", "개새", "패드립",
    "비하", "모욕", "인신공격", "막말",
]


def detect_abuse_hits(text: str):
    """
    요약 문구 생성을 위해 텍스트 내 욕설/비속어 여부만 간단히 검출.
    """
    text_compact = normalize_text(text)
    return find_keyword_hits(ABUSE_KEYWORDS, text, text_compact)


# ---------------------------------------------------
# 1) RULES + ML 종합 라벨 결정
# ---------------------------------------------------

def decide_final_label(rule_result: dict, ml_result: dict) -> str:
    """
    RULES 결과와 ML 결과를 종합해서 최종 라벨을 결정.

    교육용 정책:
    - RULES가 non_issue + 카테고리 '기타' 이면 무조건 non_issue
    - RULES가 issue이면 RULES를 최우선으로 신뢰
    - RULES가 non_issue 인데 ML이 issue라고 해도,
      → ML 확률이 0.9 이상일 때만 예외적으로 issue로 인정
    """

    rule_label = rule_result.get("label", "non_issue")
    rule_category = rule_result.get("category", "")
    ml_label = ml_result.get("label", "non_issue")
    issue_prob = ml_result.get("prob_issue", 0.0)

    if rule_label == "non_issue" and rule_category == "기타":
        return "non_issue"

    if rule_label == ml_label:
        return rule_label

    if rule_label == "issue":
        return "issue"

    if rule_label == "non_issue" and ml_label == "issue" and issue_prob >= 0.9:
        return "issue"

    return "non_issue"


# ---------------------------------------------------
# 2) 교육자용 요약 문구 자동 생성
# ---------------------------------------------------

def build_summary_text(
    game: str,
    text: str,
    rule_result: dict,
    ml_result: dict | None,
) -> str | None:
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

    parts: list[str] = []

    # 1) 핵심 이슈 문장
    if label == "issue":
        main_head = "핵심 이슈"
    else:
        main_head = "핵심 판단"

    main_type = None
    if "접속/로그인 불가" in category:
        main_type = "접속/로그인 불가(게임 접속 불가 / 서버 연결 실패)"
    elif "버그/오류 제보" in category:
        main_type = "버그/오류 제보"
    elif "보상/재화" in category or "보상/이벤트" in category:
        main_type = "결제/보상/재화 관련 이슈"
    elif "커뮤니티(" in category:
        main_type = "커뮤니티 게시글/댓글 관련 이슈"

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
# 3) 로그 저장
# ---------------------------------------------------

def append_log(
    game: str,
    text: str,
    rule_result: dict,
    ml_result: dict,
    final_label: str,
    process_desc: str | None,
) -> None:
    """
    결과를 CSV 로그 파일에 한 줄씩 추가.
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
                    "process_description",
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
                (process_desc or "").replace("\n", " "),
            ]
        )


# ---------------------------------------------------
# 4) Flask 라우트
# ---------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    game = "FM"
    content = ""
    rule_result = None
    ml_result = None
    final_label = None
    process_desc = None
    summary_text = None

    # Training Mode용 상태 변수
    training_example = None
    training_result = None
    training_explanation = None
    quiz_feedback = None
    quiz_options: list[str] = []

    if request.method == "POST":
        mode = request.form.get("mode", "classify")

        # (A) 일반 이슈 분류 모드
        if mode == "classify":
            game = request.form.get("game", "FM")
            content = request.form.get("content", "").strip()

            if content:
                # 1) RULES 기반 결과
                rule_result = rules_classify(game, content)

                # 2) ML 결과
                ml_result = predict_issue(content)

                # 3) 교육용 예시 기반 프로세스 설명 (텍스트 설명)
                process_desc = suggest_process_description(game, content)

                # rule_result 안에 교육용 설명만 추가 (카테고리/라벨은 건드리지 않음)
                if process_desc and isinstance(rule_result, dict):
                    rule_result["education_desc"] = process_desc

                # 4) 최종 라벨 결정
                final_label = decide_final_label(rule_result, ml_result)

                # 5) 요약 문구 생성 (교육자용)
                summary_text = build_summary_text(game, content, rule_result, ml_result)

                # 6) 로그 저장
                append_log(
                    game,
                    content,
                    rule_result,
                    ml_result,
                    final_label,
                    process_desc,
                )

        # (B) Training Mode - 헷갈리는 예제 불러오기
        elif mode == "example":
            ex = random.choice(CONFUSING_EXAMPLES)
            training_example = {
                "id": ex["id"],
                "game": ex["game"],
                "text": ex["text"],
            }
            training_result = rules_classify(ex["game"], ex["text"])
            training_explanation = ex.get("note")
            # 퀴즈 옵션은 해당 게임의 프로세스만 노출
            quiz_options = build_quiz_options_for_game(ex["game"])

        # (C) Training Mode - 퀴즈 답안 제출
        elif mode == "quiz":
            try:
                ex_id = int(request.form.get("example_id", "0"))
            except ValueError:
                ex_id = 0

            user_answer = request.form.get("answer_category", "")

            ex = next((e for e in CONFUSING_EXAMPLES if e["id"] == ex_id), None)
            if ex:
                training_example = {
                    "id": ex["id"],
                    "game": ex["game"],
                    "text": ex["text"],
                }
                training_result = rules_classify(ex["game"], ex["text"])
                correct_category = training_result.get("category", "")
                training_explanation = ex.get("note")
                # 퀴즈 옵션도 다시 세팅 (새로고침 없이 여러 번 푸는 경우용)
                quiz_options = build_quiz_options_for_game(ex["game"])

                if user_answer == correct_category:
                    quiz_feedback = "정답입니다. (RULES 기준 분류와 일치합니다.)"
                else:
                    quiz_feedback = (
                        "오답입니다. 위의 '툴 기준 정답'을 참고하여 "
                        "어떤 이슈가 더 우선인지 다시 한 번 확인해 주세요."
                    )

    return render_template(
        "index.html",
        game=game,
        content=content,
        rule_result=rule_result,
        ml_result=ml_result,
        final_label=final_label,
        process_desc=process_desc,
        summary_text=summary_text,
        training_example=training_example,
        training_result=training_result,
        training_explanation=training_explanation,
        quiz_feedback=quiz_feedback,
        quiz_options=quiz_options,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
