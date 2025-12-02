# -*- coding: utf-8 -*-
"""
issue_train.py

욕설/이벤트/접속 이슈 등 텍스트를
"issue / non_issue" 이진 분류하는 머신러닝 모델 학습 스크립트.

사용 예:
    python issue_train.py

기본 동작:
- issues_dataset.csv 파일을 읽음
  (컬럼 예시: NO, 게임명, 제목/내용, 이슈 판단, 검수 - 피드백)
- '이슈 판단' 값을 issue / non_issue 로 변환
  (O / X + 검수 피드백 문구를 함께 사용)
- TfidfVectorizer + LogisticRegression 파이프라인 학습
- issue_classifier.pkl 로 저장

주의:
- 실제 CSV 컬럼명/라벨 문자열이 다르면 아래 부분을 적절히 수정해야 함.
"""

import os
import argparse
from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# rules.py의 normalize_text를 그대로 재사용 (전처리 일관성 확보)
from rules import normalize_text

# ---------------------------------------------------
# 0) 경로 설정
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV / 모델을 전부 프로젝트 루트 기준으로 사용
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "issues_dataset.csv")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "issue_classifier.pkl")

# ---------------------------------------------------
# 1) 데이터 로딩 & 전처리
# ---------------------------------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    CSV에서 학습용 데이터를 읽어 온다.

    기본 가정 컬럼:
    - '게임명'
    - '제목/내용'
    - '이슈 판단'
    - '검수 - 피드백'  (라벨 보정용으로 사용)

    실제 컬럼명이 다르면 여기서 수정하면 됨.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = ["게임명", "제목/내용", "이슈 판단", "검수 - 피드백"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"CSV에 '{col}' 컬럼이 없습니다. 실제 컬럼명을 확인해 주세요.\n"
                f"현재 컬럼: {list(df.columns)}"
            )

    # 제목/내용, 이슈 판단이 비어 있는 행은 제거
    df = df.dropna(subset=["제목/내용", "이슈 판단"]).copy()

    return df


def make_label(row) -> str:
    """
    엑셀/CSV의 '이슈 판단' + '검수 - 피드백' 컬럼을 이용해
    최종 라벨(issue / non_issue)을 만든다.

    - 'O'  → issue
    - 'X'  → non_issue
    - 그 외 값이 들어오면, 검수-피드백 문구 안에
      '이슈가 아닌', '전달할 필요 없습니다' 등이 있으면 non_issue 로 보정.
    - 그래도 애매하면 보수적으로 issue 로 본다.
    """
    raw_val = str(row.get("이슈 판단", "")).strip().upper()
    feedback = str(row.get("검수 - 피드백", "")).strip()

    if raw_val == "O":
        return "issue"
    if raw_val == "X":
        return "non_issue"

    negative_patterns = [
        "이슈가 아닌",
        "전달할 필요 없습니다",
        "전달하지 않으셔도 됩니다",
        "이슈 전달 필요건이 아닙니다",
        "단순 문의로 판단",
    ]
    for pat in negative_patterns:
        if pat in feedback:
            return "non_issue"

    # 기본값은 issue 쪽으로 보수적으로 처리
    print(f"[경고] 알 수 없는 이슈 판단 값 '{raw_val}' → 임시로 'issue' 처리")
    return "issue"


def build_text_feature(row) -> str:
    """
    모델 입력용 텍스트를 구성.

    - 게임명을 앞에 prefix로 붙여서 도메인 힌트 제공
    - rules.normalize_text 를 사용해 공백/특수문자 제거 버전을 추가
      → 욕설/띄어쓰기 변형도 어느 정도 잡기 위함.

    예) "[FM] 씨 발 접속이 안 돼서 하나도 못 하겠습니다. [SEP] 씨발접속이안돼서하나도못하겠습니다"
    """
    game = str(row.get("게임명", "")).strip()
    content = str(row.get("제목/내용", "")).strip()

    if game:
        raw_text = f"[{game}] {content}"
    else:
        raw_text = content

    normalized = normalize_text(raw_text)

    # 원문 + SEP + 정규화 텍스트 조합
    return raw_text + " [SEP] " + normalized


def prepare_xy(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    DataFrame으로부터 X(text), y(label) 배열 생성.
    """
    # 라벨 생성 (O/X + 검수 피드백 기반)
    df["label"] = df.apply(make_label, axis=1)

    # 텍스트 생성
    df["text"] = df.apply(build_text_feature, axis=1)

    X = df["text"].tolist()
    y = df["label"].tolist()
    return X, y

# ---------------------------------------------------
# 2) 모델 정의 & 학습
# ---------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    TfidfVectorizer + LogisticRegression 파이프라인 구성.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigram + bigram
        min_df=2,                # 최소 2회 이상 등장 단어만 사용 (노이즈 감소)
        max_features=50000,      # 상한선 (필요시 조정)
    )

    clf = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",  # issue/non_issue 비율 차이가 큰 경우 균형 맞추기
    )

    pipe = Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", clf),
        ]
    )
    return pipe


def train_and_evaluate(X, y, model_path: str) -> Pipeline:
    """
    학습 + 간단 평가 + 모델 저장.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()

    print("▶ 모델 학습 시작...")
    pipeline.fit(X_train, y_train)
    print("▶ 모델 학습 완료.")

    # 평가
    print("\n===== 테스트 셋 평가 결과 =====")
    y_pred = pipeline.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))

    # 저장 (ml_utils 가 사용하는 파일명과 일치)
    joblib.dump(pipeline, model_path)
    print(f"\n✅ 모델 저장 완료: {model_path}")

    return pipeline

# ---------------------------------------------------
# 3) main
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Issue / Non-Issue 분류 모델 학습 스크립트")
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f"라벨 데이터 CSV 경로 (기본: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"저장할 모델 파일 경로 (기본: {DEFAULT_MODEL_PATH})",
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {args.csv}\n"
            f"실제 위치를 확인하거나 --csv 인자로 경로를 지정해 주세요."
        )

    print(f"▶ 데이터 로딩: {args.csv}")
    df = load_dataset(args.csv)

    print(f"▶ 샘플 수: {len(df)}")
    X, y = prepare_xy(df)

    print("\n===== 라벨 분포 =====")
    print(pd.Series(y).value_counts())

    train_and_evaluate(X, y, args.out)


if __name__ == "__main__":
    main()
