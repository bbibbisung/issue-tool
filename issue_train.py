# -*- coding: utf-8 -*-
"""
issue_train.py

1) Excel 데이터(issues_dataset.xlsx)를 읽어서
2) issue / non_issue 라벨을 생성하고
3) Logistic Regression 모델을 학습한 뒤
4) issue_classifier.pkl 로 저장.
5) 동시에 "어떤 예시가 어떤 검수-피드백 설명을 갖는지" 를
   process_lookup.pkl 로 저장하여, 나중에 유사도 기반 프로세스 매칭에 사용.

같은 폴더에 issues_dataset.xlsx 파일을 두고 실행하세요.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "issues_dataset.xlsx")
CSV_PATH = os.path.join(BASE_DIR, "issues_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "issue_classifier.pkl")
LOOKUP_PATH = os.path.join(BASE_DIR, "process_lookup.pkl")


def make_label(row) -> str:
    """
    엑셀의 '이슈 판단' + '검수 - 피드백' 컬럼을 이용해
    최종 라벨(issue / non_issue)을 만듭니다.
    """
    value = str(row.get("이슈 판단", "")).strip().upper()

    if value == "O":
        return "issue"
    if value == "X":
        return "non_issue"

    # 혹시 모를 예외 상황: '이슈가 아닌' 등 문구로 판단
    feedback = str(row.get("검수 - 피드백", ""))
    negative_patterns = [
        "이슈가 아닌",
        "전달할 필요 없습니다",
        "전달하지 않으셔도 됩니다",
        "이슈 전달 필요건이 아닙니다",
    ]
    for pat in negative_patterns:
        if pat in feedback:
            return "non_issue"

    # 기본값은 issue 쪽으로 보수적으로 처리
    return "issue"


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"엑셀 파일을 찾을 수 없습니다: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    required_cols = ["게임명", "제목/내용", "이슈 판단", "검수 - 피드백"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"엑셀에 '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    # 라벨 생성
    df["label"] = df.apply(make_label, axis=1)

    # 텍스트 전처리(간단하게 줄바꿈 제거 정도만)
    df["text"] = df["제목/내용"].astype(str).str.replace(r"\s+", " ", regex=True)

    # CSV로도 저장 (백업 및 외부 확인용)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    return df


def train_and_save(df: pd.DataFrame):
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",  # 데이터 불균형 보정
                ),
            ),
        ]
    )

    print("📘 모델 학습 중...")
    pipeline.fit(X_train, y_train)

    print("📘 검증 데이터 평가:")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 1) 분류 모델 저장
    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ 학습 완료! 모델이 다음 위치에 저장되었습니다: {MODEL_PATH}")

    # 2) 프로세스 매칭용 lookup 데이터 생성 & 저장
    #    - 같은 tfidf 벡터 공간에서 예시 간 유사도를 계산하기 위함
    tfidf = pipeline.named_steps["tfidf"]
    X_all_tfidf = tfidf.transform(df["text"])

    lookup = {
        "texts": list(df["text"]),                 # 제목/내용 텍스트
        "games": list(df["게임명"]),               # 게임명 (NK / FM 등)
        "feedback": list(df["검수 - 피드백"]),      # 교육용 검수-피드백 설명
        "labels": list(df["label"]),              # issue / non_issue
        "X_tfidf": X_all_tfidf,                   # 전체 예시 벡터
    }

    joblib.dump(lookup, LOOKUP_PATH)
    print(f"✅ 프로세스 매칭용 lookup 데이터가 저장되었습니다: {LOOKUP_PATH}")


def main():
    print("1) 엑셀 데이터 로드 및 라벨 생성...")
    df = load_dataset()
    print(f"   - 총 샘플 수: {len(df)}")
    print(df["label"].value_counts())

    print("\n2) 모델 학습 및 lookup 데이터 생성...")
    train_and_save(df)


if __name__ == "__main__":
    main()
