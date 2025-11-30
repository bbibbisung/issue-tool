# -*- coding: utf-8 -*-
"""
rules.py

게임별 프로세스 + 캐릭터명 / 키워드를 이용한 RULES 기반 분류 로직.
- 1차: 프로세스(PROCESS_RULES) 기준으로 최대한 구체적으로 매칭
- 2차: 안 걸리는 경우 일반 카테고리(CATEGORY_RULES)로 처리

이번 버전에서 추가된 점:
- 각 프로세스 룰에 priority(우선순위) 필드를 추가
- 같은 문장에 여러 이슈가 섞여 있을 때,
  → priority가 가장 높은 프로세스를 먼저 선택
  → priority가 같으면, 매칭된 키워드 개수가 더 많은 룰 선택
- 접속/로그인 불가 이슈에 대해, 단어가 떨어져 있어도 잡아내는
  detect_connection_issue() 전용 로직 추가
- 공백/특수문자 제거한 텍스트까지 같이 검사해서
  "씨 발", "접 속 안 됨", "접속.안돼" 등도 인식 가능
- 어떤 룰에도 안 걸리면 기본 라벨을 non_issue 로 처리

이번 패치에서 추가된 점:
- 이벤트 관련 문장 중, 단순 일정/안내 질문을
  "보상/이벤트 불만"과 구분하기 위한 is_simple_event_question() 추가
- FM/NK 보상/이벤트/재화 불만 룰의 키워드를
  "불만 표현 위주"로 좁혀서 단순 '이벤트' 언급은 걸리지 않도록 조정
- CATEGORY_RULES의 '운영/보상 불만'에서도 단순 '이벤트' 키워드는 제거
- RULES에서 쓰는 카테고리 문자열을 한 곳에서 생성하는
  make_process_category_name() 유틸 추가
"""

from typing import List, Dict, Optional
import re


# ---------------------------------------------------
# 0) 텍스트 전처리 / 공통 키워드 매칭 유틸
# ---------------------------------------------------

def normalize_text(text: str) -> str:
    """
    공백/일부 특수문자를 제거한 버전 반환.
    예) '씨 발 접속. 안 돼' -> '씨발접속안돼'
    """
    if not isinstance(text, str):
        text = str(text)
    # 공백 + 일반적인 구분자/기호 제거
    return re.sub(r"[\s\.\,\!\?\~\-\_\+\=\(\)\[\]\{\}\\\/\|\"\'\:;]+", "", text)


def find_keyword_hits(keywords: List[str], text: str, text_compact: str) -> List[str]:
    """
    키워드 리스트에 대해 다음 두 가지 방식으로 매칭:
    - 원본 키워드(그대로) in 원본 text
    - 공백 제거한 키워드 in 공백/기호 제거 text_compact
    """
    hits: List[str] = []
    for kw in keywords:
        if not kw:
            continue
        kw_compact = re.sub(r"\s+", "", kw)  # 키워드 자체의 공백 제거
        if (kw in text) or (kw_compact and kw_compact in text_compact):
            hits.append(kw)
    return hits


def is_simple_event_question(text: str) -> bool:
    """
    '이벤트 언제 함?' 같은 단순 일정/안내 질문인지 판별하는 휴리스틱.

    조건:
    - '이벤트' 언급이 있고
    - '언제/시작/진행/열리' 등 질문 계열 단어가 포함되며
    - '보상 안/환불/쓰레기/최악' 등 강한 불만 단어는 포함되지 않을 것
    """
    if not text:
        return False

    t = text
    t_comp = normalize_text(text)

    # 1) 이벤트 언급 여부
    if "이벤트" not in t and "이벤트" not in t_comp:
        return False

    # 2) 질문/일정 계열 단어
    question_keywords = [
        "언제", "시작", "진행", "여나요", "하나요", "하는지",
        "했나요", "열려", "열리나요", "기간", "공지 떴나요",
    ]
    if not any(q in t for q in question_keywords):
        return False

    # 3) 강한 불만/부정 단어가 있으면 '단순 질문'으로 보지 않음
    negative_keywords = [
        "보상 안", "보상도 안", "보상은 안", "보상 왜",
        "환불", "쓰레기", "최악", "개같", "개판", "X같",
        "진짜 뭐냐", "운영이", "운영 진짜", "빡치네",
        "이딴", "해줘라", "안 줌", "안줌", "안 준", "안주네",
        "돈만 빨", "돈만빠는",
    ]
    if any(neg in t for neg in negative_keywords):
        return False

    return True


# ---------------------------------------------------
# 1) 게임별 캐릭터 리스트 (필요할 때 계속 추가 가능)
# ---------------------------------------------------

# 신월동행 (FM) - 일부 예시
FM_CHARACTERS: List[str] = [
    # 2.1 팀
    "팀장", "센슈", "프레이즈", "남교", "명상", "성랑", "스패로우", "총원", "준지",
    # 2.2 초현상 관리국 (일부)
    "정천", "항사", "칸나기", "카라스", "머피", "리프", "조안", "번하", "천마",
    "동맹", "은광", "계향", "거주", "아가네", "섬유",
]

# 니케 (NK) - 일부 예시
NK_CHARACTERS: List[str] = [
    # 엘리시온 (일부)
    "엠마", "프리바티", "시그널", "폴리", "미란다", "브리드", "슬린", "디젤",
    "베스티", "은하", "길로틴", "메이든", "헬름", "D", "네온", "마스트",
    # 미실리스 (일부)
    "맥스웰", "유니", "리타", "올리비아", "센티", "드레이크", "크로우", "페퍼",
    # 테트라 (일부)
    "슈가", "엑시아", "앨리스", "프림", "메어리", "밀크", "율하", "루드밀라",
    # 필그림 (일부)
    "스노우 화이트", "이사벨", "라푼젤", "홍련",
]


# ---------------------------------------------------
# 2) 프로세스 기준 RULES
# ---------------------------------------------------
# priority 예시 (높을수록 더 중요한 이슈):
#   100 : 접속/로그인 불가 같은 치명적인 장애
#    95 : 버그/오류 (게임이 비정상 작동)
#    90 : 불법 유출, 정치/극단 표현 등 S급 위험
#    80 : 결제/보상/재화
#    70 : 커뮤니티 비매너/욕설 등

PROCESS_RULES: List[Dict] = [
    # ===== NK : 커뮤니티(유료 글 관련 게시물 내 게시글 및 댓글) =====
    {
        "game": "NK",
        "process_name": "커뮤니티(유료 글 관련 게시물 내 게시글 및 댓글)",
        "importance": "A",
        "detail_name": "개인 조롱/비방/모욕 게시글",
        "label": "issue",
        "priority": 70,
        "keywords": [
            "조롱", "비방", "모욕", "인신공격", "비하", "인격 모독",
            "멍청이", "병신", "ㅄ", "패드립", "인신 공격", "욕했다", "욕함",
        ],
    },
    {
        "game": "NK",
        "process_name": "커뮤니티(유료 글 관련 게시물 내 게시글 및 댓글)",
        "importance": "A",
        "detail_name": "욕설/혐오/성적 표현 게시글",
        "label": "issue",
        "priority": 70,
        "keywords": [
            "욕설", "욕", "혐오", "성적", "야한", "음란", "19금", "선정적",
            "야짤", "야동", "노출", "성희롱", "성적인", "야하다",
        ],
    },
    {
        "game": "NK",
        "process_name": "커뮤니티(유료 글 관련 게시물 내 게시글 및 댓글)",
        "importance": "S",
        "detail_name": "콘텐츠 유출/불법 공유 게시글",
        "label": "issue",
        "priority": 90,
        "keywords": [
            "유출", "불법 공유", "불법 업로드", "불법 다운로드",
            "불법으로 올림", "파일 올렸", "영상 올렸", "녹화본 올렸",
            "스크린샷 공유 금지", "캡쳐본 공유", "캡처 공유",
        ],
    },
    {
        "game": "NK",
        "process_name": "커뮤니티(유료 글 관련 게시물 내 게시글 및 댓글)",
        "importance": "S",
        "detail_name": "정치·외교·군사 갈등 조장 / 극단적 표현",
        "label": "issue",
        "priority": 90,
        "keywords": [
            "정치", "외교", "군사", "극단", "테러", "범죄 미화",
            "전쟁하자", "죽여버리자", "학살", "증오", "증오 발언",
        ],
    },
    {
        "game": "NK",
        "process_name": "커뮤니티(유료 글 관련 게시물 내 게시글 및 댓글)",
        "importance": "A",
        "detail_name": "보상/재화 관련 불만 글",
        "label": "issue",
        "priority": 80,
        "keywords": [
            # 보상/재화에 대한 '불만' 위주
            "보상 안", "보상이 안", "보상도 안", "보상은 안", "보상 왜",
            "보상 안 줌", "보상 안주네", "보상 안 주네",
            "보상 안해줌", "보상 하나도", "보상 너무 짜",
            "재화 안 줌", "재화 안주네",
            "다이아 부족", "젬 부족", "젬도 안 줌",
            "유료 패키지", "패키지 가격", "패키지 너무 비싸",
            "결제 오류", "결제가 안되", "결제가 안 됨",
            "환불", "환불해", "환불 요청", "환불 좀",
            "과금 문제", "현질 유도", "현질만 시킴", "돈만 빨",
        ],
    },

    # ===== NK : 버그/오류 제보 =====
    {
        "game": "NK",
        "process_name": "버그/오류 제보",
        "importance": "A",
        "detail_name": "게임 내 기능/전투 등 오류/버그",
        "label": "issue",
        "priority": 95,
        "keywords": [
            "버그", "오류", "에러", "에러가 나", "튕김", "튕겨", "렉", "지연",
            "멈춤", "멈춰", "프리징", "강제 종료", "캐릭터가 안 움직",
            "스킬이 안 나가", "피해가 안 들어가", "데미지가 안 들어가",
            "퀘스트가 안 깨져", "퀘스트가 진행이 안",
        ],
    },

    # ===== NK : 접속/로그인 불가 =====
    {
        "game": "NK",
        "process_name": "접속/로그인 불가",
        "importance": "A",
        "detail_name": "게임 접속 불가 / 서버 연결 실패",
        "label": "issue",
        "priority": 100,
        "keywords": [
            "접속 안 되", "접속 안 됨", "접속 안 된다고", "접속이 안 되",
            "접속이 안 됨", "접속이 안 된다고",
            "접속 안되", "접속이 안되", "접속이안되", "접속안되",
            "게임 접속이 안", "게임이 안 들어가", "들어가면 튕김", "켜면 튕김",
            "실행 안 되", "실행이 안 되", "실행 안되", "실행이 안되",
            "실행이 안됨", "게임이 안 켜져", "게임이 안켜져",
            "로딩에서 멈춰", "로딩 안됨", "로딩이 안되", "로딩이 안 됨",
            "서버 접속", "서버 안되", "서버가 안", "서버 터짐",
            "서버 에러", "서버 오류",
            "네트워크 오류", "네트워크 불안정",
            "연결 실패", "연결 오류", "연결이 안되", "연결이 안 되",
        ],
    },

    # ===== FM : 커뮤니티(게시글 및 댓글) =====
    {
        "game": "FM",
        "process_name": "커뮤니티(게시글 및 댓글)",
        "importance": "A",
        "detail_name": "비매너/욕설/불건전 게시글",
        "label": "issue",
        "priority": 70,
        "keywords": [
            "비매너", "욕설", "모욕", "비하", "인신공격", "인신 공격",
            "욕했다", "욕함", "패드립", "막말", "비하 발언",
            "씨발", "ㅅㅂ", "개새",
        ],
    },
    {
        "game": "FM",
        "process_name": "커뮤니티(게시글 및 댓글)",
        "importance": "A",
        "detail_name": "보상/이벤트 관련 불만 글",
        "label": "issue",
        "priority": 80,
        "keywords": [
            # 보상/이벤트에 대한 '불만' 표현 위주
            "보상 안", "보상이 안", "보상도 안", "보상은 안", "보상 왜",
            "보상 안 줌", "보상 안주네", "보상 안 주네",
            "보상 안해줌", "보상 하나도", "보상 너무 짜",
            "이벤트 보상", "이벤트가 쓰레기", "이벤트 쓰레기",
            "이벤트 개판", "이벤트 X같", "이벤트 왜 이러",
            "환불", "환불해", "환불 요청", "환불 좀",
            "결제 오류", "결제가 안되", "결제가 안 됨",
            "과금 문제", "유료 패키지", "패키지 가격", "패키지 너무 비싸",
            "현질 유도", "현질만 시킴", "돈만 빨",
            "돌려달라",
        ],
    },

    # ===== FM : 버그/오류 제보 =====
    {
        "game": "FM",
        "process_name": "버그/오류 제보",
        "importance": "A",
        "detail_name": "게임 내 기능/전투 등 오류/버그",
        "label": "issue",
        "priority": 95,
        "keywords": [
            "버그", "오류", "에러", "에러가 나", "튕김", "튕겨", "렉", "지연",
            "멈춤", "멈춰", "프리징", "강제 종료", "캐릭터가 안 움직",
            "스킬이 안 나가", "피해가 안 들어가", "데미지가 안 들어가",
            "퀘스트가 안 깨져", "퀘스트가 진행이 안",
        ],
    },

    # ===== FM : 접속/로그인 불가 =====
    {
        "game": "FM",
        "process_name": "접속/로그인 불가",
        "importance": "A",
        "detail_name": "게임 접속 불가 / 서버 연결 실패",
        "label": "issue",
        "priority": 100,
        "keywords": [
            "접속 안 되", "접속 안 됨", "접속 안 된다고", "접속이 안 되",
            "접속이 안 됨", "접속이 안 된다고",
            "접속 안되", "접속이 안되", "접속이안되", "접속안되",
            "접속이 안", "접속 안 ", "게임 접속이 안",
            "게임이 안 들어가", "들어가면 튕김", "켜면 튕김",
            "실행 안 되", "실행이 안 되", "실행 안되", "실행이 안되",
            "실행이 안됨", "게임이 안 켜져", "게임이 안켜져",
            "로딩에서 멈춰", "로딩 안됨", "로딩이 안되", "로딩이 안 됨",
            "서버 에러", "서버 오류", "서버 안되", "서버가 안됨", "서버가 터짐",
            "네트워크 오류", "네트워크 불안정",
            "연결 실패", "연결 오류", "연결이 안되", "연결이 안 되",
        ],
    },
]


# ---------------------------------------------------
# 3) 일반 카테고리 (백업용)
# ---------------------------------------------------

CATEGORY_RULES: List[Dict] = [
    {
        "name": "욕설/비매너",
        "label": "issue",
        "keywords": [
            "욕", "욕설", "비속어", "ㅅㅂ", "씨발", "개새", "패드립",
            "비하", "모욕", "인신공격", "막말",
        ],
    },
    {
        "name": "결제/재화/보상",
        "label": "issue",
        "keywords": [
            "결제", "구매", "환불", "보상", "재화", "유료", "패키지", "과금",
            "현질", "충전", "결제 오류", "결제가 안되", "보상 안 줌",
        ],
    },
    {
        "name": "접속/네트워크",
        "label": "issue",
        "keywords": [
            "접속 안되", "접속이 안되", "접속 안 되", "접속이 안 됨",
            "실행 안되", "실행이 안되", "실행 안 돼", "실행이 안됨",
            "서버 에러", "서버 오류", "서버 터짐", "서버 안되",
            "네트워크 오류", "연결 실패", "연결 오류", "로딩 안됨",
        ],
    },
    {
        "name": "버그/오류",
        "label": "issue",
        "keywords": [
            "버그", "오류", "튕김", "렉", "지연", "멈춤", "프리징", "강제 종료",
            "이상 현상", "깨짐", "버그인지",
        ],
    },
    {
        "name": "계정/로그인",
        "label": "issue",
        "keywords": [
            "계정", "로그인", "연동", "탈퇴", "잠김", "정지", "차단",
            "비밀번호", "비번", "아이디 찾기",
        ],
    },
    {
        "name": "운영/보상 불만",
        "label": "issue",
        "keywords": [
            "운영", "공지", "보상", "패치", "공지사항", "유저 차별",
            "운영 진짜", "운영 개판", "운영팀",
            "이벤트 보상", "이벤트 운영", "이벤트가 쓰레기",
        ],
    },
    {
        "name": "단순 문의/건의",
        "label": "non_issue",
        "keywords": [
            "건의", "문의드립니다", "질문입니다", "알고 싶습니다",
            "궁금합니다", "혹시", "알려주실 수", "알려 주세요", "알려주세",
        ],
    },
]


# ---------------------------------------------------
# 4) 카테고리 문자열 생성 유틸 (RULES + 퀴즈 공통 사용)
# ---------------------------------------------------

def make_process_category_name(
    game: str,
    process_name: str,
    importance: str,
    detail_name: str,
) -> str:
    """
    RULES 결과, 퀴즈 선택지에서 공통으로 사용하는 카테고리 문자열 생성기.

    예) [FM] 버그/오류 제보 (A) - 게임 내 기능/전투 등 오류/버그
    """
    return f"[{game}] {process_name} ({importance}) - {detail_name}"


# ---------------------------------------------------
# 5) 내부 유틸 함수들
# ---------------------------------------------------

def detect_characters(game: str, text: str) -> List[str]:
    """문의 내용에서 캐릭터 명이 포함되어 있는지 검사."""
    if not text:
        return []

    names = FM_CHARACTERS if game == "FM" else NK_CHARACTERS
    found: List[str] = []

    for name in names:
        if name and name in text:
            found.append(name)

    # 중복 제거
    return list(dict.fromkeys(found))


def detect_connection_issue(text: str) -> bool:
    """
    접속/로그인 불가 전용 휴리스틱.

    - '접속/로그인/서버/네트워크/로딩/연결' 같은 접속 계열 단어가 하나 이상 있고
    - '안 되/안되/안됨/안돼/튕김/에러/오류/멈춤/안 들어가' 같은 실패 계열 단어가
      하나 이상 있으면 접속 이슈로 간주.
    """
    if not text:
        return False

    text_compact = normalize_text(text)

    conn_words = ["접속", "로그인", "서버", "네트워크", "로딩", "연결"]
    fail_words = [
        "안 되", "안되", "안됨", "안돼", "안 됨",
        "안 들어가", "못 들어가",
        "튕김", "튕겨", "에러", "오류",
        "멈춤", "멈춰", "꺼져", "강제 종료",
    ]

    has_conn = bool(find_keyword_hits(conn_words, text, text_compact))
    has_fail = bool(find_keyword_hits(fail_words, text, text_compact))
    return has_conn and has_fail


def apply_process_rules(game: str, text: str) -> Optional[Dict]:
    """
    프로세스 기준 RULES를 적용.

    - 같은 게임에 속한 프로세스 중
      1) priority가 가장 높은 룰을 우선
      2) priority가 같다면, 매칭된 키워드 수가 더 많은 룰 선택
    - 아무 것도 매칭되지 않으면 None 반환.
    """
    if not text:
        return None

    text_compact = normalize_text(text)
    simple_event_q = is_simple_event_question(text)

    best_rule: Optional[Dict] = None
    best_hits: List[str] = []
    best_priority: int = -1

    # 0) 접속/로그인 불가 휴리스틱 우선 적용
    if detect_connection_issue(text):
        for rule in PROCESS_RULES:
            if rule["game"] == game and rule["process_name"] == "접속/로그인 불가":
                best_rule = rule
                best_priority = rule.get("priority", 0)
                best_hits = find_keyword_hits(rule["keywords"], text, text_compact)
                break

    # 1) 일반 프로세스 룰 탐색
    for rule in PROCESS_RULES:
        if rule["game"] != game:
            continue

        # 이벤트 일정에 대한 단순 질문이면,
        # '보상/이벤트 관련 불만', '보상/재화 관련 불만' 류 룰은 건너뛴다.
        if simple_event_q and (
            "보상/이벤트 관련 불만" in rule.get("detail_name", "")
            or "보상/재화 관련 불만" in rule.get("detail_name", "")
        ):
            continue

        hits = find_keyword_hits(rule["keywords"], text, text_compact)
        if not hits:
            continue

        prio = rule.get("priority", 0)

        # priority가 더 높으면 교체
        # priority가 같다면, 매칭 키워드 개수가 더 많을 때 교체
        if (prio > best_priority) or (
            prio == best_priority and len(hits) > len(best_hits)
        ):
            best_rule = rule
            best_hits = hits
            best_priority = prio

    if best_rule is None:
        return None

    return {
        "process_name": best_rule["process_name"],
        "importance": best_rule["importance"],
        "detail_name": best_rule["detail_name"],
        "label": best_rule["label"],
        "matched_keywords": best_hits,
    }


def apply_category_rules(text: str) -> Dict:
    """
    일반 카테고리 룰 적용.
    첫 번째로 매칭되는 카테고리를 하나만 사용.
    매칭이 전혀 안 되면 '기타 / non_issue' 로 처리.
    """
    if not text:
        return {
            "name": "기타",
            "label": "non_issue",
            "matched_keywords": [],
        }

    text_compact = normalize_text(text)

    for rule in CATEGORY_RULES:
        hits = find_keyword_hits(rule["keywords"], text, text_compact)
        if hits:
            return {
                "name": rule["name"],
                "label": rule["label"],
                "matched_keywords": hits,
            }

    # 아무 것도 안 걸리면 기타 + non_issue
    return {
        "name": "기타",
        "label": "non_issue",
        "matched_keywords": [],
    }


# ---------------------------------------------------
# 6) 외부에서 쓰는 메인 함수
# ---------------------------------------------------

def rules_classify(game: str, text: str) -> Dict:
    """
    RULES 기반 최종 판단 함수.
    - 1차: 프로세스 매칭 (PROCESS_RULES)
    - 2차: 일반 카테고리 (CATEGORY_RULES)
    - 캐릭터 감지 정보 추가
    """
    if not isinstance(text, str):
        text = str(text)

    characters = detect_characters(game, text)

    # 1) 프로세스 룰 먼저 적용
    proc_info = apply_process_rules(game, text)

    if proc_info:
        category_name = make_process_category_name(
            game,
            proc_info["process_name"],
            proc_info["importance"],
            proc_info["detail_name"],
        )
        base_label = proc_info["label"]
        matched_keywords = proc_info["matched_keywords"]
    else:
        # 2) 프로세스로 못 잡았으면 일반 카테고리로
        cat_info = apply_category_rules(text)
        category_name = cat_info["name"]
        base_label = cat_info["label"]
        matched_keywords = cat_info["matched_keywords"]

    # 캐릭터가 포함되어 있으면 카테고리 문구에 표시 추가
    if characters:
        category_name = f"{category_name} + 캐릭터 관련"

    return {
        "label": base_label,        # "issue" / "non_issue"
        "category": category_name,  # 프로세스 기반 카테고리 문자열
        "matched_keywords": matched_keywords,
        "characters": characters,
    }
