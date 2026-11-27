"""개인정보 챗봇 최종 점검 버전.

- PEFT Llama3: 검색한 로컬 개인정보를 참고하여 답변
- 원본 Llama3: 일반상식과 영어 번역
- 대화 히스토리: 직전 AI 답변의 번역에 사용
- 외부 OpenAI API 사용하지 않음
"""
from __future__ import annotations

import csv
import io
import re
from datetime import date
from pathlib import Path

import ollama
import streamlit as st

# Cloud에서는 인증된 ngrok 게이트웨이를 사용하며, 로컬에서는 기존 Ollama 사용.
# 인증 정보와 개인정보는 코드/GitHub가 아닌 Streamlit Secrets에 보관합니다.
def optional_secret(key):
    try:
        return st.secrets.get(key)
    except (FileNotFoundError, KeyError):
        return None


gateway_url = optional_secret("ollama_gateway_url")
gateway_token = optional_secret("ollama_gateway_token")
if gateway_url and gateway_token:
    remote_ollama = ollama.Client(
        host=str(gateway_url).rstrip("/"),
        headers={"Authorization": f"Bearer {gateway_token}"},
        timeout=300,
    )
    ollama.chat = remote_ollama.chat
    ollama.generate = remote_ollama.generate


st.set_page_config(page_title="개인정보 AI 챗봇", page_icon="🦜")
st.title("🦜 개인정보 AI 챗봇")
st.caption("PEFT Llama3 · 개인정보 검색 · 일반 상식 · 이전 답변 번역 · 수정 v2")

ROOT = Path(__file__).resolve().parent
PROFILE = ROOT / "data" / "profile.txt"
CSV_PATH = ROOT / "data" / "indata_kor.csv"
PEFT_MODEL = "llama3-personal-fixed:latest"
BASE_MODEL = "llama3:latest"
NOT_FOUND = "해당 정보는 등록되어 있지 않습니다."
# 이 앱을 개발한 사람 (공개 GitHub에 이름을 노출하고 싶지 않으면 변경하세요)
APP_DEVELOPER = "김민혁"


@st.cache_data(show_spinner=False)
def load_personal_data():
    """Streamlit Cloud Secrets 우선, 로컬에서는 원본 파일 사용."""
    secret_profile = optional_secret("profile_text")
    secret_csv = optional_secret("csv_text")
    if secret_profile is not None and secret_csv is not None:
        profile = str(secret_profile)
        csv_content = str(secret_csv)
    else:
        profile = PROFILE.read_text(encoding="utf-8-sig")
        csv_content = CSV_PATH.read_text(encoding="utf-8-sig")
    rows = []
    reader = csv.DictReader(io.StringIO(csv_content.lstrip("\ufeff\r\n")))
    for row in reader:
        question = (row.get("inputs") or row.get("instruction") or row.get("question") or "").strip()
        answer = (row.get("response") or row.get("output") or row.get("answer") or "").strip()
        if question and answer:
            rows.append((question, answer))
    return profile, rows


try:
    profile_text, qa_rows = load_personal_data()
except Exception as exc:
    st.error("data/profile.txt 또는 data/indata_kor.csv 파일을 확인해 주세요.")
    st.caption(f"{type(exc).__name__}: {exc}")
    st.stop()


def norm(text):
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", str(text)).lower()


ALIASES = {
    "이름": ("이름", "성함"),
    "전공": ("전공", "학과", "무슨과", "학부"),
    "학교": ("학교", "대학교", "대학", "재학"),
    "학번": ("학번", "학생번호"),
    "학년": ("학년",),
    "생일": ("생일", "생년월일", "몇년생", "출생"),
    "주소": ("주소", "거주지", "사는곳", "어디살", "거주", "집이어디"),
    "가족": ("가족", "부모", "아버지", "어머니", "형제", "자매", "여동생", "남동생"),
    "연락처": ("전화번호", "휴대폰", "핸드폰", "연락처"),
    "이메일": ("이메일", "메일주소", "전자우편"),
    "취미": ("취미",), "특기": ("특기",),
    "축구팀": ("축구팀", "응원하는팀", "좋아하는팀"),
    "자격증": ("자격증",), "음식": ("좋아하는음식",),
    "게임": ("좋아하는게임",), "교수": ("지도교수", "담당교수"),
    "혈액형": ("혈액형",),
    "졸업": ("졸업", "졸업연도"),
}


def categories(text):
    q = norm(text)
    return {cat for cat, words in ALIASES.items() if any(norm(w) in q for w in words)}


def is_personal(question):
    cats = categories(question)
    if not cats:
        return False
    if is_general_context(question):
        return False
    return True



def read_profile():
    result = []
    section = ""
    for line in profile_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if (line.startswith("[") and line.endswith("]")) or line.startswith("#"):
            section = line.strip("[]# ").strip()
            continue
        parts = re.split(r"\s*[:：]\s*", line, maxsplit=1)
        if len(parts) == 2:
            result.append((section, parts[0].strip(" -•"), parts[1].strip()))
        else:
            result.append((section, "", line))
    return result


profile_fields = read_profile()


def related_personal_facts(question):
    """정확 일치 및 같은 개인정보 항목만 검색; 다른 분야 정보 혼합 금지."""
    q = norm(question)
    requested = categories(question)
    if not requested:
        return [], None

    for saved_q, saved_a in qa_rows:
        if norm(saved_q) == q:
            return [f"질문: {saved_q}\n답변: {saved_a}"], saved_a

    facts = []
    for section, field, value in profile_fields:
        label = norm(field)
        section_n = norm(section)
        if not value:
            continue
        matching = any(
            any(norm(alias) in label for alias in ALIASES[cat])
            for cat in requested
        )
        if matching:
            facts.append(f"{field}: {value}")
        elif not field and "가족" in requested and "가족" in section_n:
            facts.append(value)

    # 프로필의 '가족 관계' 구역에 세부 필드가 있는 경우도 포착
    if "가족" in requested and not facts:
        for section, field, value in profile_fields:
            if "가족" in norm(section) and value:
                facts.append(f"{field}: {value}" if field else value)

    qa_candidates = []
    for saved_q, saved_a in qa_rows:
        saved_cats = categories(saved_q)
        if requested == saved_cats:
            score = 10
            if norm(saved_q) in q or q in norm(saved_q):
                score += 20
            qa_candidates.append((score, f"질문: {saved_q}\n답변: {saved_a}"))
    qa_candidates.sort(key=lambda item: item[0], reverse=True)
    facts.extend(x for _, x in qa_candidates[:3])
    return list(dict.fromkeys(facts))[:5], None


def clean(text):
    for token in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>",
                  "### Instruction:", "### Input:", "### Response:"):
        text = text.split(token)[0]
    return text.strip()


def generate_personal(question, facts):
    """PEFT 학습 때 쓰던 Alpaca 형식과 Ollama의 raw 입력 템플릿을 맞춤."""
    evidence = "\n".join(facts)[:1300]
    prompt = (
        "Below is an instruction that describes a task,\n"
        "paired with an input that provides further context.\n"
        "Write a response that appropriately completes the request.\n"
        "답변은 한국어로 작성하세요.\n\n"
        "### Instruction:\n"
        "아래 참고 자료에서 현재 질문의 정답을 찾으세요. "
        "관련이 없는 사실은 답하지 마세요. 모르면 '해당 정보는 등록되어 있지 않습니다.'라고 답하세요.\n\n"
        "### Input:\n"
        f"참고 자료:\n{evidence}\n\n현재 질문: {question}\n\n"
        "### Response:\n"
    )
    output = ollama.generate(
        model=PEFT_MODEL,
        prompt=prompt,
        options={
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 120,
            "stop": ["<|end_of_text|>", "<|eot_id|>", "<|im_end|>",
                     "### Instruction:", "### Input:", "### Response:"],
        },
    )
    return clean(output["response"])


def chatbot_identity_answer(question):
    """앱 개발자와 기반 LLM 개발사를 구별해서 사실대로 응답합니다.

    챗봇의 제작 정보는 LLM에 맡기지 않습니다(임의 생성 방지).
    """
    q = norm(question)
    asks_creator = any(w in q for w in (
        "누가만들", "누가개발", "누가제작", "누가창조",
        "만든사람", "만든분", "만든이는", "개발자는", "제작자는",
        "누구가만들", "어디서만들", "누가만든"
    )) or (
        any(w in q for w in ("만들", "개발", "제작", "창조"))
        and any(w in q for w in ("누구", "누가", "어디", "회사"))
    )
    self_reference = any(w in q for w in (
        "너", "넌", "널", "네가", "니가", "당신", "너희", "너를",
        "챗봇", "이봇", "이ai", "이프로그램", "이서비스"
    ))
    if "llama3" in q and asks_creator and not self_reference:
        return "기반 언어 모델 Llama3는 Meta에서 개발했습니다."
    if self_reference and asks_creator:
        return f"{APP_DEVELOPER}입니다."
    if (q in ("너는누구야", "넌누구야", "너누구야", "너는뭐야", "넌뭐야", "넌누구니")
            or (self_reference and any(x in q for x in ("정체", "무슨모델", "어떤모델")))):
        return (f"저는 {APP_DEVELOPER} 님이 개발한 개인 맞춤형 AI 챗봇입니다. "
                "Meta의 Llama3 기반 모델을 Ollama로 실행합니다.")
    return None


def general_answer(question):
    # 기존 개인정보 대화를 전달하지 않아 상식 질문에 주소/전공이 섞이지 않도록 합니다.
    output = ollama.chat(
        model=BASE_MODEL,
        messages=[
            {"role": "system", "content": "당신은 한국어로 짧고 정확하게 답하는 AI 도우미입니다. 질문에만 답하세요."},
            {"role": "user", "content": question},
        ],
        options={"temperature": 0.1, "num_ctx": 2048, "num_predict": 180},
    )
    return clean(output["message"]["content"])


def translate(text, language="English"):
    """답변 문자열만 번역하며 빈 문자열/None/엉뚱한 응답을 출력하지 않습니다."""
    if language not in ("English", "Korean"):
        raise ValueError("Unsupported translation language")
    source = clean(str(text or "")).strip()
    if not source or source.casefold() in ("none", "null"):
        return "번역할 이전 답변이 없습니다."

    # 매우 짧은 사실 응답은 생성형 모델이 문장을 새로 만들지 않도록 확정 번역.
    normalized = re.sub(r"[\s.!?。]+$", "", source).strip()
    known_english = {
        "서울입니다": "It is Seoul.",
        "서울이에요": "It is Seoul.",
        "서울": "Seoul.",
        "대한민국의 수도는 서울입니다": "The capital of South Korea is Seoul.",
        "한국의 수도는 서울입니다": "The capital of South Korea is Seoul.",
        "안녕하세요": "Hello.",
        "축구 시청": "Watching soccer.",
        "축구 시청입니다": "My hobby is watching soccer.",
    }
    if language == "English" and normalized in known_english:
        return known_english[normalized]

    system = (
        f"You are a professional {language} translator, NOT a chat assistant. "
        "Translate only the exact SOURCE text supplied by the user. "
        "Never reply to its contents, never add information, never continue a dialogue. "
        "Keep names, numbers, and meanings intact. Output translation only."
    )
    output = ollama.chat(
        model=BASE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Translate this SOURCE to {language}:\n<source>\n{source}\n</source>"},
        ],
        options={"temperature": 0, "num_ctx": 2048, "num_predict": 200},
    )
    response = clean(str(output.get("message", {}).get("content") or "")).strip()
    invalid = {"none", "null", "undefined", "n/a", ""}
    if response.casefold().strip(" .!\n") in invalid:
        return "번역 결과를 생성하지 못했습니다. 다시 요청해 주세요."
    # 모델이 원문의 지시를 실행하며 내놓는 흔한 환각 문구를 차단합니다.
    if language == "English" and ("let's start now" in response.lower() or "lets start now" in response.lower()) and "시작" not in source:
        return "번역 결과가 원문과 맞지 않습니다. 다시 요청해 주세요."
    return response


def english_requested(question):
    q = norm(question)
    return any(x in q for x in ("영어로", "영문으로", "영어번역", "translate", "inenglish"))


def previous_answer_requested(question):
    q = norm(question)
    return any(x in q for x in (
        "방금", "직전", "이전답변", "마지막답변", "네답변", "너의답변",
        "아까", "그답변", "위답변", "위의말", "위에말", "위에한말", "위의답변", "위문장", "그말", "직전말",
        "방금한말", "앞의말", "앞의답변", "방금한답변", "이말", "이답변",
        "첫번째", "첫째", "두번째", "둘째", "세번째", "셋째"
    )) or bool(re.search(r"\d+번째", q))


def remove_english_instruction(question):
    q = re.sub(r"(?:영어로|영문으로)\s*(?:번역해서|번역해|바꿔서|바꿔|말해줘|말해|알려줘|해줘|써줘|답해줘)?", "", question, flags=re.IGNORECASE)
    q = re.sub(r"(?:translate\s+into\s+english|in\s+english)", "", q, flags=re.IGNORECASE)
    return q.strip(" \t.?!,，")


def intro_requested(question):
    q = norm(question)
    # "김민혁에 대해 소개"처럼 정보 항목을 따로 지정하지 않은 요약 요청
    return any(k in q for k in ("소개", "자기소개", "어떤사람", "누구인지알려")) and any(
        k in q for k in ("김민혁", "민혁", "나를", "저를", "내소개", "자기소개")
    )


def exact_qa_answer(question):
    key = norm(question)
    for saved_q, saved_a in qa_rows:
        if norm(saved_q) == key:
            return saved_a
    return None


def introduction():
    """사전 등록된 답변만 이용. 주소/전화/이메일 등 민감 정보는 요약에서 제외."""
    wanted = [
        ("전공", ("김민혁의 전공은?", "무슨 전공이지?")),
        ("재학", ("김민혁은 몇 학년이야?", "김민혁의 대학교는?")),
        ("취미", ("취미가 뭐야?",)),
        ("특기", ("특기는 뭐야?",)),
        ("관심 분야", ("관심 있는 IT 분야는?",)),
    ]
    snippets = []
    for title, questions in wanted:
        for question in questions:
            answer = exact_qa_answer(question)
            if answer and answer != NOT_FOUND:
                snippets.append(f"{title}: {answer.rstrip('. ')}")
                break
    if not snippets:
        return NOT_FOUND
    return "김민혁에 대해 등록된 정보는 다음과 같습니다. " + "; ".join(snippets) + "."


def requested_translation_language(question):
    q = norm(question)
    if any(x in q for x in ("한국어로", "한글로", "우리말로", "한국어번역", "국문으로", "inkorean", "tokorean")):
        return "Korean"
    if english_requested(question):
        return "English"
    return None


def translation_intent(question):
    q = norm(question)
    return any(x in q for x in ("번역", "바꿔", "옮겨", "말해줘", "알려줘", "해줘", "써줘", "translate"))


def remove_language_instruction(question):
    q = re.sub(r"(?:영어로|영문으로|한국어로|한글로|우리말로|국문으로)\s*(?:번역해서|번역해|바꿔서|바꿔|말해줘|말해|알려줘|해줘|써줘|답해줘)?", "", question, flags=re.IGNORECASE)
    q = re.sub(r"(?:translate\s+into\s+(?:english|korean)|in\s+(?:english|korean)|to\s+korean)", "", q, flags=re.IGNORECASE)
    return q.strip(" \t.?!,，")



def format_personal_answer(question, answer):
    """국문 단일 학번/연락처 답변만 문장형으로 만듭니다."""
    if not answer or answer == NOT_FOUND or "\n" in answer:
        return answer
    requested = categories(question)
    if len(requested) == 1 and ("학번" in requested or "연락처" in requested):
        if not answer.strip().endswith(("입니다.", "입니다", "이에요.", "예요.")):
            return answer.strip().rstrip(".") + "입니다."
    return answer


def direct_fact(question, facts, exact):
    if exact:
        return exact
    requested = categories(question)
    if len(requested) != 1:
        return None
    cat = next(iter(requested))
    # 단일, 명확한 필드만 직접 답변. 복잡한 질문은 PEFT 생성 모델에 맡김.
    candidates = []
    for _, field, value in profile_fields:
        if field and value and any(norm(a) in norm(field) for a in ALIASES[cat]):
            candidates.append(value)
    if len(candidates) == 1:
        return candidates[0]
    return None




def age_requested(question):
    """나이·몇 살·연세 질문만 감지. 생일·몇 년생 질문과 구분합니다."""
    q = norm(question)
    return any(word in q for word in ("나이", "몇살", "몇세", "연세", "만몇살", "만나이"))


def extract_birthday(value):
    """생년월일 자료에 있는 날짜만 읽습니다 (임의의 숫자 추측 금지)."""
    pattern = r"(?<!\d)((?:19|20)\d{2})\s*(?:년|[-./])\s*(\d{1,2})\s*(?:월|[-./])\s*(\d{1,2})\s*(?:일)?"
    match = re.search(pattern, value)
    if not match:
        return None
    try:
        return date(*(int(v) for v in match.groups()))
    except ValueError:
        return None


def registered_birthday():
    """프로필과 생일 Q&A에서 출생일을 확인하고, 불일치하면 계산을 보류합니다."""
    found = set()
    birthday_aliases = ("생년월일", "생일", "출생일")

    for section, field, value in profile_fields:
        if field and any(alias in norm(field) for alias in birthday_aliases):
            parsed = extract_birthday(value)
            if parsed:
                found.add(parsed)

    for saved_q, saved_a in qa_rows:
        if any(alias in norm(saved_q) for alias in birthday_aliases):
            parsed = extract_birthday(saved_a)
            if parsed:
                found.add(parsed)

    if len(found) > 1:
        return None, "프로필과 학습 데이터의 생년월일이 서로 달라 나이를 계산할 수 없습니다. 등록 정보를 확인해 주세요."
    if not found:
        return None, NOT_FOUND
    return next(iter(found)), None


def age_answer(question, today=None):
    """사전 저장된 나이 문구 대신 생년월일로 매번 만 나이를 계산합니다."""
    birthday, error = registered_birthday()
    if error:
        return error
    today = today or date.today()
    if birthday > today:
        return "등록된 생년월일이 현재 날짜보다 미래여서 나이를 계산할 수 없습니다."
    age = today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))
    as_of = f"{today.year}년 {today.month}월 {today.day}일 기준"
    birth_label = f"{birthday.year}년 {birthday.month}월 {birthday.day}일생"
    if "세는나이" in norm(question) or "한국식나이" in norm(question):
        return f"{as_of} 세는나이는 {today.year - birthday.year + 1}세입니다. ({birth_label})"
    return f"{as_of} 만 {age}세입니다. ({birth_label})"


def blood_type_requested(question):
    """혈액형만 묻는 개인정보 질문인지 검사합니다."""
    return categories(question) == {"혈액형"} and not is_general_context(question)


def blood_type_answer():
    """혈액형 항목에 명시된 값만 인정합니다. 학번 등의 잘못된 연결은 거부합니다."""
    blood_value = re.compile(
        r"(?i)^\s*(?:(?:제|저의|내|본인)\s*)?"
        r"(?:(?:혈액형)\s*(?:은|는|이|가)?\s*)?"
        r"(?:(Rh\s*[+-])\s*)?(AB|A|B|O)\s*형"
        r"(?:\s*(?:입니다|이에요|예요|이다))?\s*\.?\s*$"
    )
    values = set()

    def parse(value):
        match = blood_value.fullmatch(str(value).strip())
        if not match:
            return None
        rh, abo = match.groups()
        prefix = re.sub(r"\s+", "", rh.upper()) if rh else ""
        return f"{prefix}{abo.upper()}형"

    for _, field, value in profile_fields:
        if "혈액형" in norm(field):
            item = parse(value)
            if item:
                values.add(item)

    for saved_q, saved_a in qa_rows:
        if categories(saved_q) == {"혈액형"}:
            item = parse(saved_a)
            if item:
                values.add(item)

    return next(iter(values)) if len(values) == 1 else NOT_FOUND


# 실제 질문에 등장한 명시적 연결어만 절단합니다.
# 예: '내 주소와 대한민국 수도는?', '내 전공과 취미를 알려줘'
COMPOUND_ITEMS = (
    "전화번호", "휴대폰", "핸드폰", "연락처", "생년월일", "혈액형", "이메일",
    "학생번호", "학번", "전공", "학과", "학교", "대학교", "학년", "주소",
    "거주지", "가족", "취미", "특기", "자격증", "나이", "생일", "수도",
    "게임", "축구팀", "이름", "음식", "졸업"
)
NEXT_ITEMS = COMPOUND_ITEMS + (
    "내", "나의", "저의", "제", "대한민국", "한국", "일반적으로", "세계", "국가", "나라"
)
_COMPOUND_PATTERN = re.compile(
    r"(" + "|".join(map(re.escape, sorted(COMPOUND_ITEMS, key=len, reverse=True))) + r")"
    r"(?:와|과)\s*(?=(?:" + "|".join(map(re.escape, sorted(NEXT_ITEMS, key=len, reverse=True))) + r"))"
)


def split_compound_question(question):
    """명시적인 나열/접속 형태에 한해 질문을 독립 항목으로 나눕니다."""
    marked = _COMPOUND_PATTERN.sub(r"\1 ||| ", question)
    marked = re.sub(r"\s*(?:그리고|및)\s*", " ||| ", marked)
    return [part.strip(" ,?!.\t") for part in marked.split("|||") if part.strip(" ,?!.\t")]


def is_general_context(question):
    q = norm(question)
    return any(term in q for term in (
        "대한민국", "한국의", "세계", "일반적으로", "뜻", "정의", "원리",
        "무엇인가", "차이", "예시", "사용법", "역할", "수도", "국가", "나라",
        "무엇을배우", "어떤것을배우", "무슨뜻"
    ))


def personal_category_answer(cat, question):
    """복합 질문은 항목별로 검색하고 교차 혼합 및 모델의 근거 없는 생성을 막습니다."""
    if cat == "혈액형":
        return blood_type_answer()
    if cat == "생일" and age_requested(question):
        return age_answer(question)
    matches = []
    for section, field, value in profile_fields:
        if not value:
            continue
        if field and any(norm(a) in norm(field) for a in ALIASES[cat]):
            matches.append(value)
        elif not field and cat == "가족" and "가족" in norm(section):
            matches.append(value)
    if not matches:
        for saved_q, saved_a in qa_rows:
            if categories(saved_q) == {cat} and saved_a != NOT_FOUND:
                matches.append(saved_a)
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    return NOT_FOUND


def answer_one_question(question):
    """각 질문을 올바른 모델/데이터 경로로 라우팅합니다."""
    identity = chatbot_identity_answer(question)
    if identity is not None:
        return identity
    if age_requested(question) and len(categories(question)) <= 1:
        return age_answer(question)
    if blood_type_requested(question):
        return blood_type_answer()
    if intro_requested(question):
        return introduction()
    if is_personal(question):
        facts, exact = related_personal_facts(question)
        if not facts:
            return NOT_FOUND
        direct = direct_fact(question, facts, exact)
        if direct is not None:
            return direct
        with st.spinner("PEFT 개인정보 답변 생성 중..."):
            return generate_personal(question, facts) or NOT_FOUND
    with st.spinner("일반 질문 답변 생성 중..."):
        return general_answer(question)


def answer_compound_question(question):
    """개인정보와 일반 상식을 각각 처리하고 결과만 조합합니다."""
    parts = split_compound_question(question)
    if len(parts) <= 1:
        cats = categories(question)
        # 분할이 어려운 나열은 개인정보 항목별로 안전하게 처리합니다.
        if len(cats) >= 2 and not is_general_context(question):
            lines = []
            for cat in ALIASES:
                if cat in cats:
                    value = personal_category_answer(cat, question)
                    lines.append(f"{cat}: {value}")
            return "\n".join(lines)
        return answer_one_question(question)

    outputs = []
    for part in parts:
        cats = categories(part)
        if len(cats) == 1 and not is_general_context(part):
            cat = next(iter(cats))
            value = personal_category_answer(cat, part)
            outputs.append(f"{cat}: {value}")
        elif len(cats) > 1 and not is_general_context(part):
            for cat in ALIASES:
                if cat in cats:
                    outputs.append(f"{cat}: {personal_category_answer(cat, part)}")
        else:
            outputs.append(answer_one_question(part))
    return "\n".join(outputs)


def history_target(question, messages):
    """직전 답변 외에 첫 번째/두 번째/N번째 AI 답변도 지정할 수 있습니다."""
    answers = [m["content"] for m in messages if m.get("role") == "assistant"]
    if not answers:
        return None
    q = norm(question)
    if any(x in q for x in ("첫번째", "첫째", "첫번", "처음답변")):
        index = 1
    elif any(x in q for x in ("두번째", "둘째", "두번")):
        index = 2
    elif any(x in q for x in ("세번째", "셋째")):
        index = 3
    else:
        match = re.search(r"(\d+)번째", q)
        index = int(match.group(1)) if match else None
    if index is not None:
        return answers[index - 1] if 1 <= index <= len(answers) else None
    return answers[-1]


def needs_previous_answer(question):
    return bool(requested_translation_language(question) and previous_answer_requested(question)
                and translation_intent(question))


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input("질문을 입력해 주세요"):
    with st.chat_message("user"):
        st.write(user_question)
    with st.chat_message("assistant"):
        try:
            qn = norm(user_question)
            if qn in ("ㅎㅇ", "안녕", "안녕하세요", "하이", "hi", "hello"):
                answer = "안녕하세요! 무엇을 도와드릴까요?"
            elif needs_previous_answer(user_question):
                target = history_target(user_question, st.session_state.messages)
                if target is None:
                    answer = "요청한 이전 답변을 찾을 수 없습니다."
                else:
                    lang = requested_translation_language(user_question)
                    with st.spinner("이전 답변 번역 중..."):
                        answer = translate(target, lang)
            else:
                lang = requested_translation_language(user_question)
                current_question = remove_language_instruction(user_question) if lang else user_question
                current_question = current_question or user_question
                answer = answer_compound_question(current_question)
                # 국문 응답에만 문장 어미를 붙입니다. 번역된 영어에 한글이 남지 않게 합니다.
                if lang is None:
                    answer = format_personal_answer(current_question, answer)
                elif answer != NOT_FOUND:
                    with st.spinner("번역 중..."):
                        answer = translate(answer, lang)

            st.write(answer)
            st.session_state.messages.extend((
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": answer},
            ))
        except Exception as exc:
            st.error("모델 실행 중 오류가 발생했습니다. Ollama 서버와 모델 이름을 확인해 주세요.")
            with st.expander("오류 상세"):
                st.code(f"{type(exc).__name__}: {exc}")

st.sidebar.caption("개인정보: PEFT Llama3 | 일반상식·번역: Llama3")
if st.sidebar.button("🗑️ 대화 초기화", use_container_width=True):
    st.session_state.messages = []
    st.rerun()
