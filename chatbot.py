"""개인정보 챗봇 최종 점검 버전.

- PEFT Llama3: 검색한 로컬 개인정보를 참고하여 답변
- 원본 Llama3: 일반상식과 영어 번역
- 대화 히스토리: 직전 AI 답변의 번역에 사용
- 외부 OpenAI API 사용하지 않음
"""
from __future__ import annotations

import io
import csv
import re
from datetime import date
from pathlib import Path

import ollama
import streamlit as st

st.set_page_config(page_title="개인정보 AI 챗봇", page_icon="🦜")
st.title("🦜 개인정보 AI 챗봇")


ROOT = Path(__file__).resolve().parent
PROFILE = ROOT / "data" / "profile.txt"
CSV_PATH = ROOT / "data" / "indata_kor.csv"
PEFT_MODEL = "llama3-personal-fixed:latest"
BASE_MODEL = "llama3:latest"
NOT_FOUND = "해당 정보는 등록되어 있지 않습니다."



@st.cache_data(show_spinner=False)
def load_personal_data():
    # Cloud에서는 Streamlit Secrets 사용
    try:
        profile = st.secrets["profile_text"]
        csv_content = st.secrets["csv_text"]
    except (FileNotFoundError, KeyError):
        # 로컬에서는 기존 개인정보 파일 사용
        profile = PROFILE.read_text(encoding="utf-8-sig")
        csv_content = CSV_PATH.read_text(encoding="utf-8-sig")

    reader = csv.DictReader(io.StringIO(csv_content.lstrip("\ufeff")))
    rows = []

    for row in reader:
        question = (
            row.get("inputs")
            or row.get("instruction")
            or row.get("question")
            or ""
        ).strip()

        answer = (
            row.get("response")
            or row.get("output")
            or row.get("answer")
            or ""
        ).strip()

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
    q = norm(question)
    cats = categories(question)
    # 특정 인물을 지칭해도 질문 대상이 없으면 데이터에서 근거를 찾지 않습니다.
    if not cats:
        return False
    general_context = (
        "대한민국", "한국의", "세계", "일반적으로", "뜻", "정의", "원리",
        "무엇인가", "차이", "예시", "사용법", "역할", "수도", "국가", "나라"
    )
    if any(term in q for term in general_context):
        return False
    if any(term in q for term in ("김민혁", "민혁", "본인", "그학생", "그사람", "나의", "저의", "내가", "나는", "제가", "제")):
        return True
    # 사용자가 이름을 생략한 개인정보 질문 ("주소는?", "가족은?")
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
    """로컬 Llama3를 이용해 지정된 언어로 번역합니다."""
    if language not in ("English", "Korean"):
        raise ValueError("Unsupported translation language")
    output = ollama.chat(
        model=BASE_MODEL,
        messages=[
            {"role": "system", "content": (
                f"You are a professional translator. Translate the provided text into {language}. "
                "Return ONLY the translated text, without explanations. Preserve names and numbers. "
                "Never answer the original question; translate the provided answer exactly."
            )},
            {"role": "user", "content": f"Translate into {language}:\n\n{text}"},
        ],
        options={"temperature": 0, "num_ctx": 2048, "num_predict": 250},
    )
    return clean(output["message"]["content"])


def english_requested(question):
    q = norm(question)
    return any(x in q for x in ("영어로", "영문으로", "영어번역", "translate", "inenglish"))


def previous_answer_requested(question):
    q = norm(question)
    return any(x in q for x in ("방금", "직전", "이전답변", "마지막답변", "네답변", "너의답변", "아까", "그답변", "위답변"))


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
            last_reply = next((m["content"] for m in reversed(st.session_state.messages)
                               if m["role"] == "assistant"), None)
            if qn in ("ㅎㅇ", "안녕", "안녕하세요", "하이", "hi", "hello"):
                answer = "안녕하세요! 무엇을 도와드릴까요?"
            elif (lang := requested_translation_language(user_question)) and previous_answer_requested(user_question) and translation_intent(user_question):
                if last_reply:
                    with st.spinner("직전 답변 번역 중..."):
                        answer = translate(last_reply, lang)
                else:
                    answer = "아직 번역할 이전 답변이 없습니다."
            else:
                lang = requested_translation_language(user_question)
                current_question = remove_language_instruction(user_question) if lang else user_question
                current_question = current_question or user_question

                if age_requested(current_question):
                    answer = age_answer(current_question)
                elif intro_requested(current_question):
                    answer = introduction()
                elif is_personal(current_question):
                    facts, exact = related_personal_facts(current_question)
                    if not facts:
                        answer = NOT_FOUND
                    else:
                        direct = direct_fact(current_question, facts, exact)
                        if direct is not None:
                            answer = direct
                        else:
                            with st.spinner("PEFT 개인정보 답변 생성 중..."):
                                answer = generate_personal(current_question, facts)
                            if not answer:
                                answer = NOT_FOUND
                else:
                    with st.spinner("일반 질문 답변 생성 중..."):
                        answer = general_answer(current_question)

                if lang and answer != NOT_FOUND:
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
