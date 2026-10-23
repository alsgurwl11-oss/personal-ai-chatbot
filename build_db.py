
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 프로젝트 경로 설정
BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "data" / "profile.txt"
DB_PATH = BASE_DIR / "chroma_db"

# 2. 개인정보 문서 읽기
with open(FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print("개인정보 파일 불러오기 성공!")

# 3. LangChain Document 객체 생성
documents = [
    Document(
        page_content=text,
        metadata={"source": "profile.txt"}
    )
]

# 4. 문서 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_docs = text_splitter.split_documents(documents)

print(f"문서 분할 완료: {len(split_docs)}개")

# 5. 임베딩 모델 불러오기
print("임베딩 모델 로딩 중...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 6. Chroma 데이터베이스 생성 및 저장
db = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    collection_name="personal_profile",
    persist_directory=str(DB_PATH)
)

print("Chroma 데이터베이스 생성 완료!")

# 7. 검색 테스트
query = "김민혁의 지도교수는 누구인가요?"

results = db.similarity_search(query, k=2)

print("\n===== 검색 결과 =====")

for i, doc in enumerate(results, start=1):
    print(f"\n검색 결과 {i}")
    print(doc.page_content)

print("\nRAG 데이터베이스 준비 완료!")
