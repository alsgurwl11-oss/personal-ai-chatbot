"""Build a separate Chroma index from data/indata_kor.csv.
Run: python build_qa_db.py
The existing profile TXT index (chroma_db) is not changed.
"""
import csv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "indata_kor.csv"
DB_PATH = BASE_DIR / "chroma_qa_db"
COLLECTION_NAME = "personal_qa"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"inputs", "response"}.issubset(reader.fieldnames):
            raise ValueError("CSV 첫 줄에 inputs,response 열이 필요합니다.")
        rows = [row for row in reader if row.get("inputs", "").strip() and row.get("response", "").strip()]

    if not rows:
        raise ValueError("CSV에 유효한 질문·답변 데이터가 없습니다.")

    # 질문만 검색용 벡터로 만들고, 정답은 별도 메타데이터로 보관합니다.
    docs = [
        Document(
            page_content=row["inputs"].strip(),
            metadata={"response": row["response"].strip(), "source": CSV_PATH.name},
        )
        for row in rows
    ]
    print(f"질문·답변 {len(docs)}개 읽기 완료")
    print("임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_PATH),
    )

    # 동일한 CSV로 재실행해도 데이터가 중복되지 않도록 고정된 ID를 사용합니다.
    ids = [f"qa_{i:04d}" for i in range(len(docs))]
    old_ids = db.get(include=[]).get("ids", [])
    if old_ids:
        db.delete(ids=old_ids)
    db.add_documents(documents=docs, ids=ids)
    print(f"Chroma 생성 완료: {DB_PATH}")

    query = "김민혁은 몇 년생이야?"
    results = db.similarity_search_with_score(query, k=1)
    if results:
        doc, distance = results[0]
        print(f"검색 테스트 성공 (거리: {distance:.4f})")
        print(f"검색된 질문: {doc.page_content}")
        print(f"등록된 답변: {doc.metadata['response']}")
    print("CSV RAG 데이터베이스 준비 완료!")


if __name__ == "__main__":
    main()
