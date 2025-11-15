# -*- coding: utf-8 -*-

#11월 1일 
#출처: https://wikidocs.net/259208 위키독스 - LLM-AS-Judge , https://docs.smith.langchain.com/evaluation/faq/evaluator-implementations -랜체인 공식 문서 evaluation
# PDFRAG 관련: https://wikidocs.net/265516
# 임베딩 기반 평가방식 - https://wikidocs.net/259210

#현재 RAG 모듈은 각각 LIMBUS_NEWSbot_v1.py (봇구동 파일), pdf_embedding.py(PDF문서 임베딩), querying_utf8.py(사용자 입력 처리)
#세 가지의 파일로 모듈화 시킨 관계로 RAG 모듈 객체를 가져오는 것은 어려움 
#그러므로, 모듈들의 객체중 검색기의 기능을 지닌 QA_chin의 객체 및 관련 코드들을 가져와서 사용함
#따라서, LLM-AS-Judge 기법 및  임베딩 기반  평가방식과  유사한 방식의 평가를 진행 
#이전에 langsmith으로 생성한 합성 데이터를 가져와서 사용함 

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


from langchain_teddynote import logging

logging.langsmith("Limbus-Evaluation")

from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langsmith.evaluation import evaluate, LangChainStringEvaluator

pdf_filepath = 'LIMBUS_INFO2.pdf'
loader = PyPDFLoader(pdf_filepath)
docs= loader.load()

embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
index = FAISS.from_documents(docs, embeddings)
retriever = index.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY, verbose=True)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        """
[기본 도메인 상식]
- 수감자(플레이어블 12명): 이상, 파우스트, 돈키호테, 료슈, 뫼르소, 홍루, 히스클리프, 이스마엘, 로쟈, 싱클레어, 오티스, 그레고르
- 인격 등급: [0](1성), [00](2성), [000](3성)
- E.G.O는 인격과 별개 시스템(등급 예: ZAYIN/TETH/HE/WAW 등)
- '발푸르기스의 밤(발푸밤)'은 독립 분류의 이벤트 (정가/추출 규칙이 시즌과 다름)

[정가·추출 핵심 규칙]
- 시즌5 인격·E.G.O는 정가/추출 불가(자판기 정가는 직전 시즌이 막힘).
- 시즌6 인격·E.G.O는 정가·추출 가능. 단, 자판기 '정가'는 출시 1주 후부터 가능.
- 이벤트(시즌별 이벤트/배포)는 정가 가능하더라도, 추출은 '특정 조건'을 만족해야 가능(수감자 특정 추출, 시즌 N 추출권 등).
- 발푸밤: 이벤트 기간(2주) 내 1~5회 정가/추출 가능, 6회부터는 정가 불가·추출만 가능. 이벤트 종료 후 다음 발푸밤 전까지 획득 불가. 막 출시된 발푸밤 인격/E.G.O는 자판기 정가가 막히고 오직 추출만 가능.

[시즌 '하이라이트' 인격(질문에 '하이라이트' 언급 시 필수 사용)]
- 시즌1:G사 일등대리 그레고르, 쥐어들자 싱클레어.
- 시즌2:개화 E.G.O::동백 이상.
- 시즌3:피쿼드호 선장 이스마엘.
- 시즌4:와일드헌트 히스클리프.
- 시즌5:라만차랜드 실장 돈키호테.
- 시즌6:홍원 군주 홍루.

[참고: 시즌별 대표 예시(문서 내용 기반)]
- 시즌1~4: 통상 인격/E.G.O는 정가·추출 가능.
- 시즌5: 정가·추출 불가(직전 시즌은 인격과  에고는 제외됨).
- 시즌6: 정가·추출 가능(정가는 출시 1주 후).
- 각 시즌·이벤트·발푸밤의 구체 목록/스킬/키워드는 문서(context) 안의 내용만 사용.

문서(context):
{context}

질문: {question}

답변:
"""
    )
)

#11월 9일 qa프로프트 적용 후 평가 시도 
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True
)

def ask_question(inputs: dict):
    # qa_chain 호출 → dict 반환
    output = qa_chain(inputs["question"])  # 또는 qa_chain.invoke(inputs["question"])
    # result만 반환
    return {"answer": output['result']}

dataset_name = "LIMBUS_RAG_QA"

qa_evaluator = LangChainStringEvaluator("qa")

def print_evaluator_prompt(evaluator):
    print(evaluator.evaluator.prompt.pretty_print())

print_evaluator_prompt(qa_evaluator)

experiment_results = evaluate(
    ask_question,
    data=dataset_name,
    evaluators=[qa_evaluator],
    experiment_prefix="RAG_EVAL",
    metadata={
        "variant": "QA Evaluator + PDF RAG",
    },
)


#----11월 1일 retriver 객체만 적용후  QA 평가 결과 요약 ----
# 정확: 6개 

# 부정확: 8개 

# 응답 거부: 1개 

# 따라서 사용자 만족도의 정확도보다 정확도가 더 낮게  측정됨 

# QA 템플릿 내용 수정 필요 

#---11월 4일 결과 동일---

#----11월 9일 QA 평가 결과 요약 ----

#부정확 10개 
#응답 거부 1r개 
#오히려 qa 프롬프트 수정 전보다 부정확이 늘어남

#qa 프롬프트의 길이 줄이고 재시도 

#부정확도가 13개로 측정됨으로서 정확도가 저하됨 

#이번에는 qa 프롴프트의 전제 조건을 제거후, 이전에 실패한 데이터셋의 설명만을 qa 템플릿에 추가

#정확도 8, 부정확도 5로 측정됨으로서 개선됨 

#따라서 현재 qa 템플릿과 문서개체의 개편 및 정보 갱신이 우선시되어야한다고 판단.