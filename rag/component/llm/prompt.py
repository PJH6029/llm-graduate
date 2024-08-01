from langchain_core.prompts import ChatPromptTemplate

translation_prompt_template = """
You are an assistant for Korean-English translation tasks.

I will give you the sentence.
If the sentence is already written in English, just copy the sentence.
If not, please translate the sentence from Korean to English.

You should say only the translation of the sentence, and do not say any additional information.

<sentence>
{sentence}
</sentence>

Translation:
"""
translation_prompt = ChatPromptTemplate.from_template(translation_prompt_template)

rewrite_prompt_template = """
You are an assistant for question-revision tasks.
Using given chat history, rephrase the following question to be a standalone question.
The standalone question must have main words of the original question.

Write the revised question in {lang}.

<chat-history>
{history}
</chat-history>

<question>
{query}
</question>

Revised question:
"""
rewrite_prompt = ChatPromptTemplate.from_template(rewrite_prompt_template).partial(lang="Korean")

expansion_prompt_template = """
Your task is to expand the given query, considering the chat history.
Generate {n} queries that are related to the given query and chat history.

You should provide the queries in {lang}.

All the queries should be separated by a newline.
Do not include any additional information. Only provide the queries.

<chat-history>
{history}
</chat-history>

<question>
{query}
</question>

Queries:
"""
expansion_prompt = ChatPromptTemplate.from_template(expansion_prompt_template).partial(n=3, lang="Korean")

# restrice the number of sentences to 3, to improve response latency
hyde_prompt_template = """
You are an assistant for question-answering tasks.
Please write a passage to answer the question, considering the given chat history.
Even though you cannot find the context in the chat history, you should generate a passage to answer the question.
Write the answer in {lang}.

Use up to {n} sentences to answer the question.

<chat-history>
{history}
</chat-history>

<question>
{query}
</question>

Answer:
"""
hyde_prompt = ChatPromptTemplate.from_template(hyde_prompt_template).partial(n=3, lang="Korean")

generation_with_hierarchy_prompt_template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question considering the chat history.

I have provided you with the base context and additional context.
Base context is the main context that you should consider first.
It is the most base information that contains the answer.
Additional context contains information about changes or updates to the base context, 
which are made according to the version updates of the documents or by the request of the specific clients.

You should think step by step.
You should first find the answer from the base context,
and then consider the additional context to see if there are any updates or changes to the answer.
If there are several changes, you should consider the additional context of the most recent version to provide the most up-to-date answer, also comparing the changes to the base context.

If you don't know the answer, just say that you don't know.

You can answer in descriptive form or paraphrased form if you want, and keep the answer concise.

You should answer with the format of the example answer below.
When you reference the documents, you should provide the exact title of the document.
Feel free to use markdown to format your answer.

--------------------------------------------------
**** Example 1 ****
<chat-history>
Human: Can you describe the features of UTIL-1?
</chat-history>

<context>
    <base-context>
    --- Document: Datacenter NVMe SSD Specification v2.0r21.pdf ---
    Average Score: 0.99
    DOC META:
    {{'doc_id': 's3://llm-project-demo-bucket/frequently_access_documents/OCP/2.0/Datacenter+NVMe+SSD+Specification+v2.0r21.pdf', 'doc_name': 'Datacenter NVMe SSD Specification v2.0r21.pdf', 'category': 'base', 'version': 'v2.0r21', 'uri': 'https://llm-project-demo-bucket.s3.ap-northeast-1.amazonaws.com/frequently_access_documents/OCP/2.0/Datacenter+NVMe+SSD+Specification+v2.0r21.pdf'}}
    
    --- Chunk: dbc8ea71-dd6e-45df-a78c-0bec284d170c-9869493d-2d77-484c-82c3-c9b4befcef55 ---
    Score: 0.99
    TEXT:
    16.1 NVMe CLI Management Utility
    The NVMeCLI utility (https://github.com/linux-nvme/nvme-cli) shall be used as one of the management utilities for NVMe devices.
    Requirement ID: UTIL-1
    Description: The SSD supplier must test their SSDs with this utility and ensure compatibility. The following is the minimum list of commands that need to be tested with NVMeCLI:
    - Format.
    - Secure erase.
    - FW update.
    - Controller reset to load FW.
    - Health status.
    - Log page reads including vendor log pages.
    - SMART status.
    CHUNK META:
    {{'chunk_id': 'dbc8ea71-dd6e-45df-a78c-0bec284d170c-9869493d-2d77-484c-82c3-c9b4befcef55', 'page': 110, 'score': 'HIGH'}}
    

    --- Document: datacenter-nvme-ssd-specification-v2-5-pdf.pdf ---
    Average Score: 0.97
    DOC META:
    {{'doc_id': 's3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf', 'doc_name': 'datacenter-nvme-ssd-specification-v2-5-pdf.pdf', 'category': 'base', 'version': 'v2.5', 'uri': 'https://llm-project-demo-bucket.s3.ap-northeast-1.amazonaws.com/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf'}}
    
    --- Chunk: 7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b ---
    Score: 0.97
    TEXT:
    18.1 NVMe CLI Management Utility
    The NVMeCLI utility (https://github.com/linux-nvme/nvme-cli/tree/master/plugins/ocp) shall be used as one of the management utilities for NVMe devices.
    Requirement ID: UTIL-1
    Description: The SSD supplier must test their SSDs with this utility and ensure compatibility. The following is the minimum list of commands that need to be tested with NVMeCLI:
    - Format.
    - Secure erase.
    - FW update.
    - Controller reset to load FW.
    - Health status.
    - Log page reads including vendor log pages.
    - SMART status.
    CHUNK META:
    {{'chunk_id': '7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b', 'page': 170, 'score': 'HIGH'}}

    </base-context>
    <additional-context>
    --- Document: Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf ---
    Based on: s3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf
    Average Score: 0.89
    DOC META:
    {{'doc_id': 's3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf', 'doc_name': 'Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf', 'category': 'additional', 'version': 'v2.5-addendum-v0.20', 'uri': 'https://llm-project-demo-bucket.s3.ap-northeast-1.amazonaws.com/frequently_access_documents/OCP/2.5/Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf', 'base_doc_id': 's3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf'}}
    
    --- Chunk: 7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b ---
    Score: 0.89
    TEXT:
    2. Change List
    2.1. Utilization Features
    Requirement ID: UTIL-1
    Description: The list of commands to be tested has been updated.
    - Format.
    - Secure erase.
    - FW update.
    - Controller reset to load FW.
    - Health status.
    - SMART status.
    CHUNK META:
    {{'chunk_id': '7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b', 'page': 3, 'score': 'HIGH'}}
    </additional-context>
</context>

<question>
Can you describe the features of UTIL-1?
</question>

Answer:
### Answer
I can find the requirements for UTIL-1 in the document "Datacenter NVMe SSD Specification v2.0r21.pdf" and "datacenter-nvme-ssd-specification-v2-5-pdf.pdf". 
UTIL-1 describes that the ssd supplier must test their SSDs with the following utility and ensure compatibility. The followings are the minimum list of commands that should be tested with NVMeCLI.

- Format
- Secure erase
- FW update
- Controller reset to load FW
- Health status
- SMART status.
If you want to know more details, you can refer to the document "datacenter-nvme-ssd-specification-v2-5-pdf.pdf" page 170, which is the latest version of the document.

### Changes
The requirements for UTIL-1 is consistent in the documents between version 2.0r21 and 2.5.
The only difference is the section number, which is 16.1 in the document "Datacenter NVMe SSD Specification v2.0r21.pdf" and 18.1 in the document "datacenter-nvme-ssd-specification-v2-5-pdf.pdf".

However, the document "Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf" provides an updated list of commands to be tested with NVMeCLI.
While the list of commands in the base document(datacenter-nvme-ssd-specification-v2-5-pdf.pdf) contains the command **Log page reads including vendor log pages**, it is removed in the updated list of commands in the addendum document.

### References
#### Base Documents
- Datacenter NVMe SSD Specification v2.0r21.pdf, page 110
- datacenter-nvme-ssd-specification-v2-5-pdf.pdf, page 170

#### Additional Documents
- Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20, page 3

--------------------------------------------------

<chat-history>
{history}
</chat-history>

<context>
{context}
</context>

<question>
{query}
</question>

Answer:
"""
generation_with_hierarchy_prompt = ChatPromptTemplate.from_template(generation_with_hierarchy_prompt_template)

generation_without_hierarchy_prompt_template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question considering the chat history.

If you don't know the answer, just say that you don't know.

You can answer in descriptive form or paraphrased form if you want, and keep the answer concise.

You should answer with the format of the example answer below.
Feel free to use markdown to format your answer.

You should rite answer in {lang}.

--------------------------------------------------
**** Example 1 ****
<chat-history>
Human: 주전공생 2024학번 졸업규정에 대해 알려줘
AI: 2024학번 졸업 규정에 대해 알려드리겠습니다. 주전공생이 단일 전공인 경우, 전공학점을 63학점 이수해야 하며, 전필 30학점과 전선 내규필수 8학점이 포함됩니다. 들어야 하는 전공필수 과목은 다음과 같습니다 ...(omitted)...
</chat-history>

<context>
--- 주전공(단일전공).pdf ---
Average Score: 0.99
DOC META:
{{'doc_id': '주전공(단일전공).pdf', 'doc_name': '주전공(단일전공).pdf' }}

--- Chunk: dbc8ea71-dd6e-45df-a78c-0bec284d170c-9869493d-2d77-484c-82c3-c9b4befcef55 ---
Score: 0.99
TEXT:
 ※ 컴퓨터공학부 소속 학생들의 졸업기준을 '입학년도' 이후 기준 중 학생이 선택하여 졸업기준을 정할 수 있다.


| 2025학번 | |
| --- | --- |
| 이수학점 | 전공학점 63학점 이수(전필 27학점 \+ 전선 내규필수 5학점을 포함한 63학점 이수) |
| 전필 | 이산수학(3\), 논리설계(3\), 컴퓨터프로그래밍(3\), 기계학습 개론(3\), 자료구조(3\), 컴퓨터구조(3\), 시스템프로그 래밍(3\), 알고리즘(3\), 공대 공통교과목(3\) |
| 전선내규필수 | 소프트웨어 개발의 원리와 실습(4\), 컴퓨터공학세미나(1\) 또는 컴퓨팅 살펴보기(1\) 중 1과목 선택 |
| 2021\~2024학번 | |
| 이수학점 | 전공학점 63학점 이수(전필 30학점 \+ 전선 내규필수 8학점을 포함한 63학점 이수) |
| 전필 | 이산수학(3\), 논리설계(4\), 컴퓨터프로그래밍(4\), 전기전자회로(3\), 자료구조(3\), 컴퓨터구조(3\), 시스템프로그래밍(4\), 알고리즘(3\), 공대 공통교과목(3\) |
| 전선내규필수 | 소프트웨어 개발의 원리와 실습(4\), 컴퓨터공학세미나(1\) 또는 IT\-리더십세미나(1\)(세미나는 1과목만 이수), 창의적통합설계 1(3\) 또는 창의적통합설계 2(3\) |
| 2020학번 | |
| 이수학점 | 전공학점 63학점 이수(전필 31학점 \+ 전선 내규필수 8학점을 포함한 63학점 이수) |
| 전필 | 이산수학(3\), 논리설계(4\), 컴퓨터프로그래밍(4\), 전기전자회로(3\), 자료구조(4\), 컴퓨터구조(3\), 시스템프로그래밍(4\), 알고리즘(3\), 공대 공통교과목(3\) |
| 전선내규필수 | 소프트웨어 개발의 원리와 실습(4\), 컴퓨터공학세미나(1\) 또는 IT\-리더십세미나(1\)(세미나는 1과목만 이수), 창의적통합설계 1(3\) 또는 창의적통합설계 2(3\) |
| 2019학번 | |
| 이수학점 | 전공학점 63학점 이수(전필 35학점 \+ 전선 내규필수 4학점을 포함한 63학점 이수) |
| 전필 | 이산수학(3\) 논리설계(4\), 컴퓨터프로그래밍(4\), 전기전자회로(3\), 자료구조(4\), 컴퓨터구조(3\), 소프트웨어 개발의 원리와 실습(4\), 시스템프로그래밍(4\), 알고리즘(3\), 공대 공통교과목(3\) |
| 전선내규필수 | 컴퓨터공학세미나(1\) 또는 IT\-리더십세미나(1\) (세미나는 1과목만 이수), 창의적통합설계1 (3\) 또는 창의적통합설계2 (3\) |
| 2015\~2018학번 | |
| 이수학점 | 전공학점 63학점 이수(전필 37학점 \+ 전선 내규 4학점을 포함한 63학점 이수) |
| 전필 | 이산수학, 논리설계(4\), 컴퓨터프로그래밍(4\), 전기전자회로, 자료구조(4\), 컴퓨터구조, 소프트웨어 개발의 원리와 실습, 시스템프로그래밍(4\), 하드웨어시스템설계, 알고리즘, 공대 공통교과목 |
| 전선내규필수 | 컴퓨터공학세미나(1\) 또는 IT\-리더십세미나(1\)(세미나는 1과목만 이수), 창의적통합설계 1(3\) 또는 창의적통합설계 2(3\) |
CHUNK META:
{{'chunk_id': 'dbc8ea71-dd6e-45df-a78c-0bec284d170c-9869493d-2d77-484c-82c3-c9b4befcef55', 'page': 2, 'score': '0.99'}}


--- Document: 주전공(다전공 병행).pdf ---
Average Score: 0.97
DOC META:
{{'doc_id': '주전공(다전공 병행).pdf', 'doc_name': '주전공(다전공 병행).pdf' }}

--- Chunk: 7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b ---
Score: 0.97
TEXT:
전공(다전공 병행)
===========

 ※ 컴퓨터공학부 소속 학생들의 졸업기준을 '입학년도' 이후 기준 중 학생이 선택하여 졸업기준을 정할 수 있다.

 

| 2025학번 |  |
| --- | --- |
| 이수학점 | 컴퓨터공학부의 전공학점을 45학점 이상(복수, 연합) : 컴퓨터공학부 전필 27학점 \+ 전선 내규필수 5학점 \+ 전선 13학점 이상 컴퓨터공학부의 전공학점을 48학점 이상(부,연계) : 컴퓨터공학부 전필 27학점 \+ 전선 내규필수 5학점 \+ 전선 16학점 이상 |
| 전필 | 이산수학(3\), 논리설계(3\), 컴퓨터프로그래밍(3\), 기계학습 개론(3\), 자료구조(3\), 컴퓨터구조(3\), 시스템프로그래밍(3\), 알고리즘(3\), 공대 공통교과목(3\) |
| 전선내규필수 | 소프트웨어 개발의 원리와 실습(4\), 컴퓨터공학세미나(1\) 또는 컴퓨팅 살펴보기(1\) 중 1과목 선택 |

 

| 2021\~2024학번 |  |
| --- | --- |
| 이수학점 | 컴퓨터공학부의 전공학점을 45학점 이상(복수, 연합) : 컴퓨터공학부 전필 30학점 \+ 전선 내규필수 8학점 \+ 전선 7학점 이상 컴퓨터공학부의 전공학점을 48학점 이상(부,연계) : 컴퓨터공학부 전필 30학점 \+ 전선 내규필수 8학점 \+ 전선 10학점 이상 |
| 전필 | 이산수학(3\), 논리설계(4\), 컴퓨터프로그래밍(4\), 전기전자회로(3\), 자료구조(3\), 컴퓨터구조(3\), 시스템프로그래밍(4\), 알고리즘(3\), 공대 공통교과목(3\) |
| 전선내규필수 | 소프트웨어 개발의 원리와 실습(4\), 컴퓨터공학세미나(1\) 또는 IT\-리더십세미나(1\)(세미나는 1과목만 이수), 창의적통합설계 1(3\) 또는 창의적통합설계2(3\) |

 

| 2020학번 |  |
| --- | --- |
| 이수학점 | 컴퓨터공학부의 전공학점을 45학점 이상(복수, 연합) : 컴퓨터공학부 전필 31학점 \+ 전선 내규필수 8학점 \+ 전선 6학점 이상 컴퓨터공학부의 전공학점을 48학점 이상(부,연계) : 컴퓨터공학부 전필 31학점 \+ 전선 내규필수 8학점 \+ 전선 9학점 이상 |
| 전필 | 이산수학(3\), 논리설계(4\), 컴퓨터프로그래밍(4\), 전기전자회로(3\), 자료구조(4\), 컴퓨터구조(3\), 시스템프로그래밍(4\), 알고리즘(3\), 공대 공통교과목(3\) |
| 전선내규필수 | 소프트웨어 개발의 원리와 실습(4\), 컴퓨터공학세미나(1\) 또는 IT\-리더십세미나(1\)(세미나는 1과목만 이수), 창의적통합설계 1(3\) 또는 창의적통합설계 2(3\) |

 

| 2019학번 | |
| --- | --- |
| 이수학점 | 컴퓨터공학부의 전공학점을 41학점 이상 이수 : 컴퓨터공학부 전필 35학점 \+ 전선 내규필수 4학점 \+ 전선 2학점이상 |
| 전필 | 이산수학, 논리설계(4학점), 컴퓨터프로그래밍(4학점), 전기전자회로, 자료구조(4학점), 컴퓨터구조, 소프트웨어 개발의 원리와 실습(4학점), 시스템프로그래밍(4학점), 알고리즘, 공대 공통교과목 |
| 전선내규필수 | 컴퓨터공학세미나 또는 IT\-리더십세미나(세미나는 1과목만 이수), 창의적통합설계 1 또는 창의적통합설계 2 |

 졸업규정

   
주전공(다전공 병행)
 
CHUNK META:
{{'chunk_id': '7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b', 'page': 1, 'score': '0.97'}}

</context>

<question>
2025학번도 알려줘.
</question>

Answer:
### Answer
2025학번 주전공생 졸업 규정에 대해 설명드리겠습니다.

#### 단일전공
단일전공인 2025학번 주전공생은 전공필수 27학점, 전선 내규필수 5학점을 포함한 63학점 이상을 이수해야 합니다. 

전공 필수 과목에는 다음이 포함됩니다.

- 이산수학(3)
- 논리설계(3)
- 컴퓨터프로그래밍(3)
- 기계학습 개론(3)
- 자료구조(3)
- 컴퓨터구조(3)
- 시스템프로그래밍(3)
- 알고리즘(3)
- 공대 공통교과목(3)

전선 내규 필수 과목에는 다음이 포함됩니다.

- 소프트웨어 개발의 원리와 실습(4)
- 컴퓨터공학세미나(1) 또는 컴퓨팅 살펴보기(1) 중 1과목 선택

#### 다전공 병행
다른 전공을 병행하는 주전공생의 경우에는 졸업 규정이 단일전공과 다르니 주의해야 합니다. 복수전공이나 연합전공을 병행하는 학생은 컴퓨터공학부의 전공필수 27학점, 전선 내규필수 5학점을 포함한 45학점 이상을 이수해야합니다. 즉, 전선 13학점 이상을 추가로 이수해야 합니다.
부전공이나 연계 전공의 경우, 컴퓨터공학부의 전공필수 27학점, 전선 내규필수 5학점을 포함한 48학점 이상을 이수해야합니다. 즉, 전선 16학점 이상을 추가로 이수해야 합니다.

다전공 병행 주전공생이 수강해야하는 전공 필수 과목과 전선 내규 필수 과목은 단일전공생과 동일합니다.

### References
- 주전공(단일전공).pdf, page 1
- 주전공(다전공 병행).pdf, page 2

--------------------------------------------------

<chat-history>
{history}
</chat-history>

<context>
{context}
</context>

<question>
{query}
</question>

Answer:
"""
generation_without_hierarchy_prompt = ChatPromptTemplate.from_template(generation_without_hierarchy_prompt_template).partial(lang="Korean")

verification_prompt_template = """
Given context, verify the fact in the response. If the response is correct, say "Yes". If not, say "No".
Context: {context}
Answer: {response}

Verification:
"""
verification_prompt = ChatPromptTemplate.from_template(verification_prompt_template)

classfiy_target_prompt_template = """
For the given queries, classify the targets of the question.
You should select the targest from the available options.

You can select up to {n} targets, but you should select at least one target.

Selected targets should be separated by a newline, and each line only should contain one target.

Do not provide any additional information. Only provide the targets.

-----------------------------
Example 1:

Queries:
['컴퓨터공학부를 복수전공하는 2019학번 학생의 전필 교과목', '복수전공생과 부전공생의 졸업요건을 비교해줘"]

Available options:
["복수전공", "부전공", "주전공(다전공)", "주전공(단일전공)"]

Targets:
복수전공
부전공

-----------------------------
Example 2:

Queries:
["20학번 주전공생의 졸업 요건이 뭐야?"]

Available options:
["복수전공", "부전공", "주전공(다전공)", "주전공(단일전공)"]

Targets:
주전공(다전공)
주전공(단일전공)
-----------------------------
Example 3:

Queries:
["20학번 주전공생과 부전공생의 졸업 요건이 뭐야?", "20학번 주전공생의 졸업요건과 부전공생의 졸업요건을 비교해줘"]

Available options:
["복수전공", "부전공", "주전공(다전공)", "주전공(단일전공)"]

Targets:
주전공(다전공)
주전공(단일전공)
부전공
-----------------------------

Queries:
{queries}

Available options:
{options}

Targets:
"""
classfiy_target_prompt = ChatPromptTemplate.from_template(classfiy_target_prompt_template).partial(n=2)