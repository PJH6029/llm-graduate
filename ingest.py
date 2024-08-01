import os

from rag.api import upload_data, ingest_data, ingest_data_from_local
from rag.component.loader.PDFWithMetadataLoaderS3 import PDFWithMetadataLoaderS3
from rag.component.loader.UpstageLayoutLoader import UpstageLayoutLoader
from rag.component.ingestor.PineconeMultiVectorIngestor import PineconeMultiVectorIngestor

# bucket = os.environ["S3_BUCKET_NAME"]
ref_doc_root = os.path.join(os.path.dirname(__file__), "ref_docs")

def upload():
    for root, dirs, files in os.walk(ref_doc_root):
        for file in files:
            if not file.endswith(".pdf"):
                continue
            file_path = os.path.join(root, file)
            object_location = os.path.relpath(file_path, ref_doc_root)
            upload_data(file_path, object_location)


def ingest():
    cnt = 0
    PineconeMultiVectorIngestor.CHILD_INGESTION_CNT = 0
    # for file in files:
    #     s3_url = f"s3://{bucket}/frequently_access_documents/{file}"
    #     cnt += ingest_data(s3_url)
    for root, dirs, files in os.walk(ref_doc_root):
        for file in files:
            if not file.endswith(".pdf"):
                continue
            # if "다전공" not in file:
            #     continue
            file_path = os.path.join(root, file)
            cnt += ingest_data_from_local(file_path)
    
    print(f"{cnt} parent chunks ingested")
    print(f"{PineconeMultiVectorIngestor.CHILD_INGESTION_CNT} child chunks ingested")

if __name__ == "__main__":
    # upload()
    ingest()