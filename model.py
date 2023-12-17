from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "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"

loader = CSVLoader(file_path = 'data.csv')

index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])

chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key= "question")


query = input("Enter your query : ")
response = chain({"question" : query})
print(response['result'])
    