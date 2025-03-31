from typing import List
import requests
from requests import Response
from bs4 import BeautifulSoup
import os
import argparse
from argparse import Namespace
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document


def parse_args() -> Namespace:
    '''
    Parses the command line arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--download", help="download the wiki pages", action="store_true")
    parser.add_argument("--download_dir", type=str, help="path to the directory to save the downloaded data to", default="data/raw/")
    parser.add_argument("-p", "--process", help="run the processing step", action="store_true")
    parser.add_argument("--processed_dir", type=str, help="path to the directory to save the processed data to", default="data/processed/")
    parser.add_argument("--process_model", type=str, help="the ollama model to use for summarization during processing", default="qwen2.5:7b")
    parser.add_argument("-e", "--embed", help="embed the data and create a vector store", action="store_true")
    parser.add_argument("--vectorstore_dir", type=str, help="path to the directory to save the vector store to", default="vectorstore/")
    parser.add_argument("--embed_model", type=str, help="the ollama model to use for embedding the processed data into a vectorstore", default="nomic-embed-text:latest")
    return parser.parse_args()


def get_links_from(url: str) -> list[str]:
    '''
    Returns a list of all the links from a given URL.

    :param url: URL to get links from

    :return: List of links from the given URL
    '''
    response: Response = requests.get(url)
    soup: BeautifulSoup = BeautifulSoup(response.text, "html.parser")
    links: list[str] = []
    for link in soup.find_all('a', href=True):
        links.append(link['href'])
    return links


def get_links_from_wiki(wiki_base_url: str = "https://galaxylife.wiki.gg", all_pages_loc: str = "/wiki/Special:AllPages") -> list[str]:
    '''
    Retrieves all the links to relevant content pages from the wiki.
    
    :param wiki_base_url: Base URL of the wiki
    :param all_pages_loc: Subfolder of the "All Pages" page on the wiki. wiki_base_url + all_pages_loc should be the URL of the "All Pages" page.

    :return: List of links to relevant content pages on the wiki
    '''
    print(f"Retrieving links from {wiki_base_url + all_pages_loc}...")
    all_pages_url: str = wiki_base_url + all_pages_loc
    all_pages_links: list[str] = get_links_from(all_pages_url)
    wiki_links: list[str] = [link for link in all_pages_links if link.startswith("/wiki/")]
    link_blacklist: list[str] = ["/wiki/Special:CreateAccount?returnto=Special%3AAllPages",
                                 "/wiki/Special:UserLogin?returnto=Special%3AAllPages",
                                 "/wiki/Special:AllPages",
                                 "/wiki/Galaxy_Life_Wiki",
                                 "/wiki/Special:RecentChanges",
                                 "/wiki/Special:Random",
                                 "/wiki/Galaxy_Life_Wiki:Sandbox"]
    filtered_wiki_links: list[str] = [wiki_base_url + link for link in wiki_links if link not in link_blacklist]
    print(f"Found {len(filtered_wiki_links)} wiki links.")
    return filtered_wiki_links


def download_urls(urls: list[str], folder: str) -> None:
    '''
    Downloads the HTML content of all pages from a given list of URLs.

    :param urls: List of URLs of the pages to download
    :param folder: Folder to save the downloaded HTML content to
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    print("Downloading HTML content...")
    for ii, url in enumerate(urls):
        #print(f"{url=}")
        response: Response = requests.get(url)
        with open(os.path.join(folder, str(ii) + ".html"), "w", encoding="utf-8") as file:
            file.write(response.text)
        #print(f"finished {ii+1}/{len(urls)}")
        print("#", flush=True, end="")
    print("\nFinished downloading HTML content")
    

def process_data(model_name: str, raw_data: str, processed_data: str) -> None:
    '''
    Processes the raw HTML data into better human readable text files.
    Note that this will not make them perfect.

    :param raw_data: Folder containing the raw HTML data
    :param processed_data: Folder to save the processed data to
    '''
    if not os.path.exists(raw_data):
        print("No raw data found. Please download the data first and make sure to specify the correct path.")
        return
    
    if not os.path.exists(processed_data):
        os.makedirs(processed_data)

    llm = ChatOllama(model=model_name)
    raw_summarization_prompt = """CONTEXT: {context}

    TASK: Above is provided a raw text output from a website. Your job is to summarize the information in it about the game galaxy life.
    Stay true to the provided context. Don't leave out any important information. The summary should be detailed and structured in paragraphs.
    Do not talk about the Header/Footer information. Do not talk about anything else."""
    summarization_prompt = PromptTemplate.from_template(raw_summarization_prompt)


    print("Processing data...")
    for file in os.listdir(raw_data):
        ii: int = int(file.split(".")[0])
        with open(raw_data + file, "r", encoding="utf-8") as f:
            html_content: str = f.read()
            soup: BeautifulSoup = BeautifulSoup(html_content, "html.parser")
            content: str = soup.get_text() # type: ignore

            message = summarization_prompt.invoke({"context": content})
            response = llm.invoke(message)
            
            with open(os.path.join(processed_data, str(ii) + ".txt"), "w", encoding="utf-8") as text_file:
                text_file.write(str(response.content))
        print("#", flush=True, end="")
    print("\nFinished processing data")


def embed(model_name: str, data_dir: str, vectorstore_dir: str) -> None:
    '''
    Embeds the processed data and creates a vector store such that it 
    doesnt need to be re-embedded during runtime. Changes to the data
    will require re-embedding.
    
    :param data_dir: Folder containing the processed data
    '''
    # get raw docs
    doc_loader = DirectoryLoader(os.path.expanduser(data_dir), glob="**/*.txt", show_progress=True, use_multithreading=False, loader_cls=TextLoader)  
    raw_documents = doc_loader.load()

    # create embeddings
    embeddings: OllamaEmbeddings = OllamaEmbeddings(model=model_name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=500,length_function=len)
    documents: List[Document] = text_splitter.split_documents(raw_documents)

    # build vectorstore
    vector_store = FAISS.from_documents(documents[:1], embeddings)
    print(f"\t[PROCESS DOCS] ({len(documents)})")
    for doc in documents:
        _ = vector_store.add_documents([doc])
        print("#", end="", flush=True)
    print("")
    vector_store.save_local(vectorstore_dir)



if __name__ == "__main__":
    args: Namespace = parse_args()

    if args.download:
        links: list[str] = get_links_from_wiki()
        download_urls(urls=links, folder=args.download_dir)
    else:
        if os.path.exists(args.download_dir) and len(os.listdir(args.download_dir)) > 0:
            print(f"Found {len(os.listdir(args.download_dir))} raw data files. Skipping download of wiki data.")
        else:
            print("No wiki data found! Consider setting the '--download' flag!")

    if args.process:
        process_data(model_name=args.process_model, raw_data=args.download_dir, processed_data=args.processed_dir)
    else:
        if os.path.exists(args.processed_dir) and len(os.listdir(args.processed_dir)) > 0:
            print(f"Found {len(os.listdir(args.processed_dir))} processed data files. Skipping processing of raw data.")
        else:
            print("No processed data found! Consider setting the '--process' flag!")

    if args.embed:
        embed(model_name=args.embed_model, data_dir=args.processed_dir, vectorstore_dir=args.vectorstore_dir)
    else:
        if os.path.exists(args.vectorstore_dir) and len(os.listdir(args.vectorstore_dir)) > 0:
            print(f"Found vectorstore files. Skipping embedding of data and vector store creation.")
        else:
            print("No vector store found! Consider setting the '--embed' flag!")


    
    

