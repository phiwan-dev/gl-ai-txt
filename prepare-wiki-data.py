import requests
from requests import Response
from bs4 import BeautifulSoup
import os
import argparse
from argparse import Namespace
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


def parse_args() -> Namespace:
    '''
    Parses the command line arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--download", help="download the wiki pages", action="store_true")
    parser.add_argument("-p", "--preprocess", help="run the preprocessing step", action="store_true")
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


def download_urls(urls: list[str], folder: str = "data/raw/") -> None:
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
        with open(folder + str(ii) + ".html", "w", encoding="utf-8") as file:
            file.write(response.text)
        #print(f"finished {ii+1}/{len(urls)}")
        print("#", flush=True, end="")
    print("\nFinished downloading HTML content")
    

def preprocess_data(raw_data: str = "data/raw/", processed_data: str = "data/processed/", model_name: str="qwen2.5:7b") -> None:
    '''
    Preprocesses the raw HTML data into better human readable text files.
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


    print("Preprocessing data...")
    for file in os.listdir(raw_data):
        ii: int = int(file.split(".")[0])
        with open(raw_data + file, "r", encoding="utf-8") as f:
            html_content: str = f.read()
            soup: BeautifulSoup = BeautifulSoup(html_content, "html.parser")
            content: str = soup.get_text() # type: ignore

            message = summarization_prompt.invoke({"context": content})
            response = llm.invoke(message)
            
            with open(processed_data + str(ii) + ".txt", "w", encoding="utf-8") as text_file:
                text_file.write(str(response.content))
        print("#", flush=True, end="")
    print("\nFinished preprocessing data")


if __name__ == "__main__":
    args: Namespace = parse_args()

    if args.download:
        links: list[str] = get_links_from_wiki()
        download_urls(links)
    else:
        if os.path.exists("data/raw/") and len(os.listdir("data/raw/")) > 0:
            print(f"Found {len(os.listdir('data/raw'))} raw data files. Skipping download of wiki data.")
        else:
            print("No wiki data found! Consider setting the '--download' flag!")

    if args.preprocess:
        preprocess_data()
    else:
        if os.path.exists("data/processed/") and len(os.listdir("data/processed/")) > 0:
            print(f"Found {len(os.listdir('data/processed'))} processed data files. Skipping preprocessing of raw data.")
        else:
            print("No processed data found! Consider setting the '--preprocess' flag!")


    
    

