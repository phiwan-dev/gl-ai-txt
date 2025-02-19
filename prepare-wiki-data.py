import requests
from requests import Response
from bs4 import BeautifulSoup
import os


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
    
    for ii, url in enumerate(urls):
        print(f"{url=}")
        response: Response = requests.get(url)
        with open(folder + str(ii) + ".html", "w", encoding="utf-8") as file:
            print(response.text)
            file.write(response.text)
        print(f"finished {ii+1}/{len(urls)}")
    print("Finished downloading HTML content")
    


if __name__ == "__main__":
    links: list[str] = get_links_from_wiki()
    download_urls(links)
