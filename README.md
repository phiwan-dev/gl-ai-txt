# Introduction
This repository contains my source files for building a customized LLM RAG chatbot for the online game [Galaxy Life](https://galaxylifegame.net/) (2011-12-08). It operates on the information found on the [GalaxyLife Wiki](https://galaxylife.wiki.gg/). The project makes use of technology like [Ollama](https://github.com/ollama/ollama), [Langchain/Langgraph](https://github.com/langchain-ai/langchain) with the [FAISS](https://github.com/facebookresearch/faiss) vectorstore and [Open-Webui](https://github.com/open-webui/open-webui) as an optional frontend.


# Quickstart
There are 2 ways to run this model: through a basic cli interface or a open-webui pipeline for a pretty browser frontend. I assume a GNU/Linux system (NixOS) with an Nvidia GPU with at least 6 GB of VRAM to be given.

### Dependencies
The repo contains a nix flake to get you strated using the nix package manager. You can set up the environment through `nix develop`. If nix-direnv is configured on the system, you can run `direnv allow` to automatically load and unload the nix environment when entering and leaving the project directory.
> [!NOTE]
> Other methods for setting up python environments are not supported at the moment and need to be prepared manually. Have a look at the [flake.nix](flake.nix) file for dependency hints.  

The flake environment provides ollama v0.6.0 which is able to run models like qwen2.5 or gemma3. Let's download `qwen2.5:7b`, which is the default model used at the core of the bot and the `nomic-embed-text:latest` for dealing with embedded text documents from the wiki later on. For this, you can use
```
ollama pull qwen2.5:7b
ollama pull nomic-embed-text:latest
```
I recommend a model with the intelligence of at least qwen2.5:7b, which makes this a decently heavy project to run. You can always get by by running the models on the CPU but that is rather slow. Lastly, run the ollama service with `ollama serve`.

### Wiki-Data
In order to execute our RAG pipeline, we need to gather some data first. Preparing the data is a 3 step process. First we scrape the raw .html from the GalaxyLife Wiki, then we process it into human readable text files and finally we embedd said files into a vector store. The vector store is saved to disk such that we don't have to compute it at runtime. The script [prepare-wiki-data.py](prepare-wiki-data.py) provides a conveniet way to do all 3 steps. Please remember that scraping every site on the wiki puts additional burdon on their servers! Please use this responsibly and only re-download when necessary. Use the following command to do all 3 steps at once:
```
python prepare-wiki-data.py --download --process --embed
```
> [!TIP]
> For a more complete overview about the available arguments you can run `python prepare-wiki-data.py --help`.

### CLI usage
We are now ready to run the bot using the CLI using the command `python bot.py`. By default the bot will automatically answer a default question upon startup. The default question is "What NPC are there?". You can now chat with the bot about anything GalaxyLife related!

### Open-WebUI usage
In order to have a pretty frontend framework, the bot can also work as a Open-WebUI pipeline. Let's set up Open-Webui. While there are many ways to install it, I personally prefer to install it locally in a conda environment. Make sure you have a conda ready. On NixOS you can use:
```
nix-shell -p conda
conda-shell
conda env list
```
As of today, the latest version of open-webui currently has a bug when it comes to pipelines. Hence, we install the slightly older open-webui version 0.5.19. This is likely to get resolved in the future. Now create the environment and run it:
```
conda create --name open-webui python==3.12 open-webui==0.5.19
conda activate open-webui
open-webui serve
```
Open-WebUI should now be running at `http://localhost:8080`.  
Now let's install pipelines for which we again need conda, which I assume to be available by now.
> [!NOTE]
> Note that pipelines offers arbitrary code execution! If you don't feel comfortable with this you should stop here. Never run pipelines from people you don't trust!
> Pipelines also offers the possibility to be installed in a Docker container to minimize risk.
```
conda create --name pipelines python==3.12
conda activate pipelines
git clone https://github.com/open-webui/pipelines.git
cd pipelines
pip install -r requirements.txt
sh ./start.sh
```
After this is done we have to configure the connection settings in Open-Webui. Go to `Admin Panel > Settings > Connections`. Under `OpenAI API Connections` put in url=`http://localhost:9099` with key=`0p3n-w3bu!`. At this point you can also double check that the ollama connection works properly.  
Lastly we need to upload the actual pipeline. Go to `Admin Panel > Settings > Pipelines` and upload the `bot.py` file from this repository. You can leave all parameters/valves at their default value. You can now reload the page (F5) and enjoy chatting with the bot! Make sure the correct pipeline/model is selected in the top left corner.  
Have fun!