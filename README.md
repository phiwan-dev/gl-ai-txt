
# Introduction
This repository contains my source files for building a customized LLM RAG chatbot for the online game [Galaxy Life](https://galaxylifegame.net/) (2011-12-08). It operates on the information found on the [GalaxyLife Wiki](https://galaxylife.wiki.gg/). The project makes use of technology like [Ollama](https://github.com/ollama/ollama), [Langchain/Langgraph](https://github.com/langchain-ai/langchain) with the [FAISS](https://github.com/facebookresearch/faiss) vectorstore and [Open-Webui](https://github.com/open-webui/open-webui) as an optional frontend.


# Quickstart
There are 2 ways to run this model: through a basic cli interface or a open-webui pipeline for a pretty browser frontend. If you want to use the open-webui interface you need to clone this repository recursivly with its submodules: `git clone https://github.com/phiwan-dev/gl-ai-txt.git --recurse-submodules`. I assume a GNU/Linux system with an Nvidia GPU with at least 6 GB of VRAM to be given. The Python uv package manger is used to manage the dependencies.

### Dependencies
The `uv.lock`, `pyproject.toml` and `.python-version` files are used by the uv package manager to manage the pinned dependencies automatically. You can optionally run `uv sync` to install the packages into `.venv` manually.  
For the NixOS users: The repo contains a nix flake to set the Nvidia CUDA environment variables. You can set up the environment through `nix develop`. If nix-direnv is configured on the system, you can run `direnv allow` to automatically load and unload the nix environment when entering and leaving the project directory.
> [!NOTE]
> The project and its submodule pipelines have been modified to make use of the uv package mangager. However, because that is officially not supported we have to have our dependencies for our custom pipelines already installed. Trying to have pipelines manage the dependencies dynamically will result in an error so make sure to comment out any requirements in the header of the pipelines you plan on using with this pipelines instance. Feel free to set up a secondary, unmodified pipelines instance for all your other pipeline needs.  
  
> [!NOTE]
> Note for the future if the dependencies should be updated: Setting up the uv environment can be quite painful due to a lot of fought for common dependencies. What I found to work is to first add all the pipeline dependencies through `uv add -r pipelines/requirements.txt` and then adding the dependencies for this bot itself. They can be found in the header of `bot.py`. Especially pydantic is heavily fought for and on coflict it helps to remove it and add it at the end to let uv figure its versioning out.

The flake environment provides ollama v0.6.0 which is able to run models like qwen2.5 or gemma3. If you do not make use of the flake and do not have it installed and up running (`ollama serve`) then it's time to do so now. Let's download `qwen2.5:7b`, which is the default model used at the core of the bot and the `nomic-embed-text:latest` for dealing with embedded text documents from the wiki later on. For this, you can use
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
In order to have a pretty frontend framework, the bot can also work as a Open-WebUI pipeline. Let's set up Open-Webui. While there are many ways to install it, we can make use of the uv package manger for that:
old version had problems
```
uv pip install open-webui==0.6.5
uv run open-webui serve
```
Open-WebUI should now be running at `http://localhost:8080`.  
> [!NOTE]
> While newer versions fo open webui will likely work too, I have only tested it with its latest release as of today which is 0.6.5. There have been broken versions in the past like 0.5.21.  
  
> [!WARNING]
> Note that pipelines offers arbitrary code execution! If you don't feel comfortable with this you should stop here. Never run pipelines from people you don't trust!
> Pipelines also offers the possibility to be installed in a Docker container to minimize risk. (Not explained here; please refer to its documentation for further information)
In a new window/shell session we can now run pipelines:
```
cd [gl-ai-txt directory]/pipelines/
sh ./start.sh
```
After this is done we have to configure the connection settings in Open-Webui. Go to `Admin Panel > Settings > Connections`. Under `OpenAI API Connections` put in url=`http://localhost:9099` with key=`0p3n-w3bu!`. At this point you can also double check that the ollama connection works properly from within the interface.  
Lastly we need to upload the actual pipeline. Go to `Admin Panel > Settings > Pipelines` and upload the `bot.py` file from this repository. You can leave all parameters/valves at their default value. You can now reload the page (F5) and enjoy chatting with the bot! Make sure the correct pipeline/model is selected in the top left corner.  
Have fun!  
> [!NOTE]
> The bot works best when using the english language. This is largely limited by the ollama model you use. I tried prompting it in German and while it is definitely able to pick up some informatation provided by the wiki, it does struggle to form proper German sentences. Your mileage may vary.

