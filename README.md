# Introduction
This repository contains my source files for building a customized LLM chatbot for the online game [Galaxy Life](https://galaxylifegame.net/) (2011-12-08).

# Status
In progress...

# Quickstart
Before we start, we need to set up our environment. You can do this using nix flakes through `nix develop`. Other methods for setting up python environments are not supported at the moment and need to be prepared manually. Have a look at the [flake.nix](flake.nix) file for dependency hints.  

There are 2 important files: [prepare-wiki-data.py](prepare-wiki-data.py) and [run.py](run.py).  
Before we run the LLM, we need to set up the data for it to work on. This is done through `python prepare-wiki-data.py --download --preprocess` which downloads the raw .html articles from the Galaxy life wiki and then preprocesses them into a more suitable form.  
Once we are set up, we can run the LLM through `python run.py`. Have fun!
