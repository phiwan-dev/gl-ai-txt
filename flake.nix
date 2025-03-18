{
    description = "Python 3.12 development environment";

    inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    outputs = { self, nixpkgs }:
    let
        system = "x86_64-linux";
        pkgs = import nixpkgs { inherit system; config.cudaSupport = true; config.allowUnfree = true; };
    in {
        devShells.${system}.default = pkgs.mkShell {
            buildInputs = with pkgs; [
                ollama

                python312
                (with python312Packages; [
                    langchain
                    langchain-core
                    langchain-community
                    langchain-ollama
                    langchain-text-splitters
                    langgraph
                    faiss
                    numpy
                    beautifulsoup4
                    tqdm
                ])
            ];
        };
    };
}

