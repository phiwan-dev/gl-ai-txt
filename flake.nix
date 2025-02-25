{
    description = "Python 3.12 development environment";

    outputs = { self, nixpkgs }:
    let
        system = "x86_64-linux";
        pkgs = import nixpkgs { inherit system; allowUnfree = true; };
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
                    faiss
                    numpy
                    beautifulsoup4
                ])
            ];
        };
    };
}

