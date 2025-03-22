
from typing import List, Union, Generator, Iterator


class Pipeline:

    def __init__(self) -> None:
        self.name = "GL Bot"

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        # If you'd like to check for title generation, you can add the following check
        if body.get("title", False):
            print("Title Generation Request")

        print(f"{messages=}")
        print("---")
        print(f"{user_message=}")
        print("---")
        print(f"{body=}")
        print("---")

        return f"response"

