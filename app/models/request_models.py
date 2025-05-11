from pydantic import BaseModel


class SummarizeRequest(BaseModel):
    text: str
    # Maybe add optional params like max and min length
    # max_length: int = 150
    # min_length: int = 15
