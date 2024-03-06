from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel


class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int


app = FastAPI()


# This allow sending of data (our TaggedItem) via POST to the API
@app.post("/items/")
async def create_item(item: TaggedItem):
    return item
