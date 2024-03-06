from fastapi import FastAPI


app = FastAPI()


@app.get("/")
async def hello_world():
    return {"greeting": "hello world hein!"}


@app.get("/items/{item_id}")
async def get_itemns(item_id: int, count: int = 1):
    return {"fetch": f"Fetched {count} of {item_id}"}
