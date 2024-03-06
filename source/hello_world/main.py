from fastapi import FastAPI

app = FastAPI()


# Define a GET on the specified endpoint
@app.get("/")
async def hello_world():
    return {"greeting": "hello world hein!"}
