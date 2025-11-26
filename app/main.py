from fastapi import FastAPI

app = FastAPI(title="AI + APEX Project")

@app.get("/")
def read_root():
    return {"message": "Hello, AI + APEX World!"}
