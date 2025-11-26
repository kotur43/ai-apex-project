from fastapi import FastAPI

app = FastAPI(title="AI + APEX Project")

@app.get("/")
def read_root():
    return {"message": "Hello, AI + APEX World!"}

# -------------------------
# GET with path parameters
# Example: /hello/Dusan
# -------------------------
@app.get("/hello/{name}")
def say_hello(name: str):
    return {"message": f"Hello, {name}!"}