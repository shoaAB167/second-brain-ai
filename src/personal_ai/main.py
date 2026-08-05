from fastapi import FastAPI

app = FastAPI(
    title="Personal AI",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Personal AI is running 🚀"}
