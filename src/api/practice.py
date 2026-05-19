from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

app = FastAPI(title="Learning API", version="0.1.0")


@app.get("/calc/add")
async def add(
    a: float = Query(..., description="First number"),
    b: float = Query(..., description="Second number"),
) -> dict[str, float]:
    """Add two numbers."""
    return {"result": a + b}


@app.get("/calc/subtract")
async def subtract(
    a: float = Query(..., description="First number"),
    b: float = Query(..., description="Second number"),
) -> dict[str, float]:
    """Subtract two numbers."""
    return {"result": a - b}


@app.get("/calc/multiply")
async def multiply(
    a: float = Query(..., description="First number"),
    b: float = Query(..., description="Second number"),
) -> dict[str, float]:
    """Multiply two numbers."""
    return {"result": a * b}


@app.get("/calc/divide")
async def divide(
    a: float = Query(..., description="First number"),
    b: float = Query(..., description="Second number"),
) -> dict[str, float]:
    """Divide two numbers."""
    if b == 0:
        raise HTTPException(status_code=400, detail="Second number cannot be zero")
    return {"result": a / b}


class Operation(BaseModel):
    op: Literal["add", "subtract", "multiply", "divide"]
    a: float
    b: float


class BatchRequest(BaseModel):
    operations: list[Operation] = Field(min_length=1, max_length=100)


class BatchResponse(BaseModel):
    results: list[float | None]  # None for failed ops (e.g., divide by zero)


@app.post("/calc/batch", response_model=BatchResponse)
async def batch_calculate(payload: BatchRequest) -> BatchResponse:
    """Perform a batch of operations."""
    results: list[float | None] = []
    for op in payload.operations:
        try:
            if op.op == "add":
                results.append(op.a + op.b)
            elif op.op == "subtract":
                results.append(op.a - op.b)
            elif op.op == "multiply":
                results.append(op.a * op.b)
            elif op.op == "divide":
                if op.b == 0:
                    results.append(None)  # Indicate error for this operation
                else:
                    results.append(op.a / op.b)
        except Exception:
            results.append(None)  # Catch-all for unexpected errors
    return BatchResponse(results=results)


_NOTES: dict[int, dict[str, Any]] = {
    1: {"id": 1, "title": "First Note", "content": "This is the first note."},
    2: {"id": 2, "title": "Second Note", "content": "This is the second note."},
    3: {"id": 3, "title": "Third Note", "content": "This is the third note."},
}


def get_notes_db() -> dict[int, dict[str, Any]]:
    """Simulate a notes database."""
    return _NOTES


class NoteCreate(BaseModel):
    title: str = Field(
        min_length=1,
        max_length=200,
    )

    content: str = Field(
        min_length=1,
        max_length=10000,
    )


class NoteResponse(BaseModel):
    id: int
    title: str
    content: str


@app.get("/notes", response_model=list[NoteResponse])
async def get_notes(
    db: dict[int, dict[str, Any]] = Depends(get_notes_db),
) -> list[NoteResponse]:
    """Return a list of notes."""
    return [NoteResponse(**note) for note in db.values()]


@app.get("/notes/{id}", response_model=NoteResponse)
async def get_note(
    id: int,
    db: dict[int, dict[str, Any]] = Depends(get_notes_db),
) -> NoteResponse:
    """Return a specific note by ID."""
    if id not in db:
        raise HTTPException(status_code=404, detail="Note not found")
    return NoteResponse(**db[id])


@app.post("/notes", status_code=201, response_model=NoteResponse)
async def create_note(
    note: NoteCreate,
    db: dict[int, dict[str, Any]] = Depends(get_notes_db),
) -> NoteResponse:
    new_id = max(db.keys(), default=0) + 1
    record: dict[str, Any] = {"id": new_id, "title": note.title, "content": note.content}
    db[new_id] = record
    return NoteResponse(**record)


@app.delete("/notes/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_note(
    id: int,
    db: dict[int, dict[str, Any]] = Depends(get_notes_db),
) -> None:
    if id not in db:
        raise HTTPException(status_code=404, detail="Note not found")
    del db[id]
