# Python / FastAPI → Go mapping

For quick reference when porting the AI Agent Platform to Go.

| Python / FastAPI | Go equivalent | Concept |
|---|---|---|
| `structlog` | `zap` / `zerolog` / `slog` | Structured logging |
| `@asynccontextmanager` + `yield` | `main()` with `defer` + `<-ctx.Done()` | Startup / shutdown lifecycle |
| `FastAPI(...)` | `chi.NewRouter()` / `gin.Engine` | The app / router |
| `@app.get("/health")` | `r.Get("/health", handler)` | Route registration |
| `async def` / `await` | goroutines + channels | Concurrency model |
| `return {"status": "ok"}` | `json.NewEncoder(w).Encode(...)` | Auto JSON serialization |
| `@decorator` | explicit function wrapping / middleware | Behavior augmentation |



If the breaker counted every retry failure, a single transient outage could generate multiple failure events and trip the circuit prematurely. By counting only the final outcome of the entire call (after retries are exhausted), retries and circuit breakers work together: retries handle short-lived failures, while the breaker reacts only when requests genuinely fail end-to-end.