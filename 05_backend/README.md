# FastAPI Backend Microservice

## Overview
Exposes RESTful API endpoints integrating the four research components and serving the React frontend.

## Key Features
- FastAPI asynchronous routing with versioned endpoints (`/api/v1`)
- Isolated adapters per component (`app/adapters/`)
- Orchestration and decision flows (`app/orchestration/`)
- Pydantic validation schemas (`app/schemas/`)

## Run API Server
```bash
uvicorn app.main:app --reload --port 8000
```
