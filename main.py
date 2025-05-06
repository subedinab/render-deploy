from routes_app.blog import app

# This is needed for Gunicorn
app = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 