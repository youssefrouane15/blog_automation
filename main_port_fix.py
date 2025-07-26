# Add this to the END of your main.py file:

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Railway provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    
    # Run the app
    uvicorn.run("main:app", host="0.0.0.0", port=port)