# 🚀 Deploy Blog Automation to Railway - Simple Guide

## 📋 What You'll Need
- GitHub account (free at github.com)
- Railway account (free at railway.app)
- Your Blog Automation project with config.json

## 🎯 Overview
This guide uses the **simplest approach** - we'll deploy everything including your config.json file. Just make sure your GitHub repository is **PRIVATE** to keep your API keys secure.

---

## 📁 Step 1: Prepare Your Project

Your project already has most files needed. Just add this to the end of your `main.py`:

```python
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Railway provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    
    # Run the app
    uvicorn.run("main:app", host="0.0.0.0", port=port)
```

✅ **Files you already have:**
- `config.json` (with all your settings and API keys)
- `lbc-automation-process.json` (Google credentials)
- `requirements.txt`
- `Procfile`
- `railway.json`
- `.gitignore`

---

## 🔐 Step 2: Create PRIVATE GitHub Repository

1. **Go to [github.com](https://github.com)** and sign in
2. Click the **"+"** button (top right) → **"New repository"**
3. Fill in:
   - **Repository name:** `blog-automation`
   - **Description:** "Automated blog content generation system"
   - **⚠️ IMPORTANT: Set to PRIVATE** (because config.json contains API keys)
   - Click **"Create repository"**

---

## 📤 Step 3: Upload Your Code to GitHub

Open Command Prompt or Terminal and run these commands:

```bash
# Navigate to your project folder
cd C:\Users\rouan\OneDrive\Desktop\WORKSPACE\Automations\Blog_Automation

# Initialize git
git init

# Add all files (including config.json)
git add .

# Commit files
git commit -m "Initial commit - Blog Automation with config"

# Connect to GitHub (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/blog-automation.git

# Push to GitHub
git push -u origin main
```

**Note:** If it says "main" doesn't exist, use:
```bash
git branch -M main
git push -u origin main
```

---

## 🚂 Step 4: Deploy to Railway

1. **Go to [railway.app](https://railway.app)** and sign in
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. **Connect your GitHub account** if prompted
5. Find and select your **`blog-automation`** repository
6. Click **"Deploy Now"**

That's it! Railway will:
- ✅ Automatically detect it's a Python app
- ✅ Read your `config.json` file
- ✅ Use your Google credentials file
- ✅ Install all requirements
- ✅ Start your application

---

## 🌐 Step 5: Get Your App URL

1. In Railway dashboard, click on your project
2. Go to **"Settings"** tab
3. Under **"Domains"**, click **"Generate Domain"**
4. Railway will give you a URL like: `blog-automation-production.up.railway.app`
5. Your app is now live! Visit the URL to access it

---

## 📊 Step 6: Monitor Your App

1. In Railway dashboard, click **"View Logs"**
2. You should see:
   ```
   INFO:     Started server process
   INFO:     Waiting for application startup
   INFO:     Application startup complete
   INFO:     Uvicorn running on http://0.0.0.0:PORT
   ```
3. If you see any errors, check your config.json and credentials

---

## 🔄 How to Update Your App

Whenever you make changes:

```bash
# Make your changes
git add .
git commit -m "Update: description of what you changed"
git push origin main
```

Railway will **automatically redeploy** within seconds!

---

## ⚡ Quick Tips

### 1. **Keep Repository Private**
Since your config.json contains API keys, your GitHub repo MUST be private.

### 2. **Check Your Logs**
If something isn't working, always check Railway logs first:
- Railway Dashboard → Your Project → "View Logs"

### 3. **File Paths**
Make sure your config.json references correct file paths:
```json
{
  "google_sheets": {
    "credentials_file": "lbc-automation-process.json"
  }
}
```

### 4. **Free Tier Limits**
Railway's free tier includes:
- $5 credit per month
- 500 hours of usage
- Perfect for testing and small projects

---

## 🆘 Troubleshooting

### App won't start?
1. Check logs for specific error messages
2. Verify all files are uploaded (check GitHub repo)
3. Make sure `main.py` has the PORT configuration at the end

### Can't connect to WordPress?
1. Check your WordPress URLs in config.json include `/xmlrpc.php`
2. Verify WordPress has XML-RPC enabled
3. Confirm username/password are correct

### Google Sheets not working?
1. Verify `lbc-automation-process.json` is in your repository
2. Check if the service account has access to your Google Sheets
3. Confirm sheet IDs in config.json are correct

### Railway deployment failing?
1. Make sure `requirements.txt` has all dependencies
2. Check Python version compatibility
3. Verify `Procfile` exists and has: `web: uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## 🎉 Success Checklist

- [ ] GitHub repository is PRIVATE
- [ ] All files uploaded including config.json
- [ ] main.py has PORT configuration
- [ ] Railway deployment successful
- [ ] Generated domain URL works
- [ ] Can access app dashboard
- [ ] Logs show no errors

---

## 📚 Next Steps

1. **Set up monitoring** - Use Railway's metrics dashboard
2. **Configure alerts** - Get notified if app goes down
3. **Regular backups** - Download your data periodically
4. **Usage tracking** - Monitor your Railway credit usage

---

**That's it!** Your Blog Automation is now deployed on Railway. No complex environment variables needed - just your config.json and you're good to go! 🚀

**Need help?** 
- Railway Docs: [docs.railway.app](https://docs.railway.app)
- Check your deployment logs first - they usually tell you exactly what's wrong