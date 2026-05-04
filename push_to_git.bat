@echo off
cd /d "c:\Users\huzai\OneDrive\Desktop\AI\backend"
git config user.email "huzaifa@example.com"
git config user.name "huzaifa-2006-mh"
git add .
git commit -m "Initial commit: MHS AI Backend"
git branch -M main
git remote add origin https://github.com/huzaifa-2006-mh/AI-backend.git
git push -u origin main
