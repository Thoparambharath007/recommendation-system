"""Run with Python, or double-click start.bat."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT / '.vendor'))
if __name__=='__main__':
    import uvicorn
    uvicorn.run('app:app',host='127.0.0.1',port=int(os.getenv('SHIRT_APP_PORT','8010')))
