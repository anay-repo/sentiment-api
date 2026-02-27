#!/bin/bash
pip install textblob
python -m textblob.download_corpora
```

Then in Render's **Settings → Build & Deploy**, change the **Build Command** to:
```
pip install -r requirements.txt && python -m textblob.download_corpora
