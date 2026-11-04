"""Exact-signature SQLite result cache. No pickle or executable payloads."""
import json
from pathlib import Path
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from .contracts import canonical, digest

class MappingCache:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS mappings (key TEXT PRIMARY KEY, signature TEXT NOT NULL, result TEXT NOT NULL, checksum TEXT NOT NULL, created_at TEXT NOT NULL)')
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()
    def get(self,signature):
        key=digest(signature)
        with self.connect() as db:row=db.execute('SELECT signature,result,checksum FROM mappings WHERE key=?',(key,)).fetchone()
        if not row or row[0]!=canonical(signature):return None
        try:result=json.loads(row[1])
        except (ValueError,TypeError):return None
        if digest(result)!=row[2]:return None
        return result
    def put(self,signature,result):
        with self.connect() as db:db.execute('INSERT OR REPLACE INTO mappings VALUES (?,?,?,?,?)',(digest(signature),canonical(signature),canonical(result),digest(result),datetime.now(timezone.utc).isoformat()))
    def count(self):
        with self.connect() as db:return db.execute('SELECT COUNT(*) FROM mappings').fetchone()[0]
    def clear(self):
        with self.connect() as db:db.execute('DELETE FROM mappings')
