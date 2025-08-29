import sqlite3, json
from time import time
import voyageai
from dotenv import load_dotenv
from src.config import config
from src.algorithms.common import CommonAlgorithm
from src.algorithms.snurt_engine import SnurtEngine

load_dotenv()

try:
    vo = voyageai.Client()
except:
    vo = None

def encode_fn(texts):
    if vo is None:
        return [[0.1] * 384 for _ in texts]
    return vo.embed(texts, model="voyage-3.5-lite", input_type="document").embeddings

class ConsumerMemory:
    def __init__(self, db_path, provider_id, encode_fn):
        self.db = sqlite3.connect(db_path)
        self.provider_id = provider_id
        self.encode = encode_fn
        self.h = config.HALF_LIFE_DAYS
        self.tau = config.TAU_ABS
        self.delta = config.DELTA
        self.eta = config.ETA
        self.k = config.K
        self.snurt_engine = SnurtEngine()
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.execute("PRAGMA foreign_keys=ON;")

    def init_schema(self):
        cur = self.db.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS topics(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id TEXT NOT NULL,
              name TEXT NOT NULL,
              S REAL NOT NULL DEFAULT 0.0,
              t REAL NOT NULL DEFAULT 0.0,
              UNIQUE(user_id,name)
            );
            CREATE TABLE IF NOT EXISTS centroids(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              topic_id INTEGER NOT NULL,
              provider_id TEXT NOT NULL,
              vec TEXT NOT NULL,
              count INTEGER NOT NULL DEFAULT 1,
              FOREIGN KEY(topic_id) REFERENCES topics(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS messages(
              id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              text TEXT NOT NULL,
              ts REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS topic_members(
              topic_id INTEGER NOT NULL,
              msg_id TEXT NOT NULL,
              UNIQUE(topic_id,msg_id),
              FOREIGN KEY(topic_id) REFERENCES topics(id) ON DELETE CASCADE,
              FOREIGN KEY(msg_id) REFERENCES messages(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_topics_user ON topics(user_id);
            CREATE INDEX IF NOT EXISTS ix_centroids_topic ON centroids(topic_id);
            """
        )
        self.db.commit()

    def _now(self):
        return time()

    def _topic_id(self, user_id, name):
        row = self.db.execute("SELECT id FROM topics WHERE user_id=? AND name=?", (user_id,name)).fetchone()
        if row: return int(row[0])
        cur = self.db.execute("INSERT INTO topics(user_id,name,S,t) VALUES(?,?,0.0,0.0)", (user_id,name))
        self.db.commit()
        return cur.lastrowid

    def _centroids_for_user(self, user_id):
        sql = "SELECT c.id,c.topic_id,c.vec FROM centroids c JOIN topics t ON t.id=c.topic_id WHERE t.user_id=? AND c.provider_id=?"
        for cid, tid, vjson in self.db.execute(sql, (user_id,self.provider_id)):
            yield int(cid), int(tid), json.loads(vjson)

    def _nearest(self, user_id, v):
        best = (None,None,-1.0)
        second = -1.0
        for cid, tid, cvec in self._centroids_for_user(user_id):
            s = CommonAlgorithm.cosine_similarity(v, cvec)
            if s > best[2]:
                second = best[2]
                best = (cid, tid, s)
            elif s > second:
                second = s
        return best, second

    def _update_centroid(self, centroid_id, v):
        row = self.db.execute("SELECT vec,count FROM centroids WHERE id=?", (centroid_id,)).fetchone()
        if not row: return
        old, cnt = json.loads(row[0]), int(row[1])
        new = CommonAlgorithm.normalize_vector([(cnt*o + n)/(cnt+1) for o,n in zip(old,v)])
        self.db.execute("UPDATE centroids SET vec=?, count=? WHERE id=?", (json.dumps(new), cnt+1, centroid_id))

    def _add_centroid(self, topic_id, v):
        self.db.execute("INSERT INTO centroids(topic_id,provider_id,vec,count) VALUES(?,?,?,?)", (topic_id,self.provider_id,json.dumps(v),1))

    def assign(self, user_id, text, allow_new=True):
        v = CommonAlgorithm.normalize_vector(self.encode([text])[0])
        (cid, tid, s), s2 = self._nearest(user_id, v)
        if cid is None:
            if not allow_new: return -1, 0.0
            name = (" ".join(text.split())[:32] or "Topic")
            tid = self._topic_id(user_id, f"{name} · {int(self._now())}")
            self._add_centroid(tid, v)
            self.db.commit()
            return tid, 1.0
        if s >= self.tau and (s - s2) >= self.delta:
            return tid, s
        if not allow_new:
            return tid, s
        name = (" ".join(text.split())[:32] or "Topic")
        tid = self._topic_id(user_id, f"{name} · {int(self._now())}")
        self._add_centroid(tid, v)
        self.db.commit()
        return tid, 1.0

    def ingest(self, user_id, msg_id, text, ts=None, dwell_seconds=None, session_freq=None, clicked=0, provided=0):
        ts = ts or self._now()
        self.db.execute("INSERT OR REPLACE INTO messages(id,user_id,text,ts) VALUES(?,?,?,?)", (msg_id,user_id,text,ts))
        tid, _ = self.assign(user_id, text, allow_new=True)
        v = CommonAlgorithm.normalize_vector(self.encode([text])[0])
        row = self.db.execute("SELECT id FROM centroids WHERE topic_id=? AND provider_id=? ORDER BY id ASC", (tid,self.provider_id)).fetchone()
        if row: self._update_centroid(row[0], v)
        else: self._add_centroid(tid, v)
        
        row = self.db.execute("SELECT S,t FROM topics WHERE id=?", (tid,)).fetchone()
        S_prev, t_prev = float(row[0]), float(row[1])
        
        event_score = self.snurt_engine.snurt_reg(text, dwell_seconds, session_freq, 0.0, 0.0, clicked, provided)
        S_now = self.snurt_engine.update_topic_score(S_prev, t_prev, ts, event_score)
        
        self.db.execute("UPDATE topics SET S=?, t=? WHERE id=?", (S_now, ts, tid))
        self.db.execute("INSERT OR IGNORE INTO topic_members(topic_id,msg_id) VALUES(?,?)", (tid, msg_id))
        self.db.commit()
        return tid

    def scores(self, user_id, now_ts=None):
        now_ts = now_ts or self._now()
        rows = self.db.execute("SELECT name,S,t FROM topics WHERE user_id=?", (user_id,)).fetchall()
        names, vals = [], []
        for name, S, tlast in rows:
            names.append(name)
            vals.append(CommonAlgorithm.exponential_decay(float(S), float(tlast), now_ts, self.h))
        if not vals: return {}
        normalized = self.snurt_engine.normalize_scores(vals)
        return {n: score for n, score in zip(names, normalized)}

if __name__ == "__main__":
    db = ConsumerMemory("topics.db", provider_id="voyage-3.5-lite", encode_fn=encode_fn)
    db.init_schema()
    msgs = [
        ("m1","how do I fix my git push error?"),
        ("m2","git remote and branches confusion"),
        ("m3","SAT reading score curve and how many wrong for 1600"),
        ("m4","SAT practice test timing strategies"),
        ("m5","github actions failing on main pipeline")
    ]
    for mid, text in msgs:
        db.ingest("tarush", mid, text)
    print(db.scores("tarush"))