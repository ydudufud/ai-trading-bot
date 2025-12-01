const express = require('express');
const fs = require('fs').promises;
const path = require('path');
require('dotenv').config();

const app = express();
app.use(express.json());

const ADMIN_TOKEN = process.env.ADMIN_TOKEN || 'change-me';

function checkAuth(req, res, next){
  const t = req.headers['x-admin-token'] || '';
  if(t !== ADMIN_TOKEN) return res.status(401).json({error:'unauthorized'});
  next();
}

const PROJECT_ROOT = process.env.PROJECT_ROOT || process.cwd();
function safePath(p){
  const resolved = path.resolve(PROJECT_ROOT, p);
  if(!resolved.startsWith(PROJECT_ROOT)) throw new Error('unsafe path');
  return resolved;
}

app.get('/', (req, res) => res.json({status:'AI Company API Online'}));

app.post('/agent/edit', checkAuth, async (req, res) => {
  try{
    const { filePath, content } = req.body;
    if(!filePath) return res.status(400).json({error:'filePath required'});
    const safe = safePath(filePath);
    await fs.mkdir(path.dirname(safe), { recursive:true });
    await fs.writeFile(safe, content, 'utf8');
    return res.json({status:'ok', path:filePath});
  }catch(err){
    return res.status(500).json({error:String(err)});
  }
});

app.post('/agent/create', checkAuth, async (req, res) => {
  try{
    const { filePath, content } = req.body;
    const safe = safePath(filePath);
    const exists = await fs.stat(safe).then(()=>true).catch(()=>false);
    if(exists) return res.status(400).json({error:'file exists'});
    await fs.mkdir(path.dirname(safe), { recursive:true });
    await fs.writeFile(safe, content, 'utf8');
    return res.json({status:'created', path:filePath});
  }catch(err){
    return res.status(500).json({error:String(err)});
  }
});

const PORT = process.env.PORT || 4040;
app.listen(PORT, ()=> console.log(`Server running on ${PORT}`));
