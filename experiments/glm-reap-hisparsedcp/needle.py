import json,urllib.request,sys,time
target=int(sys.argv[1])
PORT=sys.argv[2] if len(sys.argv)>2 else "30000"
MODEL=sys.argv[3] if len(sys.argv)>3 else "glm52-reap"
SECRET="BANANA-42751-ZEBRA"
filler="The quick brown fox jumps over the lazy dog near the calm riverbank at dawn. "
needle=" IMPORTANT FACT: The secret vault passphrase is "+SECRET+". Memorize it carefully. "
nfill=max(1,target//16); half=int(nfill*0.55)
ctx=filler*half + needle + filler*(nfill-half)
prompt=ctx + "\n\nQuestion: What is the secret vault passphrase mentioned above? Reply with ONLY the passphrase."
payload={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":2048,
         "temperature":0.6,"top_p":0.95}
b=json.dumps(payload).encode()
t0=time.time()
try:
    req=urllib.request.Request("http://localhost:%s/v1/chat/completions"%PORT,data=b,headers={"Content-Type":"application/json"})
    r=json.load(urllib.request.urlopen(req,timeout=2400))
    dt=time.time()-t0; u=r.get("usage",{}) or {}
    msg=r["choices"][0]["message"]
    ans=(msg.get("content") or msg.get("reasoning_content") or "")
    print("target=%d prompt_tokens=%s wall=%.0fs FOUND=%s"%(target,u.get("prompt_tokens"),dt,SECRET in ans))
    print("answer:",ans[-120:].replace("\n"," "))
except Exception as e:
    print("target=%d ERROR after %.0fs: %s"%(target,time.time()-t0,e))
