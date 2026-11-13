import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Loan Approval Predictor")

model = joblib.load("/content/loan_model.pkl")
prep = joblib.load("/content/preprocessor.pkl")

class Applicant(BaseModel):
    Age: float
    Gender: str
    Marital_Status: str
    Education: str
    Employment_Type: str
    City: str
    Years_Experience: float
    Monthly_Income: float
    Loan_Amount: float
    Credit_Score: float

@app.post("/predict")
def predict(data: Applicant):
    df = pd.DataFrame([data.dict()])
    X = prep.transform(df)
    prob = float(model.predict_proba(X)[0][1])
    return {"approved": prob >= 0.5, "probability": round(prob * 100, 1)}

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loan Approval Predictor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  min-height:100vh;
  font-family:'Segoe UI',Roboto,Arial,sans-serif;
  background:linear-gradient(135deg,#1e1b4b,#4c1d95 50%,#be185d);
  background-attachment:fixed;
  color:#fff;
  padding:20px 14px 40px;
}
.blob{position:fixed;border-radius:50%;filter:blur(70px);opacity:.45;z-index:-1}
.b1{width:260px;height:260px;background:#22d3ee;top:-60px;left:-60px}
.b2{width:300px;height:300px;background:#f472b6;bottom:-80px;right:-80px}
.card{
  max-width:480px;margin:auto;padding:24px 20px;
  background:rgba(255,255,255,.1);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  border:1px solid rgba(255,255,255,.22);
  border-radius:26px;
  box-shadow:0 20px 50px rgba(0,0,0,.35);
}
.logo{font-size:46px;text-align:center}
h1{
  text-align:center;font-size:26px;margin-top:4px;
  background:linear-gradient(90deg,#67e8f9,#c4b5fd,#f9a8d4);
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.sub{text-align:center;font-size:13px;color:#ddd6fe;margin:6px 0 10px}
.section{
  margin-top:20px;font-size:14px;font-weight:700;letter-spacing:.5px;
  color:#a5f3fc;border-bottom:1px solid rgba(255,255,255,.18);padding-bottom:6px;
}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.full{grid-column:1 / -1}
label{display:block;margin-top:12px;font-size:12px;color:#e9d5ff}
input,select{
  width:100%;margin-top:5px;padding:12px;font-size:15px;color:#fff;
  background:rgba(255,255,255,.1);
  border:1px solid rgba(255,255,255,.25);border-radius:12px;outline:none;
  transition:.2s;
}
input:focus,select:focus{border-color:#67e8f9;box-shadow:0 0 0 3px rgba(103,232,249,.3)}
option{background:#1e1b4b;color:#fff}
input[type=range]{padding:0;height:8px;accent-color:#f472b6;background:transparent;border:0}
.score{display:flex;justify-content:space-between;align-items:center}
#scoreVal{
  font-weight:700;font-size:16px;color:#fff;
  background:linear-gradient(90deg,#8b5cf6,#ec4899);
  padding:2px 12px;border-radius:20px;
}
button{
  width:100%;margin-top:26px;padding:15px;border:0;border-radius:14px;
  font-size:17px;font-weight:700;color:#fff;cursor:pointer;
  background:linear-gradient(90deg,#06b6d4,#8b5cf6,#ec4899);
  box-shadow:0 10px 25px rgba(139,92,246,.5);
  transition:.2s;
}
button:active{transform:scale(.97)}
#result{display:none;margin-top:24px;padding:20px;border-radius:20px;text-align:center}
.ok{background:linear-gradient(135deg,rgba(16,185,129,.35),rgba(5,150,105,.2));border:1px solid #34d399}
.no{background:linear-gradient(135deg,rgba(239,68,68,.35),rgba(185,28,28,.2));border:1px solid #f87171}
.ring{
  --p:0;--c:#34d399;
  width:140px;height:140px;border-radius:50%;margin:auto;
  display:flex;align-items:center;justify-content:center;
  background:conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,.15) 0);
}
.inner{
  width:108px;height:108px;border-radius:50%;background:#2a1a5e;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
}
.pct{font-size:28px;font-weight:800}
.small{font-size:11px;color:#ddd6fe}
#status{margin-top:14px;font-size:22px;font-weight:800}
#msg{margin-top:4px;font-size:13px;color:#e9d5ff}
.foot{text-align:center;margin-top:18px;font-size:11px;color:#c4b5fd}
</style>
</head>
<body>
<div class="blob b1"></div><div class="blob b2"></div>

<div class="card">
  <div class="logo">🏦</div>
  <h1>Loan Approval Predictor</h1>
  <p class="sub">AI-powered instant loan decision</p>

  <div class="section">👤 Personal Details</div>
  <div class="grid">
    <div><label>Age<input id="Age" type="number" value="35"></label></div>
    <div><label>Gender<select id="Gender"><option>Male</option><option>Female</option></select></label></div>
    <div><label>Marital Status<select id="Marital_Status"><option>Single</option><option>Married</option><option>Divorced</option></select></label></div>
    <div><label>City<select id="City"><option>Karak</option><option>Kohat</option><option>Peshawar</option><option>Lahore</option><option>Islamabad</option></select></label></div>
  </div>

  <div class="section">💼 Work and Education</div>
  <div class="grid">
    <div><label>Education<select id="Education"><option>Matric</option><option>Intermediate</option><option selected>Bachelors</option><option>Masters</option></select></label></div>
    <div><label>Employment<select id="Employment_Type"><option>Salaried</option><option>Self-Employed</option><option>Business</option><option>Unemployed</option></select></label></div>
    <div class="full"><label>Years of Experience<input id="Years_Experience" type="number" value="10"></label></div>
  </div>

  <div class="section">💰 Finance</div>
  <div class="grid">
    <div><label>Monthly Income<input id="Monthly_Income" type="number" value="150000"></label></div>
    <div><label>Loan Amount<input id="Loan_Amount" type="number" value="500000"></label></div>
    <div class="full">
      <label>
        <div class="score"><span>Credit Score</span><span id="scoreVal">700</span></div>
        <input id="Credit_Score" type="range" min="300" max="850" value="700">
      </label>
    </div>
  </div>

  <button id="btn" onclick="send()">✨ Check Approval</button>

  <div id="result">
    <div class="ring" id="ring"><div class="inner"><div class="pct" id="pct">0%</div><div class="small">approval chance</div></div></div>
    <div id="status"></div>
    <div id="msg"></div>
  </div>

  <div class="foot">Built with FastAPI and Random Forest 🌲</div>
</div>

<script>
const slider = document.getElementById("Credit_Score");
slider.oninput = function(){ document.getElementById("scoreVal").innerText = slider.value; };

async function send(){
  const ids = ["Age","Gender","Marital_Status","Education","Employment_Type","City","Years_Experience","Monthly_Income","Loan_Amount","Credit_Score"];
  const nums = ["Age","Years_Experience","Monthly_Income","Loan_Amount","Credit_Score"];
  const body = {};
  ids.forEach(function(id){
    const v = document.getElementById(id).value;
    body[id] = nums.includes(id) ? parseFloat(v) : v;
  });

  const btn = document.getElementById("btn");
  btn.innerText = "⏳ Analyzing...";
  try{
    const r = await fetch("/predict", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(body)
    });
    const d = await r.json();
    const box = document.getElementById("result");
    const ring = document.getElementById("ring");
    box.style.display = "block";
    box.className = d.approved ? "ok" : "no";
    ring.style.setProperty("--p", d.probability);
    ring.style.setProperty("--c", d.approved ? "#34d399" : "#f87171");
    document.getElementById("pct").innerText = d.probability + "%";
    document.getElementById("status").innerText = d.approved ? "✅ Loan Approved" : "❌ Loan Rejected";
    document.getElementById("msg").innerText = d.approved
      ? "Strong profile. This application looks good."
      : "Risk is high for this profile. Try a lower loan amount.";
    box.scrollIntoView({behavior:"smooth"});
  }catch(e){
    alert("Error: could not reach the server.");
  }
  btn.innerText = "✨ Check Approval";
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML
