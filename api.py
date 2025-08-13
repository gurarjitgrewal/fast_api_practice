from fastapi import FastAPI
from pydantic import BaseModel
from model import SimpleEmailData, DynamicSpamDetector

app = FastAPI(title="Dynamic Spam Detector API")

# Create instances
data_gen = SimpleEmailData()
detector = DynamicSpamDetector()


class EmailRequest(BaseModel):
    text: str


class newData(BaseModel):
    text: str
    label: int

email=[]
label=[]

@app.post("/train")
def train_model():
    emails = []
    labels = []
    for _ in range(50):
        e, l = data_gen.generate_email()
        emails.append(e)
        labels.append(l)
    email.extend(emails)
    label.extend(labels)
    #print(email,label)
    detector.initial_training(emails, labels)
    
    return {"message": "Initial training complete", "total_emails": len(detector.all_emails)}
    

@app.post("/predict")
def predict_email(request: EmailRequest):
    label, confidence = detector.predict_email(request.text)
    return {
        "prediction": "spam" if label == 1 else "not spam",
        "confidence": confidence
    }


@app.post("/new-input")
def new_input(request: newData):
    detector.learn_from_new_email(request.text, request.label)
    return {"message": "New input recorded and model retrained"}


@app.get("/evaluate")
def evaluate_model():
    emails1 = []
    labels1 = []
    for _ in range(30):
        e, l = data_gen.generate_email()
        emails1.append(e)
        labels1.append(l)
    email.extend(emails1)
    label.extend(labels1)
    acc = detector.evaluate(email, label)
    return {"accuracy": acc}

