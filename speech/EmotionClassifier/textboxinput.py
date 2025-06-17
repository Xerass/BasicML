import torch
import tkinter as tk
import json
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch.nn as nn
from tkinter import messagebox

#open up labels list

with open("saved_model/label_list.json", "r") as f:
    LABELS = json.load(f)
NUM_CLASSES = len(LABELS)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizerFast.from_pretrained("saved_model")

#define basic model, literal copy paste of training model
class EmotionClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, NUM_CLASSES)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        return self.fc(cls_output)

#load trained weights
model = EmotionClassifier()
model.load_state_dict(torch.load("saved_model/emotion.pt", map_location = device))
model.to(device)
model.eval()

#inference func
def detect_emotions(text, threshold = 0.5):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        results = [(LABELS[i], float(probs[i])) for i in range(len(LABELS)) if probs[i] > threshold]
        results.sort(key=lambda x: x[1], reverse=True)
    return results

#setup a gui
def on_input():
    text = entry.get("1.0", tk.END).strip()

    if not text:
        messagebox.showwarning("No input")
        return
    
    results = detect_emotions(text)

    if not results:
        output_text.set("No emotions can be detected with confidence")

    else:
        formatted = "\n".join([f"{label}: {score:.2f}" for label, score in results])
        output_text.set(f"Detected Emotions:\n\n{formatted}")

# Theme setup
BG_COLOR = "#1e1e1e"
FG_COLOR = "#d4d4d4"
BTN_COLOR = "#3c3c3c"
FONT_CODE = ("Consolas", 11)
FONT_LABEL = ("Arial", 12)

# Create root window
root = tk.Tk()
root.title("Emotion Detector 9000")
root.geometry("540x420")
root.configure(bg=BG_COLOR)

# Use a main frame to hold everything (ensures background color is consistent)
main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Label
tk.Label(
    main_frame,
    text="Enter text below:",
    font=FONT_LABEL,
    bg=BG_COLOR,
    fg=FG_COLOR,
    anchor="w"
).pack(pady=(5, 5), anchor="w")

# Text input
entry = tk.Text(
    main_frame,
    height=6,
    width=64,
    bg="#2e2e2e",
    fg=FG_COLOR,
    insertbackground=FG_COLOR,  # makes cursor visible
    font=FONT_CODE,
    wrap=tk.WORD,
    bd=1,
    relief=tk.FLAT
)
entry.pack(pady=(0, 10), fill="x")

# Button
tk.Button(
    main_frame,
    text="Detect Emotions",
    command=on_input,
    font=FONT_LABEL,
    bg=BTN_COLOR,
    fg=FG_COLOR,
    activebackground="#555",
    activeforeground=FG_COLOR,
    padx=10,
    pady=5,
    relief=tk.FLAT
).pack(pady=(0, 10))

# Output text area
output_text = tk.StringVar()
tk.Label(
    main_frame,
    textvariable=output_text,
    font=FONT_CODE,
    bg=BG_COLOR,
    fg=FG_COLOR,
    justify="left",
    wraplength=480,
    anchor="w"
).pack(pady=(5, 5), fill="both", expand=True)

root.mainloop()