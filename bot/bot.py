from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

from src.predict import SentimentModel
import os
from dotenv import load_dotenv
load_dotenv()


TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")

app = ApplicationBuilder().token(TOKEN).build()


model = SentimentModel(
    "models/model.pkl",
    "models/vectorizer.pkl"
)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    result = model.predict(text)


    label = result["label"]
    confidence = result["confidence"]

    emoji = {
    "positive": "😊",
    "neutral": "😐",
    "negative": "😠"
    }

    await update.message.reply_text(
    f"{emoji[label]} Тональность: *{label}*\n"
    f"Уверенность модели: *{confidence:.1%}*",
    parse_mode="Markdown"
)
    




def main():
    app = ApplicationBuilder().token("8012889857:AAHDbfqofD3rK1MemI3dIgWqUivJbxBpMkw").build()
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
