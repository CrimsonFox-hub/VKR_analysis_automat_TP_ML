import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000"

def query_arch(text, architecture):
    payload = {"text": text, "architecture": architecture}
    try:
        resp = requests.post(API_URL + "/query", json=payload, timeout=120)
        data = resp.json()
    except Exception as e:
        return (f"Ошибка: {e}", [], 0, "", 0, "", "", "", 0,0,0,0)

    answer = data.get("answer", "")
    sources = data.get("sources", [])
    time_val = data.get("p_time", 0)
    category = data.get("category", "")
    confidence = data.get("confidence", 0)
    tags = data.get("tags", [])
    if isinstance(tags, str):          # на случай, если tags — строка
        tags = [t.strip() for t in tags.split(',') if t.strip()]
    answer_prec = data.get("answer_precision")
    answer_rec = data.get("answer_recall")
    answer_f1 = data.get("answer_f1")
    context_prec = data.get("context_precision")

    return (
        answer,
        sources,
        round(time_val, 3),
        category,
        round(confidence, 4) if confidence else "",
        ", ".join(tags) if tags else "",
        round(answer_f1, 4) if answer_f1 else 0.0,
        round(context_prec, 4) if context_prec else 0.0,
    )

with gr.Blocks(title="Техподдержка — сравнение МЛ и ЛЛМ архитектур") as demo:
    gr.Markdown("## Сравнение архитектур технической поддержки")

    with gr.Row():
        text_input = gr.Textbox(label="Запрос", lines=3)
        arch_radio = gr.Radio(choices=["A", "B"], label="Архитектура", value="A")
        submit_btn = gr.Button("Отправить", variant="primary")

    with gr.Row():
        answer_output = gr.Textbox(label="Ответ", lines=6)
        with gr.Column():
            time_output = gr.Number(label="Время (сек)")
            category_output = gr.Textbox(label="Категория")
            confidence_output = gr.Number(label="Уверенность")
            tags_output = gr.Textbox(label="Теги")

    with gr.Row():
        answer_f1_out = gr.Number(label="Answer F1")
        context_prec_out = gr.Number(label="Context Precision")

    with gr.Accordion("Источники", open=False):
        sources_output = gr.JSON(label="Документы")

    submit_btn.click(
        fn=query_arch,
        inputs=[text_input, arch_radio],
        outputs=[
            answer_output, sources_output, time_output, category_output,
            confidence_output, tags_output,
            answer_f1_out, context_prec_out
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)