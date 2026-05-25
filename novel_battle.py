import os
import base64
from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 1. 環境変数の読み込み
load_dotenv()

# 画像データをAIに送れる形式（Base64）に変換する関数
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 2. データベース（Chroma）の準備
def setup_database():
    print("データベースを構築中...")
    loader = TextLoader("data/character_data.txt", encoding="utf-8")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(separator="---", chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    # データベースの検索用モデルもOpenAIの高精度なものを使用
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
    
    print("データベースの構築完了！")
    return vectorstore.as_retriever()

retriever = setup_database()

# 3. マルチモーダル対応のチャット関数
def generate_novel(message, history):
    # Gradioのmultimodal=Trueの場合、messageの中にテキストと画像が両方入ってきます
    text_input = message["text"]
    files = message.get("files", [])

    # ユーザーの入力テキストを使って、ChromaDBから設定資料を検索
    docs = retriever.invoke(text_input)
    context = "\n".join([doc.page_content for doc in docs])

    # AIに渡すメッセージの中身を組み立てる
    user_content = [{"type": "text", "text": f"【ユーザーの指示】\n{text_input}"}]

    # もし画像がアップロードされていれば、Vision機能のために画像データを追加
    if len(files) > 0:
        image_path = files[0]
        base64_image = encode_image(image_path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    # AIへの最終的な指示書
    messages = [
        SystemMessage(content=f"""
        あなたは天才的なダークファンタジー小説家です。
        提供された【設定資料】を守り、戦闘シーンを描写してください。
        もしユーザーからキャラクターの画像が提供された場合は、その視覚情報（外見、武器、色合い、雰囲気など）を深く読み取り、描写に反映させてください。
        
        【設定資料】
        {context}
        """),
        HumanMessage(content=user_content)
    ]

    # GPT-4o-mini（VLM）を呼び出す
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    response = llm.invoke(messages)
    
    return response.content

with open("data/character_data.txt", "r", encoding="utf-8") as f:
    character_info_text = f.read()

# 4. Gradio UI設定（Blocksを使ってレイアウトをカスタマイズ）
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚔️ VLMバトルノベルジェネレーター")
    gr.Markdown("左下のクリップマークからキャラクターの画像をアップロードし、「この画像の子 vs ガルバ」のように指示を出してみてください！")
    
    # ここに折りたたみメニューを追加！
    with gr.Accordion("📚 現在登録されているキャラクターと世界観を見る", open=False):
        gr.Markdown(f"```text\n{character_info_text}\n```")
    
    # メインのチャット画面
    gr.ChatInterface(
        fn=generate_novel,
        multimodal=True
    )

if __name__ == "__main__":
    demo.launch()