from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from PIL import Image
import base64,io

app = FastAPI()

# 添加跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return HTMLResponse("""
        <html>
        <body>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit">
            </form>
        </body>
        </html>
    """)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_base64 = base64.b64encode(contents).decode("utf-8")

    # 在这里添加对上传图片的处理逻辑
    # 返回处理结果，包括Base64编码的图片
    return {"filename": file.filename, "image_base64":"data:image/" + image.format.lower() + ";base64," + image_base64}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3033)