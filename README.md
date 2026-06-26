# AI Agent Service

这是一套独立的 AI Agent 示例项目，包含：

- Python FastAPI 后端
- OpenAI 兼容的大模型接口
- 原生 Python 文件读取工具
- Node.js + Express 前端
- 多轮对话聊天界面

## 目录结构

- `main.py` - FastAPI 后端入口
- `config.py` - 环境变量和默认配置
- `requirements.txt` - Python 依赖
- `server.js` - Node 前端服务入口
- `package.json` - Node 依赖和启动脚本
- `public/` - 前端静态页面
- `test.txt` - 示例文本文件

## 后端启动

```powershell
cd "E:\COMSOL62\Multiphysics\eg\AI agent-1"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## 前端启动

```powershell
cd "E:\COMSOL62\Multiphysics\eg\AI agent-1"
npm install
npm start
```

前端地址：

```text
http://localhost:3000
```

后端地址：

```text
http://localhost:8000
```

## 环境变量

- `DEBUG=True` - 打印大模型请求和响应日志
- `AI_AGENT_BASE_URL` - OpenAI 兼容 base_url
- `AI_AGENT_API_KEY` - API Key
- `AI_AGENT_MODEL` - 模型名
- `AI_AGENT_WORKSPACE_ROOT` - 文件读取默认根目录
- `AI_AGENT_CORS_ORIGINS` - 逗号分隔的前端来源

## 接口

- `POST /chat`
- `GET /health`
- `GET /config`

`/chat` 请求示例：

```json
{
  "message": "读取 /tmp/test.txt",
  "history": [
    { "role": "user", "content": "读取 /tmp/test.txt" }
  ]
}
```

## 验收提示

如果你想直接测试文件读取，可以先确认 `test.txt` 已存在，或者让后端读取 `/tmp/test.txt`，后端会优先尝试系统临时目录与项目示例文件。
