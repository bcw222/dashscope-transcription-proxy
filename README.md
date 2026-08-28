# DashScope Transcription Proxy

将 [阿里云百炼 DashScope](https://help.aliyun.com/zh/model-studio/)（Aliyun Bailian） 语音识别包装为  
**OpenAI-compatible `POST /audio/transcriptions`** 代理，可直接替换 Open WebUI
等下游的 OpenAI STT 后端。

> **AI 项目声明**：本项目为**纯 AI 项目**——仓库主要由 AI（LLM）生成，人类仅负责需求描述与验收，对代码的可靠性不做保证。

## 支持模型

| 模型 | 模式 | 时长限制 | 说明 |
|------|------|---------|------|
| `qwen3-asr-flash` | 同步 | ≤ 5 min / 10 MB | **默认**，支持情感识别、热词、上下文增强 |
| `qwen3-asr-flash-us` | 同步 | ≤ 5 min / 10 MB | 美国地域版本 |
| `qwen-audio-3.0-asr-flash` | 同步 | ≤ 5 min / 10 MB | 响应结构不同，代理已兼容 |
| `fun-asr-flash` | 同步 | ≤ 5 min / 10 MB | 响应结构不同，代理已兼容 |

> 以上同步模型均可通过请求体的 `model` 字段动态指定，或设置
> `DASHSCOPE_PROXY_MODEL` 作为默认值。
>
> 长音频异步模型（`*-filetrans`、`fun-asr`、`paraformer` 系列）暂未实现——
> 它们需要公网音频 URL + 任务轮询，与当前「本地文件 → 同步返回」的模式不同，
> 未来可通过 OSS 托管 + 回调扩展。

## 快速启动

```bash
# 1. 安装
pip install -e .

# 2. 设置 DashScope 端点
export DASHSCOPE_PROXY_BASE_HTTP_API_URL="https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1"

# 3. 启动
python -m dashscope_transcription_proxy
# → http://0.0.0.0:8000
```

## 认证

**API Key 由下游请求透传** — 代理从 `Authorization: Bearer <dashscope-key>` 提取，
然后传给 DashScope SDK。无需在代理侧配置 Key。

```bash
curl http://localhost:8000/audio/transcriptions \
  -H "Authorization: Bearer sk-xxxxxxxxxxxxxxxx" \
  -F "file=@audio.mp3"
```

也支持 `X-API-Key` 头作为备选。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DASHSCOPE_PROXY_BASE_HTTP_API_URL` | — | DashScope 基础 URL |
| `DASHSCOPE_PROXY_MODEL` | `qwen3-asr-flash` | 模型 ID |
| `DASHSCOPE_PROXY_ENABLE_ITN` | `true` | 逆文本正则化 |
| `DASHSCOPE_PROXY_ENABLE_LID` | `false` | 自动语种检测 |
| `DASHSCOPE_PROXY_DEFAULT_LANGUAGE` | `zh` | 默认语言 |
| `DASHSCOPE_PROXY_RESULTS_DIR` | — | 保存原始结果 JSON |
| `DASHSCOPE_PROXY_LOCAL_MODE` | `false` | 返回 mock 数据 |
| `DASHSCOPE_PROXY_MAX_UPLOAD_MB` | `10.0` | 上传音频大小上限（MB），`0` 表示不限 |
| `LOG_LEVEL` | `info` | 日志级别 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |

## API

### `POST /audio/transcriptions`

兼容 OpenAI `/audio/transcriptions`。支持：

**multipart/form-data：**

```bash
curl http://localhost:8000/audio/transcriptions \
  -H "Authorization: Bearer sk-xxx" \
  -F "file=@audio.mp3" \
  -F "model=qwen3-asr-flash" \
  -F "language=zh"
```

**application/json：**

```bash
curl http://localhost:8000/audio/transcriptions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-asr-flash",
    "language": "zh",
    "response_format": "verbose_json",
    "input_audio": {"data": "data:audio/mpeg;base64,..."}
  }'
```

### 响应格式

| `response_format` | 返回 | Content-Type |
|-------------------|------|--------------|
| `json` (默认) | `{"text": "..."}` | `application/json` |
| `text` | 纯文本 | `text/plain` |
| `verbose_json` | `text` + `words[]` + `segments[]` + `duration` + `language` + `usage` | `application/json` |
| `srt` | SubRip 字幕（需字级时间戳） | `application/x-subrip` |
| `vtt` | WebVTT 字幕（需字级时间戳） | `text/vtt` |

> 不支持的值返回 400；`stream=true` 返回 501（暂不支持流式）。

### `GET /health`

```json
{"status": "ok", "model": "qwen3-asr-flash", "local_mode": false}
```

## Docker

```bash
docker build -t dashscope-transcription-proxy .
docker run -p 8000:8000 \
  -e DASHSCOPE_PROXY_BASE_HTTP_API_URL="https://xxx.cn-beijing.maas.aliyuncs.com/api/v1" \
  dashscope-transcription-proxy
```

### GHCR

```bash
docker pull ghcr.io/<org>/dashscope-transcription-proxy:latest
```

镜像为**多架构**（`linux/amd64` + `linux/arm64`），在 x86 服务器、树莓派、Apple Silicon 上均可直接拉取，Docker 会自动选择对应架构。

CI 流水线在 `.github/workflows/ci.yml`，push 到 `main` 或打 `v*` tag 后自动构建并推送至 GHCR。tag 规则：

| 触发 | 镜像 tag |
|------|---------|
| push `main` | `latest` + `main` + `<short-sha>` |
| push `v1.2.3` | `1.2.3` + `1.2` |
| PR | `pr-<number>`（仅构建，不推送） |

## 开发

```bash
pip install -e ".[dev]"
uvicorn dashscope_transcription_proxy.app:app --reload

# 本地 mock 模式（不走 DashScope）
DASHSCOPE_PROXY_LOCAL_MODE=true python -m dashscope_transcription_proxy
```

## 下游配置示例（Open WebUI）

在 Open WebUI 中将 STT 引擎配置为 OpenAI：

- **API Base URL**: `http://your-host:8000`
- **API Key**: 你的 DashScope API Key（`sk-xxx`），代理会透传至百炼
- **STT Model**: `qwen3-asr-flash`

## 许可证

本项目基于 [PolyForm Noncommercial License 1.0.0](LICENSE) 发布——
**仅限非商业用途**（个人研究、学习、非营利组织等）。商业使用需另行获得授权，
详见 [LICENSE](LICENSE)。
