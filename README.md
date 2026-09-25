# DashScope Transcription Proxy

将 [阿里云百炼 DashScope](https://help.aliyun.com/zh/model-studio/)（Aliyun Bailian）语音识别，
或**本地自建的 Qwen3-ASR 服务**（`backend=vllm`，见下文），包装为  
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
| `DASHSCOPE_PROXY_BACKEND` | `dashscope` | `dashscope`（云端）或 `vllm`（本地 Qwen3-ASR） |
| `DASHSCOPE_PROXY_REQUIRE_DOWNSTREAM_KEY` | `true` | 是否强制下游带 `Authorization`；纯内网本地部署可设 `false` |
| `DASHSCOPE_PROXY_VLLM_BASE_URL` | `http://127.0.0.1:8001/v1` | 本地 ASR 服务的 OpenAI 兼容 base URL |
| `DASHSCOPE_PROXY_VLLM_API_KEY` | — | 本地服务的 `--api-key`；留空则不带认证头 |
| `DASHSCOPE_PROXY_VLLM_MODEL` | `qwen3-asr` | 本地服务的 `--served-model-name`；**下游传的 `model` 会被忽略** |
| `DASHSCOPE_PROXY_VLLM_TIMEOUT` | `600` | 单请求超时（秒），长音频要给足 |
| `DASHSCOPE_PROXY_VLLM_MAX_TOKENS` | `4096` | 补全上限，长音频调大 |
| `DASHSCOPE_PROXY_VLLM_PASSTHROUGH_CONTEXT` | `true` | 把下游 `prompt` 作为 system message 传下去（热词/上下文） |
| `DASHSCOPE_PROXY_QWEN_OUTPUT_PARSER` | `auto` | Qwen3-ASR 原始输出拆分方式：`auto`/`official`/`none` |

## 本地后端（`backend=vllm`）—— 离线 Qwen3-ASR

除了转发到百炼云端，本代理还能把请求打到**本地自建的 Qwen3-ASR 服务**
（官方 `qwen-asr-serve`，即 `vllm serve` 的包装，暴露 OpenAI 兼容的
`POST /v1/chat/completions` + `audio_url`）。下游（Open WebUI 等）仍然只看到
一个标准的 `/v1/audio/transcriptions`，而推理跑在你自己的 GPU 上：
**零 API 费用、可完全离线**。

```bash
DASHSCOPE_PROXY_BACKEND=vllm \
DASHSCOPE_PROXY_VLLM_BASE_URL=http://asr:8000/v1 \
DASHSCOPE_PROXY_VLLM_MODEL=qwen3-asr \
DASHSCOPE_PROXY_REQUIRE_DOWNSTREAM_KEY=false \
python -m dashscope_transcription_proxy
```

### 与云端后端的能力差异

| 能力 | `dashscope` | `vllm`（本地） |
|---|---|---|
| `json` / `text` / `verbose_json` | ✅ | ✅ |
| `srt` / `vtt` | ✅（字级时间戳） | ❌ **返回 400** —— vLLM 的 chat 面不返回字级时间戳 |
| `verbose_json` 的 `words[]` | ✅ | 恒为空 |
| `verbose_json` 的 `language` | ✅ | ✅（从模型输出解析） |
| `usage` | 音频秒数 | token 计数（本地无音频秒数） |
| 强制 `language` | ✅ | ❌ 忽略，模型自动检测 |
| 下游 `model` 字段 | ✅ 透传 | ❌ 忽略，恒用 `VLLM_MODEL` |
| `prompt`（热词/上下文） | ✅ | ✅ 作为 system message |

**故意失败而非静默降级**：本地后端收到 `response_format=srt|vtt` 会返回 400
并说明原因，而不是回一个空字幕文件。

### Qwen3-ASR 输出解析（唯一需要你现场确认的点）

vLLM 返回的 `choices[0].message.content` 是 Qwen3-ASR 的**原始输出**，其中可能带
语种标记。官方解析函数是 `qwen_asr.parse_asr_output(content) -> (language, text)`。
本代理按 `QWEN_OUTPUT_PARSER` 处理：

- `auto`（默认）：能 import 到官方 `qwen_asr` 就用它；否则走**保守启发式**
  （只剥离开头的 `<|…|>` 特殊标记、或显式的 `language: X` 前缀）；再否则原样返回
- `official`：强制用官方解析器（代理容器里没装 `qwen-asr` 就直接报错）
- `none`：完全原样返回

启发式**故意做得很钝** —— 猜错会静默污染转写文本，所以拿不准就不动。
**部署后请先用一条真实录音看 `response_format=verbose_json` 的 `text` 与 `language`
是否干净**；若发现残留标记，把 `QWEN_OUTPUT_PARSER` 定为 `official` 并在代理镜像里
装上 `qwen-asr`。

### 与本地 ASR 服务的编排

配套的 `qwen3-asr-serving` 交付物里 `compose.yaml` 已把两者编排好：`asr`（GPU 推理）
+ `proxy`（本代理）同网络，proxy 的 `VLLM_BASE_URL=http://asr:8000/v1`，
**对外只暴露 proxy**。

## API

### `POST /audio/transcriptions`

兼容 OpenAI `/audio/transcriptions`。**同时挂载了 `POST /v1/audio/transcriptions` 别名** ——
OpenAI SDK 默认带 `/v1` 前缀，而部分下游（如 Open WebUI）填裸 base URL，两种都能用。

支持：

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
