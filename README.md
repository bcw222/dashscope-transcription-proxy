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
| `DASHSCOPE_PROXY_REALTIME_ENABLED` | `true` | 是否提供流式 WebSocket 端点 |
| `DASHSCOPE_PROXY_REALTIME_MODEL` | `qwen3-asr-flash-realtime` | 默认实时模型 |
| `DASHSCOPE_PROXY_REALTIME_WS_URL` | — | 上游 WS 地址；留空则由 `BASE_HTTP_API_URL` 推导（`/api/v1` → `/api-ws/v1/realtime`） |
| `DASHSCOPE_PROXY_REALTIME_SAMPLE_RATE` | `16000` | 上游采样率，`16000` 或 `8000`（电话/听写场景用 8000） |
| `DASHSCOPE_PROXY_REALTIME_INPUT_FORMAT` | `pcm16` | 下游送入的音频格式：`pcm16`/`g711_ulaw`/`g711_alaw` |
| `DASHSCOPE_PROXY_REALTIME_LANGUAGE` | — | 语种提示（`zh`/`en`/`yue`…），留空交给模型判断 |
| `DASHSCOPE_PROXY_REALTIME_VAD` | `true` | `true`=服务端 VAD 自动断句；`false`=Manual 模式（由下游 commit） |
| `DASHSCOPE_PROXY_REALTIME_VAD_THRESHOLD` | `0.0` | VAD 触发阈值 `[-1,1]` |
| `DASHSCOPE_PROXY_REALTIME_SILENCE_DURATION_MS` | `400` | VAD 断句静音时长 `[200,6000]` ms |
| `DASHSCOPE_PROXY_REALTIME_FINISH_TIMEOUT` | `5.0` | 发 `session.finish` 后等 `session.finished` 的上限（秒） |
| `DASHSCOPE_PROXY_REALTIME_USE_ENV_PROXY` | `true` | 上游 WS 是否沿用 `HTTPS_PROXY`/`HTTP_PROXY`；上游在本机/内网或代理不通时设 `false` |
| `DASHSCOPE_PROXY_REALTIME_PROXY_URL` | — | 显式指定上游 WS 代理，优先级高于上一项 |

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

> 不支持的值返回 400；`stream=true` 返回 501 —— 流式转写请走下面的 WebSocket 端点。

### `GET /health`

```json
{"status": "ok", "model": "qwen3-asr-flash", "local_mode": false}
```

### `WS /v1/realtime`（实时流式转写）

把 DashScope 的 **Qwen-ASR-Realtime**（WebSocket）包装成 **OpenAI Realtime 转写方言**，
供需要「边说边出字」的下游使用（如 OpenClaw 的听写/Talk 中继）。同时挂载 `/realtime` 别名。

上游模型：`qwen3-asr-flash-realtime`（可用 `?model=` 或 `session.update` 覆盖）。
认证同 HTTP 端点：**透传** `Authorization: Bearer <dashscope-key>`；
浏览器无法设置 WebSocket 请求头，因此也支持 `?api_key=***` 查询参数。

```bash
# websocat 手测：发送 PCM16 16kHz 单声道 base64 块
websocat "ws://127.0.0.1:8000/v1/realtime?model=qwen3-asr-flash-realtime" \
  -H "Authorization: Bearer ***"
```

#### 事件映射

| 方向 | 下游（OpenAI Realtime 方言） | 上游（DashScope Qwen-ASR-Realtime） |
|------|------------------------------|--------------------------------------|
| → | `session.update` | `session.update`（补 `event_id`，翻译 `turn_detection`/`language`） |
| → | `input_audio_buffer.append` | `input_audio_buffer.append`（G.711 → PCM16 后 base64） |
| → | `input_audio_buffer.commit` | `input_audio_buffer.commit`（Manual 模式） |
| → | `session.finish` / `session.close` / 断开 | `session.finish` |
| ← | `session.created` / `session.updated` | 同名（附本代理的音频契约字段） |
| ← | `input_audio_buffer.speech_started` / `.speech_stopped` / `.committed` | 同名透传 |
| ← | `conversation.item.input_audio_transcription.delta` | `…transcription.text`（`text`+`stash` → 增量 `delta`） |
| ← | `…transcription.completed` / `.failed` | 同名透传 |
| ← | `session.finished` / `error` | 同名透传 |

两个关键行为：

1. **partial 语义重塑**：DashScope 的实时事件给的是「已确认前缀 `text` + 可修订草稿 `stash`」，
   而 OpenAI 客户端期望增量 `delta`。本代理按 `item_id` 跟踪累计前缀，只把新增尾部作为 `delta` 下发；
   同时额外附带 `text`（累计全文）与 `revised`（本轮是否发生回改）两个**非标准字段**，
   便于客户端在模型回改时精确对齐。最终结果以 `completed.transcript` 为准。
2. **`session.finish` 不可省**：DashScope 在未收到该事件就断连时会丢弃 in-progress item，
   最终识别结果不会下发。因此下游**主动断开**时，本代理也会补发 `session.finish`，
   并在 `REALTIME_FINISH_TIMEOUT` 内等待 `session.finished`。

音频格式：下游可送 `pcm16`（默认）或 `g711_ulaw` / `g711_alaw`（别名 `mulaw`/`alaw`/`ulaw`）。
G.711 会在代理内解码为 PCM16 再上行；采样率支持 `16000` 与 `8000`
（电话线路、OpenClaw 听写中继均为 8 kHz µ-law，设 `8000` 即可，DashScope 内部升采样）。
裸二进制帧（无事件封装）也会被当作音频处理。

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

## 下游配置示例（OpenClaw 听写 / Talk 中继）

OpenClaw 的输入框听写走**流式**转写 provider（`registerRealtimeTranscriptionProvider`），
内置只有 Deepgram / ElevenLabs / Mistral / OpenAI / xAI 五家。接本代理需要一个极小的本地插件
把 provider 指向 `ws://your-host:8000/v1/realtime`，并在 `plugins.entries.voice-call.config.streaming`
下声明：

```json5
{
  plugins: {
    entries: {
      "voice-call": {
        config: {
          streaming: {
            provider: "dashscope",
            providers: {
              dashscope: {
                url: "ws://127.0.0.1:8000/v1/realtime",
                model: "qwen3-asr-flash-realtime",
                inputFormat: "g711_ulaw",   // OpenClaw 听写中继送 8 kHz µ-law
                sampleRate: 8000,
                language: "zh",
                // apiKey 留空则回退读环境变量 DASHSCOPE_API_KEY
              },
            },
          },
        },
      },
    },
  },
}
```

> 仅需「语音消息转写」（一次性音频附件）的场景不必用流式端点：
> OpenClaw 的 `tools.media.audio` 批量管线直接指向 `POST /v1/audio/transcriptions` 即可。

## 许可证

本项目基于 [PolyForm Noncommercial License 1.0.0](LICENSE) 发布——
**仅限非商业用途**（个人研究、学习、非营利组织等）。商业使用需另行获得授权，
详见 [LICENSE](LICENSE)。
