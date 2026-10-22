# Background: Generic Fleet Telemetry Schema

Background only, written before this project's scope was set. It describes a
multi-GPU, multi-system, multi-model, multimodal fleet — far wider than one H100
running one model. **Not requirements.** `VISION.md` is the source of intent.

Two things here are still worth keeping when we design our own fields:

- the join unit — a query interval joined to a power timeseries on wall-clock
  timestamps
- the per-system `power_draw` alongside per-GPU `power_draw`, since the GPU is
  only about a third of this box's idle draw

The three "correlation" analyses below that name modalities, model identity, and
token usage are all either out of scope or already covered by v1. What this
project actually turns on is the fourth bullet, "accounting for simultaneous
queries" — and under `VISION.md` that means measuring cost *at* an occupancy
level, not decomposing one batch's joules among its members.

## Data Collection

### Per Query

| Field | Type | Importance | Notes | Example |
|---|---|---|---|---|
| `query_id` | string | Low | Unique identifier per query | `123456-abcdef-123456` |
| `session_id` | string | Medium | Links queries through the same session | `567890-ghijkl-567890` |
| `gpu_id` | string | Critical | Links queries to GPU usage | `654321-fedcba-654321` |
| `query_start` | datetime | Critical | When the query begins execution on device (ms or ns; ns preferred) | `2026-09-30T11:42:13,123456789` |
| `query_end` | datetime | Critical | When the query ends execution on device | `2026-09-30T11:42:13,123456789` |
| `streaming` | boolean | Medium | Whether query returned streaming tokens | `True` |
| `time_to_first_token` | int | Low | Latency to first token generation (ms) | `123` |
| `input_tokens` | int | High | Token count before encoding | `123` |
| `output_tokens` | int | High | Tokens generated | `123` |
| `input_modality` | enum | High | Input type: text, image, audio, video, document | `text`; `text,image`; `text,document,audio` |
| `output_modality` | enum | High | Output type: text, image, audio, video, document | `text` |
| `queue_time` | int | Low | Time spent waiting for GPU access | `123` |
| `model` | string | Critical | Model used for query | `GPT-5.1` |

### Per GPU

| Field | Type | Importance | Notes | Example |
|---|---|---|---|---|
| `gpu_id` | string | Low | Unique identifier per GPU | `654321-fedcba-654321` |
| `gpu_model` | string | High | GPU model number | `A100 80GB` |
| `system_id` | string | Low | Links GPU to installed system | `654321-fedcba-654321` |
| `sample_interval` | string | High | How frequently metrics are sampled (ms or ns; ns preferred) | `10ms` |
| `power_draw` | timeseries | Critical | Power draw of card over time | — |
| `temperature` | timeseries | Medium | Temperature of card over time; identify thermal throttling | — |
| `clock_frequency` | timeseries | Medium | GPU clock frequency over time | — |
| `utilization_compute` | timeseries | Critical | GPU compute utilization over time | — |
| `utilization_memory` | timeseries | Medium | GPU memory utilization over time | — |
| `bandwidth_memory` | timeseries | Medium | GPU memory bandwidth utilization over time | — |

### Per System

| Field | Type | Importance | Notes | Example |
|---|---|---|---|---|
| `system_id` | string | Low | Unique identifier per GPU | `654321-fedcba-654321` |
| `cpu_model` | string | Low | — | — |
| `cpu_count` | int | Low | Number of CPU cores | — |
| `memory_ram` | int | Low | GB of RAM | — |
| `gpu_count` | int | Medium | Number of installed GPUs | — |
| `power_draw` | timeseries | High | Measure system usage to compare against GPU usage | — |

### Per Model

| Field | Type | Importance | Notes | Example |
|---|---|---|---|---|
| `model_id` | string | Low | Model name | `GPT-5.1` |
| `param_size` | int | Medium | Number of parameters in model | `8000000000` |
| `architecture` | string | Medium | Specific model architecture | `LLM`, `VLM`, transformer-based, etc. |
| `precision` | enum | Medium | Whether model is quantized | `FP16`, `BF16`, `W8A8` |
| `context_window` | int | Medium | Maximum context window | `512000` |
| `attention_model` | enum | Medium | Which attention mechanism is used | `MHA`, `GQA`, `MQA` |

## Other Artifacts

Generate the following artifacts to support analysis:

- Baseline GPU power draw from a fresh boot, with no applications started.
- Baseline GPU power draw with model(s) loaded into memory.
- Loading and unloading cost, in power and time, per model.

## Data Analysis

The fields above support the following measurements and correlations:

- Total GPU power usage per query.
- Total system power usage per query.
- Correlation of token usage to power consumption.
- Correlation of modalities to power consumption.
- Correlation of specific models to power consumption.
- Accounting for simultaneous queries.

## Concerns

- Need a method to evict or account for caching, including system prompts and related cached data.