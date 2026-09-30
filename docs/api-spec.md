# Mentor App 後端 API 規格(v0.1 draft)

> 狀態:Draft,供 App 與後端平行開發對齊
> 第一個領域:投資 Mentor(沿用 `investment-mentor-skill-mvp-spec` 的產品契約與資料契約)
> 實作基礎:axum + worldfn(`worldfn-axum`)
> 設計目標:核心 API 跨領域共用;領域差異只在 strategy、decision 內容與 entity key。

本文件定義「App 看得到的後端契約」。內部如何以 worldfn 實作,見第 10 節。

---

## 0. 設計原則

1. **不是每個 API 都經過 LLM。** 建立、查詢、確認、刪除都是確定性的一般 API。只有「引導、複盤、比較、對話」會執行 agent,並以 SSE 串流回傳。
2. **數值由程式計算,agent 只負責解釋。** API 回傳的金融數值一律來自程式計算或 provider,並附上來源與時間。
3. **歷史不可改寫。** confirmed 的 decision 是 snapshot,不能修改;要改變看法,就新增 review 或新 decision。
4. **使用者確認才生效。** agent 抽取的內容一律先是 `draft`,要由使用者經 API 明確確認。
5. **資料只屬於登入的帳號。** 所有資源都隱含在目前帳號之下,API 不接受 account id 參數。
6. **不給買賣指令。** 伺服器端會檢查輸出;違反 policy 的內容不會送到 App(第 8 節)。

## 1. 通用約定

| 項目 | 約定 |
|---|---|
| Base URL | `https://api.<domain>/v1` |
| 格式 | Request/response 為 `application/json; charset=utf-8`;agent 端點回應 `text/event-stream` |
| 驗證 | `Authorization: Bearer <JWT>`;`sub` claim 即 account id |
| ID | 帶前綴的不透明字串:`dec_`、`rev_`、`wat_`、`ses_`、`msg_`、`run_`、`mem_`、`dev_` |
| 時間 | RFC 3339,**必須帶時區偏移**,例如 `2026-07-29T20:00:00+08:00` |
| 金融商品 | `MARKET:SYMBOL`,例如 `TWSE:6239`;不接受只有 ticker |
| 分頁 | `?limit=`(預設 20,最大 100)+ `?cursor=`;回應帶 `next_cursor`(沒有下一頁時為 `null`) |
| 冪等 | 所有 `POST` 接受 `Idempotency-Key` header;24 小時內同一個 key 回傳同一個結果 |
| 版本 | 路徑帶主版本 `/v1`;新增欄位不算破壞性變更,App 必須忽略不認得的欄位 |
| Request ID | 每個回應都帶 `X-Request-Id`,回報問題時附上 |

### 1.1 Session 與 Principal

- `account`:來自 JWT。
- `session`:代表一段對話,由 `POST /v1/sessions` 建立。agent 端點用路徑中的 session id。
- 後端把兩者組成 worldfn 的 `Principal`。對話歷史以 `(account, session)` 作為 key,所以別的帳號拿到 session id 也讀不到內容。

## 2. 錯誤模型

非 2xx 回應統一使用:

```json
{
  "error": {
    "code": "invalid_input",
    "message": "invalidation 至少需要一項",
    "details": [{ "field": "invalidation", "issue": "required" }],
    "request_id": "req_..."
  }
}
```

| HTTP | `code` | 意義 |
|---|---|---|
| 400 | `invalid_input` | 格式或 schema 錯誤 |
| 401 | `unauthenticated` | 缺少 JWT 或 JWT 無效 |
| 403 | `forbidden` | 已登入但沒有權限 |
| 404 | `not_found` | 不存在,或不屬於目前帳號(兩者不區分,避免洩漏資源是否存在) |
| 409 | `conflict` | 狀態衝突,例如刪除仍有 active Watch 的 decision |
| 422 | `strategy_requirements_unmet` | 缺少 strategy 規定的欄位;`details` 列出缺哪些 |
| 429 | `quota_exceeded` / `rate_limited` | 附 `Retry-After` |
| 502 | `upstream_unavailable` | LLM、行情或檢索後端失敗 |
| 500 | `internal` | 伺服器設定或程式錯誤 |

agent 端點**開始串流之後**,錯誤改以 SSE 的 `error` 事件送出(第 3 節),HTTP status 維持 200。

## 3. Agent 端點的串流協定(SSE)

agent 端點一律是 `POST`,回應為 `text/event-stream`。瀏覽器原生的 `EventSource` 只支援 GET,所以 App 與 Web 請用 fetch 的串流讀取(或各平台的 SSE client)來解析。

每個事件都是 `event:` 加上 JSON 格式的 `data:`:

| event | data | 說明 |
|---|---|---|
| `run.started` | `{ "run_id": "run_..." }` | 一定是第一個事件 |
| `status` | `{ "stage": "retrieving", "message": "找到 2 筆相關紀錄" }` | 進度,給 UI 顯示用 |
| `context` | `{ "strategy": "fundamental-thesis@1.0.0", "skills": ["..."], "recalled": 3, "sources": [...] }` | 這次引用了哪些資料,讓使用者可以查證 |
| `token` | `{ "delta": "..." }` | 逐字輸出。runtime 以 `Llm::chat_streaming` / `LoopEvent::Text` 提供 |
| `tool.call` / `tool.result` | `{ "tool": "growth_rate", ... }` | 程式計算的呼叫與結果。runtime 以 `LoopEvent::ToolCall` / `ToolResult` 提供 |
| `policy` | `{ "rule": "no_trade_instruction", "action": "redirected" }` | 輸出被 policy 攔下並改為引導 |
| `result` | 端點各自定義的具型別結果 | 最終結果,只送一次 |
| `error` | `{ "code": "...", "message": "..." }` | 串流中的錯誤,代碼同第 2 節 |
| `done` | `{}` | 一定是最後一個事件 |

- `token` 是模型生成當下就送出,**還沒經過 policy 檢查**。需要檢查的端點有兩種做法:(1) 不送 `token`,檢查完才送 `result`;(2) 照送 `token`,若最後被攔下就送 `policy`,前端必須用 `result` 取代已顯示的文字。投資建議類端點預設用 (1)。
- 連線中斷時 agent 仍會跑完,結果可以用 `GET /v1/runs/{run_id}` 取得。v0.1 不支援 `Last-Event-ID` 續傳。
- 心跳:伺服器定期送出 SSE 註解行,避免 proxy 因連線閒置而斷線。

## 4. 核心資源(跨領域共用)

### 4.1 帳號

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/v1/me` | 帳號資訊、方案、用量 |
| GET | `/v1/me/export` | 匯出全部資料(202 + 下載連結,非同步) |
| DELETE | `/v1/me` | 刪除帳號與全部資料,包括 cache(202,非同步)。完成前帳號會被凍結 |

### 4.2 Session(對話)

| 方法 | 路徑 | LLM | 說明 |
|---|---|---|---|
| POST | `/v1/sessions` | 否 | 建立對話;可指定 `strategy`(本次暫時使用,不改變預設) |
| GET | `/v1/sessions` | 否 | 列出對話 |
| GET | `/v1/sessions/{id}/messages` | 否 | 對話歷史 |
| POST | `/v1/sessions/{id}/messages` | **是** | 送出訊息,SSE 回覆。`result` 為 `{ message, suggestions: { decisions: [draft], watches: [draft] } }` |
| DELETE | `/v1/sessions/{id}` | 否 | 刪除對話 |

agent 建議建立的 decision 或 Watch,只會出現在 `suggestions`,狀態是 `draft`。要由 App 呼叫第 4.4、4.6 節的 API 確認後才會生效。

### 4.3 Memory(長期記憶)

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/v1/memory?entity=TWSE:6239&include=superseded` | 依 entity 查詢;預設只回傳目前有效的 |
| POST | `/v1/memory` | 使用者主動新增 |
| POST | `/v1/memory/{id}/supersede` | 以新內容取代舊內容(舊的保留作為歷史) |
| POST | `/v1/memory/{id}/confirm` | 確認 agent 抽取的 draft |
| DELETE | `/v1/memory/{id}` | 刪除 |

```json
{
  "id": "mem_...",
  "entity": "TWSE:6239",
  "text": "我偏好月營收連三個月成長才進場",
  "source": "user_said",
  "status": "confirmed",
  "valid_from": "2026-07-20T14:30:00+08:00",
  "superseded_by": null,
  "source_ref": { "session": "ses_...", "message": "msg_..." }
}
```

`source` 的值:`user_said`、`agent_extracted`(一律先是 draft)、`computed`。

> supersede 的完整語意(有效期間、衝突偵測)待記憶設計定案;本節的欄位是暫定的最小集合。

### 4.4 Decision

| 方法 | 路徑 | LLM | 說明 |
|---|---|---|---|
| POST | `/v1/decisions` | 否 | 建立;依 strategy 驗證必填欄位,缺少時回 422 |
| GET | `/v1/decisions?entity=&strategy=&status=` | 否 | 列表 |
| GET | `/v1/decisions/{id}` | 否 | 單筆,包含 strategy snapshot |
| PATCH | `/v1/decisions/{id}` | 否 | **只能修改 draft**;confirmed 之後回 409 |
| POST | `/v1/decisions/{id}/confirm` | 否 | 確認,並凍結 strategy snapshot |
| DELETE | `/v1/decisions/{id}` | 否 | 仍有 active Watch 時回 409 |

內容沿用 MVP spec 6.3 的 Decision Contract。`entity` 在投資領域就是 `instrument.id`。

### 4.5 Review 與 Compare

| 方法 | 路徑 | LLM | 說明 |
|---|---|---|---|
| POST | `/v1/decisions/{id}/reviews` | **是** | 複盤。body 可帶 `observations`、`lens`(另一個 strategy → `alternate_lens: true`) |
| GET | `/v1/decisions/{id}/reviews` | 否 | 複盤歷史 |
| POST | `/v1/decisions/{id}/compare` | **是** | `{ "strategies": ["momentum-breakout", "fundamental-thesis"] }` |

- Review 的 `result` 沿用 MVP spec 6.4 的 Review Contract,並且已經存檔(response 帶 `id`)。
- 預設依 decision 自己的 strategy snapshot 複盤,不因預設 strategy 改變而變。
- Compare 的 `result` 固定分成三區:`{ "by_strategy": { "<id>": {...} }, "shared_gaps": [...] }`,**沒有合併的結論欄位**。

### 4.6 Watch

| 方法 | 路徑 | 說明 |
|---|---|---|
| POST | `/v1/watches` | 建立,一律是 `draft`;缺少 `source_quote` 時回 422 |
| POST | `/v1/watches/{id}/confirm` | `draft → active` |
| GET | `/v1/watches?status=&due_before=` | 列表 |
| POST | `/v1/watches/{id}/acknowledge` | `notified → acknowledged` |
| POST | `/v1/watches/{id}/pause` \| `/dismiss` | 狀態變更 |

狀態機與規則沿用 MVP spec 6.5。v0.1 只支援 `kind: "date"`;`price` / `metric` 可以建立 draft,但不會自動檢查。

### 4.7 Strategy

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/v1/strategies` | 可用的 strategy pack 與版本 |
| GET | `/v1/strategies/{id}/versions/{version}` | 必填欄位、欄位 schema、可建立的 Watch 種類 |

App 依照這裡回傳的 schema 產生表單,不必把各 strategy 的規則寫死在 App 裡。

### 4.8 裝置與通知

| 方法 | 路徑 | 說明 |
|---|---|---|
| POST | `/v1/devices` | 註冊推播 token(APNs / FCM) |
| DELETE | `/v1/devices/{id}` | 取消註冊 |
| PATCH | `/v1/me/notification-settings` | `detail: "minimal"`(預設)或 `"full"` |

推播內容預設是 minimal:不含 ticker、價格或筆記內容。App 收到推播後,再透過 API 取得細節。

### 4.9 Run

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/v1/runs/{id}` | agent 執行的狀態(`running`/`succeeded`/`failed`)與最終 `result`,供斷線後補取 |

## 5. 領域擴充點

核心資源不變,每個領域提供以下內容:

| 擴充點 | 投資 | 語言學習(驗證用的第二個領域) |
|---|---|---|
| entity key 格式 | `MARKET:SYMBOL` | `vocab:<lemma>` / `grammar:<id>` |
| strategy pack | `momentum-breakout`、`fundamental-thesis` | 例如 `spaced-review`、`error-pattern` |
| decision 的內容 | thesis、evidence、invalidation | 學習目標、練習計畫 |
| 計算工具 | 成長率、均線、ATR… | 間隔重複排程、錯誤率統計 |
| policy | 不給買賣指令、不預測價格 | 幾乎沒有 |
| 領域專屬 API | `GET /v1/market/{instrument}/quote` | 無 |

### 5.1 投資專屬:行情

`GET /v1/market/{instrument}/quote` 回傳 MVP spec 第 10 節的 `DataEnvelope`,一定帶有 `status`、`asOf`、`provider`。資料是 `delayed`、`partial` 或 `unavailable` 時,後端不會補值,相關的 Watch 也不會命中。

## 6. 用量與成本控制

- agent 端點依帳號方案計算次數與 token;超過時回 429 `quota_exceeded`。
- 回應 header:`X-Quota-Limit`、`X-Quota-Remaining`、`X-Quota-Reset`。
- 一般 API 另有速率限制(`rate_limited`)。

## 7. 隱私與資料處理

- 日誌不記錄使用者輸入、筆記或 LLM 輸出的內容,只記錄 id、狀態與耗時。
- `DELETE /v1/me` 會刪除 Postgres 資料**並清除 cache**;清除無法確認時,刪除工作會重試,不會回報完成。
- 推播與 email 預設不含敏感內容(第 4.8 節)。
- 除非方案條款另有明示,不把使用者資料用於訓練。

## 8. Policy 檢查

後端在把 agent 的輸出送到 App 之前檢查,不只靠 prompt:

1. 直接的買賣、加減碼指令 → 攔下,改為回到檢查框架,並送出 `policy` 事件。
2. 價格預測或報酬保證 → 同上。
3. 未標示時間與來源的市場數字 → 拒絕輸出這段內容。
4. 把 agent 推論記成使用者親口說的 evidence → 在資料層阻擋:`source` 不可以是 `user_said`。

規則清單沿用 MVP spec 11.1。檢查結果會記錄事件,方便稽核誤判。

## 9. v0.1 範圍

| 包含 | 不包含(v0.2 以後) |
|---|---|
| 第 4 節全部核心資源 | `token` 逐字串流、`tool.call` 事件 |
| agent 端點:messages、reviews、compare | price / metric Watch 自動檢查 |
| date Watch + 推播 | SSE 續傳(`Last-Event-ID`) |
| 投資領域兩個 strategy | 第二個領域上線 |
| policy 檢查 1–4 | 使用者自訂 strategy pack |

## 10. 與 worldfn 的對應(後端內部)

| API | 實作 |
|---|---|
| JWT → account + session | `worldfn-axum` 的 extractor → `Principal`(**待做**) |
| `POST /sessions/{id}/messages` | agent:`Input<Message>`、`Context<Conversation<N>>`、`Context<Recall<N, Message>>`、`Context<StrategyPack>`、`SessionLog`、`Emit<MentorEvent>` |
| `POST /decisions/{id}/reviews` | agent:`Input<ReviewRequest>`、`Context<DecisionSnapshot>`、`Context<StrategyPack>`、`Tool<…計算>`、`Emit<MentorEvent>` → 具型別的 `Review` |
| SSE 事件 | `MentorEvent: SseEvent`,經 `worldfn_axum::sse` 送出 |
| 錯誤對應 | `RunError::http_status()` + 本文件第 2 節的 `code` |
| 對話 / 記憶儲存 | `SessionStore`、`AccountMemoryStore` → Postgres(+pgvector),以 `Cached` 放 Redis |
| 帳號刪除 | 兩個 store 的 `forget_account`(會連 cache 一起清,失敗就報錯) |
| strategy pack | `SkillLibrary`(沿用 skill 的 `references/`)或具型別的 `StrategyPack` |
| 一般 CRUD、Watch、推播 | 一般 axum handler 與背景 worker,共用同一套儲存與 `Principal` |

worldfn 待補的部分:JWT extractor、supersede 與以 entity 為 key 的記憶、policy 檢查、tool calling、Postgres / Redis crate、用量計算。

## 11. 待決策

1. 身分服務:自建,還是接 Auth0、Supabase Auth 等現成服務?
2. Session 的生命週期:閒置多久算結束?是否自動摘要進長期記憶?(若要自動摘要,仍須經使用者確認)
3. 方案與額度的具體數字。
4. 推播供應商與 Web Push 是否在 v0.1。
5. `GET /v1/me/export` 的格式(JSON,或是與本機 Skill 相容的 Markdown / JSONL,方便使用者在兩邊之間搬資料)。
