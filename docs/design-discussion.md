# worldfn：以函式簽名宣告所需世界的 Rust runtime

> **Function signature declares the world it needs.**
>
> 一個普通 Rust function，透過參數型別宣告它需要的 dependency、context、tool、capability 與 effect；runtime 負責準備、驗證與執行。

- 專案：`bistin/worldfn`
- 文件日期：2026-09-27
- 狀態：設計討論整理與 prototype implementation brief，尚非已實作 API 規格
- 原始討論：[說服我用 Lisp](chatgpt-conversation://6ab7e6f0-c534-83ee-a89f-567e07628514)
- 第一個落地領域：AI agents；長期定位：**A typed function runtime for Rust**

本文整理原對話從 Lisp、Rust metaprogramming、Bevy，到 typed function runtime 與 AgentWorld 的設計脈絡。原對話中的程式碼大多是概念示意，沒有經過編譯驗證。本文將「已收斂的方向」、「建議的 MVP 決策」與「未來探索」分開；補上的工程細節是為了讓實作可執行，不代表原討論已決定所有細節，也不代表已檢查 repository 現況。

## 1. 設計的起點：讓 host language 長成 domain language

討論最初從 Lisp 的能力出發：程式不只能使用既有語言，也能建立貼近問題領域的語言。Rust 的 procedural macro 同樣能解析 DSL、生成型別與程式碼，但 Bevy 提供了另一個很有吸引力的路線：**普通函式加上型別與 trait，就能進入新的 programming model。**

```rust
fn movement(
    mut positions: Query<&mut Position>,
    time: Res<Time>,
) {
    // 普通 Rust function body
}
```

這個簽名不只描述傳入哪些值，也宣告對 world 的存取需求。worldfn 想把這個模式用在 agent，再視實際需求抽出可重用的核心。

原討論後來修正了一個關鍵理解：普通 function 轉成 Bevy system，核心是 trait machinery 與不同參數數量的實作，並不要求每個 function 都加上 proc macro attribute。worldfn 應沿用這個方向；macro 可以減少重複，但不應成為使用者進入 runtime 的必要語法。

## 2. 核心理念與目標體驗

```rust
// 目標 API 草案，非已存在的 API。
async fn researcher(
    llm: Llm,
    web: Tool<WebSearch>,
    memory: Context<RelevantMemory>,
) -> Answer {
    // ...
}

let answer = world.run(researcher).await?;
```

runtime 應能從參數型別建立下列宣告：

```text
researcher
├── requires Llm
├── can call WebSearch
└── requires materialized RelevantMemory
```

使用者不應再重複維護一份獨立的 dependency list。函式也不需要自行呼叫 `world.get::<T>()`、解析字串 key，或接收整個 `AgentWorld`。

這裡的「從 signature 得知」是透過編譯期 trait 實作，產生執行期可使用的 metadata；不是 runtime reflection、讀取 Rust 原始碼，也不是分析 function body。

### 已收斂的方向

1. 普通 `async fn` 是主要使用介面。
2. 參數型別是宣告；`AgentParam` 定義如何準備參數及提供 metadata。
3. 執行環境由 world 提供；測試換 world/provider，不改 agent 邏輯。
4. prototype 先驗證核心 trait 與 async 可行性。
5. 第一版使用 fake，不接真實 LLM provider。
6. 先不要求 `#[agent]` 或其他 proc macro。
7. graph、probabilistic transitions、完整 effect scheduler 放在後續階段。

## 3. 為何 agent 特別適合

每個 agent 需要的世界不同，差異不只在輸入資料，也包括它能取得的背景資訊、可呼叫的工具，以及允許造成的改變。

| Agent | Context | Capability / effect |
|---|---|---|
| Researcher | RelevantMemory、Conversation | 呼叫 WebSearch |
| Coding agent | CurrentModule、RelevantDocs | 修改 Repository、執行 Sandbox |
| Reviewer | ChangedFiles、CodingStandards | 讀取 Git、呼叫 TestRunner |
| Approval agent | PurchaseRequest、Policy | 輸出或發送 ApprovalDecision |

```rust
// 遠景示意；Read / Write / Emit 不屬於第一版必要範圍。
async fn coding_agent(
    llm: Llm,
    code: Context<RepoContext<CurrentModule>>,
    docs: Context<RelevantDocs<5>>,
    repo: Write<Repository>,
    shell: Tool<Sandbox>,
) -> Patch {
    // ...
}
```

此模式將幾個原本分散的問題串到同一個宣告入口：dependency injection、context selection、tool provisioning、capability 檢查、testing/mock，以及 workflow scheduling 所需的資訊。

Agent function 仍然是普通 Rust 邏輯：它可以只執行一次 LLM 呼叫，也可以有自己的迴圈與工具使用策略。runtime 不應偷偷替所有函式加上一個未宣告的自主 agent loop。

## 4. Bevy 的啟發與應保留的分層

對照關係如下：

| Bevy | worldfn 草案 | 責任 |
|---|---|---|
| `World` | `AgentWorld` | 保存可供解析的資源與 provider |
| `SystemParam` | `AgentParam` | 參數初始化、狀態、存取宣告與取得值 |
| `SystemParamFunction` | `AgentFunction` | 讓符合參數規則的普通 function 可被呼叫 |
| `IntoSystem` | `IntoAgent` | 將 function 轉成持有狀態的 runtime object |
| `FunctionSystem` | `FunctionAgent` | 保存 function、param state 與 metadata |
| `System` | `Agent`，名稱待定 | runtime 執行介面；第一版可保持很薄 |
| `SystemState` | prepared param state | 不依賴完整 scheduler 也能初始化及取得參數 |

Bevy 的 `SystemParam` 區分持續存在的 `State` 與依 world/state lifetime 取得的 `Item`，並要求正確註冊存取資訊；它的 unsafe 契約與 ECS 的存取控制直接相關。worldfn 要學習責任分層，不應直接複製這套 unsafe 取值方式。[Bevy SystemParam 文件](https://docs.rs/bevy_ecs/0.19.1/bevy_ecs/system/trait.SystemParam.html)

`SystemParamFunction` 表達普通函式如何接受 system parameters；`IntoSystem` 負責轉換；`FunctionSystem` 是承載函式與執行狀態的具體 system。[SystemParamFunction](https://docs.rs/bevy_ecs/0.19.1/bevy_ecs/system/trait.SystemParamFunction.html)、[IntoSystem](https://docs.rs/bevy_ecs/0.19.1/bevy_ecs/system/trait.IntoSystem.html)、[FunctionSystem](https://docs.rs/bevy_ecs/0.19.1/bevy_ecs/system/struct.FunctionSystem.html)

```text
ordinary async fn
       │ 參數與回傳 Future 符合 trait bounds
       ▼
AgentFunction<Marker>
       │ IntoAgent
       ▼
FunctionAgent<F, Marker>
       ├── function
       ├── parameter state
       └── AgentMeta
       │ initialize → resolve → invoke → await
       ▼
typed output
```

Bevy 的 parameter state 也能透過 `SystemState` 在 scheduler 外使用。這支持 worldfn 先做 `run()` 與 prepared execution，再考慮 scheduler 的分層方式。[SystemState 文件](https://docs.rs/bevy_ecs/0.19.1/bevy_ecs/system/struct.SystemState.html)

`bevy_ecs` 已提供獨立於完整遊戲引擎的 ECS 功能，但上述型別仍以 ECS World 為基礎。worldfn 的目標是探索面向 agent 的 async、context 與 capability 語意；目前不要求依賴 Bevy，也不宣稱已證實市場上沒有類似抽象。

## 5. 這套設計比 dependency lookup 多了什麼

底層用 `TypeId → value` 做 prototype 完全合理。差異在於上層是否保留以下契約：

```text
parameter type
   ├── 如何取得值
   ├── 如何初始化及重用 state
   ├── 宣告需要的資源／能力
   └── 失敗時指出缺少什麼
```

若最後只有 `HashMap<TypeId, Box<dyn Any>>` 和一組 `get()`，而 function 自己到 container 找資料，就沒有驗證到最重要的設計。第一版至少應有由參數產生的 dependency metadata，以及不必手動注入的 ordinary async function。

但這也不是 Rust 語言層級的完整 effect system：runtime 無法只憑 signature 知道任意 Rust body 裡發生的所有 I/O。

## 6. Dependency、context、tool、capability、effect 的區別

| 概念 | 回答的問題 | 例子 | 備註 |
|---|---|---|---|
| Dependency | 執行需要什麼服務或資源？ | `Llm`、`Service<Clock>` | 可是長生命週期 handle |
| Context | 這次執行需要看見什麼資訊？ | `Context<RelevantMemory>` | 通常與 task、權限、版本有關 |
| Tool | 可以呼叫什麼具型別的操作？ | `Tool<WebSearch>` | 有 request、response、error contract |
| Capability | 此次執行獲准做什麼？ | `ReadTool<Git>`、`Require<Approval>` | 需要 runtime/provider 實際執行限制 |
| Effect | 操作可能造成什麼外部變化？ | `Write<Repo>`、`Emit<Audit>` | 可供政策、排程及追蹤使用 |
| Input / output | 工作之間傳遞什麼？ | `Input<Plan>` → `Patch` | 不應和長期 shared resource 混淆 |

同一參數可以同時具有多個角色。例如 `Tool<WebSearch>` 既是 dependency，也是「可呼叫搜尋」的 capability，呼叫時還可能消耗額度或產生外部 I/O。

`Read<T>` 也不等於「沒有副作用」：若 `T` 暴露 interior mutability 或能寫入遠端服務的方法，單靠 shared reference 不會限制這些操作。需要真正受限的 facade、provider contract 或執行環境。

## 7. 核心架構

### 7.1 AgentWorld

建議責任：

- 註冊 logical dependency 與對應 provider/value。
- 驗證參數是否可以被滿足。
- 提供初始化、prepared execution 與一次性 `run()` 的入口。
- 將 runtime 層錯誤與 function 回傳值分開。
- 後續承載 context resolver、policy 與 tracing hooks。

MVP 不必一開始就有 entity、component、完整 scheduler 或 dependency constructor graph。先採用型別索引的安全容器；每個 key 一個明確 binding，避免隱性覆蓋造成難查的測試結果。

### 7.2 AgentParam

`AgentParam` 定義「參數需要什麼」及「如何取得」。長期模型可分成：

```text
describe → initialize → resolve for invocation → cleanup if needed
```

**建議 MVP 簡化：**參數是 owned value 或可複製的 handle；已 materialize 的 context 也以 owned snapshot/Arc 傳入。先用同步解析取得 handle，然後非同步執行 function。這不會假裝已實現 task-aware retrieval，但可先驗證 async function machinery。

以下為介面方向，省略錯誤型別及完整 bounds；不是保證可直接編譯的最終 trait：

```rust
trait AgentParam: Sized {
    type State;

    fn describe(meta: &mut AgentMeta);
    fn init(world: &AgentWorld) -> Result<Self::State, InitError>;
    fn resolve(
        state: &mut Self::State,
        world: &AgentWorld,
    ) -> Result<Self, ResolveError>;
}
```

後續若需要 async context retrieval，再讓 resolve 回傳 future；如果需要借用 world，才加入 `Item<'world, 'state>` 等 GAT。不要在第一天同時追求 borrowed params、異質 graph、dyn dispatch 與 async materialization。

### 7.3 AgentFunction

負責將已解析的 tuple 參數展開成普通 function call，保留 function 的具體 future 與 output。

```rust
// 概念介面。
trait AgentFunction<Marker> {
    type Params;
    type Output;
    type Fut: Future<Output = Self::Output>;

    fn call(&mut self, params: Self::Params) -> Self::Fut;
}
```

對 owned parameters，核心 bound 可從 `F: FnMut(P1, P2, ...) -> Fut` 與 `Fut: Future<Output = O>` 開始。這是探索 ordinary `async fn` 的直接路線。

`Marker` 用於區分不同 arity/signature 的 impl，處理 generic parameters 約束與 impl coherence；不應要求一般使用者手寫 marker。

### 7.4 IntoAgent

將符合 `AgentFunction` 的 function 包成 `FunctionAgent`。conversion 可只建立未初始化物件；world 相關的 init 交給 prepare/run。這樣 conversion 不必隱含 global state。

### 7.5 FunctionAgent

保存 function、每個參數的 state、metadata，以及必要的 world identity/version。反覆執行時重用穩定資訊，但每次取得當次 invocation 的資料。

不要只用 function 的型別作為所有 closure 實例的快取 key：同型別 closure 可捕捉不同值；function pointer 的型別也可能對應多個不同函式。prepared object 應持有實際 callable instance。

### 7.6 AgentMeta

第一版只需小型而可檢查的 metadata：函式診斷名稱、所需參數、logical resource key、參數種類。後續才加入 read/write、tool schema、input/output port、policy requirements。

`type_name` 適合診斷，不是穩定的儲存格式。需要跨程序或跨版本傳遞時，應另訂顯式名稱與 schema version。

## 8. API 草案與錯誤語意

### 8.1 一次性執行

```rust
// API 草案；provide_llm / provide_tool 的命名可在實作時調整。
let fake_llm = FakeLlm::with_answer("A concise answer");
let fake_web = FakeTool::<WebSearch>::with_response(search_results);

let mut world = AgentWorld::new();
world.provide_llm(fake_llm.clone());
world.provide_tool::<WebSearch>(fake_web.clone());
world.provide_context::<RelevantMemory>(memory_snapshot);

let answer: Answer = world.run(researcher).await?;
```

原討論使用 `.provide(FakeLlm::new())` 等簡潔示意。實作必須明確定義 fake 如何綁到 signature 要求的 logical dependency，不能以為 `TypeId<FakeLlm>` 自動等於 `TypeId<Llm>`。可以保留統一 `.provide(...)`，但需有明確且不衝突的 provider conversion。

### 8.2 可重用的 prepared execution

```rust
// 建議擴充；用於展示 init-time caching。
let mut agent = world.prepare(researcher)?;
let first = agent.run(&world).await?;
let second = agent.run(&world).await?;
```

`world.run(researcher)` 可是方便的一次性 init + run；不要未實作快取卻聲稱反覆呼叫它只 init 一次。效能測試需區分 one-shot 與 prepared 路徑。

MVP 建議 prepared state 綁定 world identity 與 registry generation。更換 binding 後，舊 prepared object 明確失效、要求重新 prepare；或採用不可變 world 的設計。兩者擇一並文件化，避免靜默使用過期 handle。

### 8.3 Runtime error 與 domain error

若函式回傳 `Answer`，runtime 的型別可以是 `Result<Answer, RunError>`。若函式回傳 `Result<Answer, AgentError>`，最單純且不隱藏語意的 MVP 是：

```rust
Result<Result<Answer, AgentError>, RunError>
```

外層代表 init/resolve 失敗，內層代表函式自己的錯誤；使用者可明確寫 `.await??`。未來若要 flatten，另設有明確規則的 adapter，不要使用互相重疊的 blanket impl 偷偷特判所有 `Result`。

缺少參數時，錯誤應接近：

```text
Cannot prepare researcher:
  ✓ Llm
  ✗ Tool<WebSearch>: no provider registered
  ✓ Context<RelevantMemory>
```

不應以 downcast panic、空值或默默跳過執行呈現。

## 9. Context materialization：signature 成為 context engineering

`Read<Memory>` 不應自動意味著把所有 memory 塞進 prompt。**資料來源、挑選策略與已生成 context 是三件不同的事。**

```text
Task / invocation input + identity + policy
                  │
                  ▼
         Context requirement
                  │
        retrieve / select / rank
                  │
     filter / deduplicate / budget
                  │
       materialized typed snapshot
                  │
                  ▼
             agent function
                  │ 顯式或由 adapter 組裝 prompt
                  ▼
                 LLM
```

建議統一以 `Context<Spec>` 表達第一層語意；原對話出現過直接把 `RelevantMemory<10>` 等當參數的寫法，可在未來做為 convenience alias 或自訂 `AgentParam`，不必同時實作兩套概念。

| Context specification | Materialization 行為草案 |
|---|---|
| `RelevantMemory<10>` | 依目前任務檢索，挑選最多 10 筆記憶 |
| `RecentConversation<20>` | 取得最近 20 個符合規則的訊息單位 |
| `RelevantDocs<5>` | 查找相關文件，去重並保留來源 |
| `RepoContext<CurrentModule>` | 依任務定位模組，收集程式碼與必要依賴摘要 |
| `RepoContext<ChangedFiles>` | 依指定 base/head 或工作目錄快照產生變更 context |
| `SystemPrompt<CodeReviewer>` | 提供可版本化的角色指示 |
| `UserProfile` | 取得與當次身份相符且允許使用的個人資訊 |

原討論中的 `RelevantMemory` 可視為預設策略名稱；`RelevantMemory<10>` 則是可選的 const-generic 語法。數量上限不等於 token 上限，context budget 仍需獨立存在。

後續 context resolver 必須定義：

- **輸入範圍：**當次 task 與 identity 從 invocation scope 取得，不以 mutable global「current task」傳遞。
- **來源與新鮮度：**保留文件 ID、repo revision、檢索條件與必要的時間資訊。
- **Budget：**檢索數量、token、時間或費用限制，以及超額時的裁切順序。
- **失敗：**必需 context 缺失時失敗；只有明確標為 optional 才能降級。
- **快取：**依 query、身份／權限、來源版本、策略版本及 budget 建 key；不能只看 Rust 型別。
- **信任層級：**retrieved text 是內容資料，不應自動取得 system instruction 的地位。

函式收到 context 不代表 runtime 已自動把它放入 prompt。MVP 由函式顯式使用 context；統一 Context Builder、prompt template 與 token accounting 屬後續工作。

## 10. Capability 與 effect 的 runtime 意義

```rust
// 長期方向示意。
async fn reviewer(
    llm: Llm,
    code: Context<RepoContext<ChangedFiles>>,
    standards: Context<CodingStandards>,
    git: ReadTool<Git>,
) -> Review {
    // ...
}
```

runtime 能依這份宣告提供受限的 Git facade，而不提供 repository write handle。這比只在 prompt 裡要求「請勿修改」更具體，但保證範圍必須清楚：

1. 宣告描述的是經由 worldfn 提供的 capability。
2. provider/facade 必須真的限制可用操作；不能仍暴露全權底層 client。
3. 任意 native Rust code、closure captures、global handles 或廣泛授權的 shell 仍可能繞過這個模型。
4. 執行不受信任程式碼時，另需 process、OS 或 sandbox 隔離；型別 wrapper 本身不是 sandbox。

對 effect，可從 `READ Repo`、`WRITE Repo`、`CALL WebSearch`、`EMIT Audit` 等 metadata 開始。未來 scheduler 能據此採用保守的衝突規則，但資源的 identity 與 scope 不能被忽略：兩個不同 repository 都叫 `Repo`，不代表一定衝突；兩個不同型別指向同一外部資源，也不代表一定安全。

`Write<Db>` 不會自動建立資料庫 transaction，`Emit<Audit>` 也不會自動保證 audit 已持久化。這些效果需要 adapter 明確實作原子性、提交與失敗語意。

## 11. 測試模型：相同 function，不同 world

測試不是附屬展示，而是第一階段最重要的驗證方式。

### FakeLlm

- 以預設序列回傳 typed response 或 error。
- 記錄收到的 request、context 與呼叫次數。
- 對超出預期的呼叫明確失敗。
- 可模擬失敗與多輪回應；取消和 streaming 可後續再加。

### FakeTool

- 綁定到 `Tool<WebSearch>` 等 logical tool，而不是要求 agent 改收 `FakeWebSearch`。
- 工具定義可含 associated `Request`、`Response`、`Error`；fake 遵循同一 contract。
- 記錄 typed requests，提供可預期的成功或失敗結果。
- concrete fake 可保留，並以 clone handle 註冊，方便執行後斷言。

### 最小測試矩陣

| 驗證 | 預期 |
|---|---|
| 零參數、1 參數、多參數 async function | 回傳原本的 typed output |
| `Llm + Tool<WebSearch> + Context<RelevantMemory>` | 能由 world 解析並完成執行 |
| 不支援的參數型別 | 編譯期不符合 `AgentParam` bound |
| 未註冊 dependency | 明確 runtime/init error，body 尚未執行 |
| FakeLlm / FakeTool 替換 | 不修改 agent signature/body |
| Provider error | 保留其錯誤，不誤報為 missing dependency |
| Metadata | 與 signature 所需依賴一致 |
| Prepared agent 執行兩次 | 穩定 state init 一次；當次值依所定契約取得 |
| World/binding 變更 | 明確失效或遵循已文件化的快取規則 |
| 兩個 test world | 資源與呼叫紀錄彼此隔離 |

測試不應依賴網路、API key、真實 LLM 或不固定的時間。後續 graph/routing 測試再使用固定 RNG seed、fake clock 與錄製好的外部回應。

## 12. Multi-agent typed graph

原始構想是將函式的 output/input 型別接起來：

```rust
// 未來 API 示意；第一版不需實作 Input 或 graph。
async fn planner(task: Input<Task>, llm: Llm) -> Plan { /* ... */ }
async fn coder(plan: Input<Plan>, repo: Write<Repo>, llm: Llm) -> Patch { /* ... */ }
async fn reviewer(patch: Input<Patch>, repo: Read<Repo>, llm: Llm) -> Review { /* ... */ }
```

```text
Task → planner → Plan → coder → Patch → reviewer → Review
                         │                    │
                     WRITE Repo           READ Repo
```

型別能驗證相容的 edge，並支援推導候選連線；但不能單靠 `Plan` 的型別決定「哪個 planner 的哪一份 Plan」應送到哪個 coder。多 producer、版本、條件分支與迴圈都會使自動連線產生歧義。

因此建議先採用**顯式連線、型別驗證**：

```rust
// 未來語法候選。
graph.connect(planner_node.output(), coder_node.input());
graph.connect(coder_node.output(), reviewer_node.input());
```

後續再讓無歧義的情況自動連線。Graph 至少需要 node identity、typed ports、error edges、取消與重試規則。

共享資源的 access graph 與 typed dataflow graph 是兩張相關但不同的圖：前者說明誰可能衝突，後者說明資料從哪裡來。read/write conflict 不能自行決定 producer/consumer 的順序。

Fan-out 要決定 move、clone 或共享 immutable artifact；iteration 要定義版本與終止條件。MVP 不需要 `AgentOutput` trait；只有 graph 真正需要 schema、serialization 或 artifact identity 時再增加限制。

## 13. Probabilistic transitions

後續可把 agent 結果建模成下一步的分布：

```rust
enum NextStep {
    Search,
    Code,
    Clarify,
    Stop,
}

// 未來方向。
async fn planner(/* typed params */) -> Distribution<NextStep> {
    // ...
}
```

```text
Search   0.55
Code     0.30
Clarify  0.10
Stop     0.05
```

責任分層：

```text
agent produces candidate distribution
                 ↓
validate weights and transition targets
                 ↓
policy filters eligible transitions
                 ↓
selection strategy: sample / argmax / explicit rule
                 ↓
materialize next agent's context
                 ↓
run next agent
```

`Distribution` constructor 應檢查非負、有限值、非空／非零總和，並明訂接受 normalized probabilities 還是未正規化 weights。Policy 過濾後若仍採 sampling，需對剩餘權重重正規化；沒有合法下一步時，明確停止或交給預定 fallback。

LLM 提供的數字只能先視為分數或建議權重，不能直接宣稱是經校準的成功機率。Selection policy 與分布產生分開，才能替換策略、測試與追蹤。

Runtime 另需最大步數、時間／token budget、終止條件與循環控制。固定 seed 只能重現給定分布上的抽樣；若 LLM/tool/context 本身不同，整條執行不會因此自動 deterministic。

這個方向將早期「State → probabilistic transition → State」構想，接到「每個 state/agent 宣告不同 context requirements」；它是 graph 階段之後的擴充，不是第一個 prototype 的前置需求。

## 14. 效能模型與 init-time caching

目標是把穩定成本移往 compile/init time，避免每次執行重複發現相同資訊。

| 階段 | 工作 | 注意事項 |
|---|---|---|
| Compile time | Trait selection、arity impl、monomorphization | 可能增加編譯時間與 code size |
| Init / prepare | Dependency lookup、binding validation、metadata 建立 | 對 prepared instance 重用 |
| 每次 invocation | 取得 handle、context retrieval、呼叫 function | 動態 context 不能只 init 一次 |
| Scheduler | 檢查當前資源／policy、決定執行 | 靜態 metadata 可預算，但動態狀態仍需檢查 |

### Dependency storage 的演進

1. **MVP：**`TypeId` map + 安全 downcast；簡單、容易診斷。
2. **Prepared handles：**init 時解析 provider handle，後續避免重複 lookup；需定義替換語意。
3. **Typed slots：**init 時解析 slot，runtime 以 slot 存取；需 world identity、generation 與型別驗證。
4. **靜態 context：**未來若 `Param<C>` 可直接存取 `C` 的欄位，有機會進一步 inline；尚非目前承諾。

快取 slot 不表示 downcast、bounds check 或引用計數都消失；更不能為了「zero cost」省略必要驗證。Context retrieval 的結果快取和 dependency binding 快取也要分開。

### Static dispatch 與 async boxing

核心 function adapter 應優先維持 `F` 與 concrete `Fut`，使編譯器有 monomorphize/inline 的機會。

異質 agent registry 或可替換 provider 若採 `dyn`，可能需要 `Pin<Box<dyn Future<Output = ...> + Send + 'a>>`：它引入 allocation、indirection 與動態 poll dispatch。這些邊界可以是合理取捨，但需明確標示，不可聲稱整個 runtime 完全 static dispatch。

尤其固定寫法 `Llm` 若要在 runtime 換不同 backend，往往需要 type erasure、enum，或某種 facade。`Llm<P>` 可保留 generic provider，但會把 backend 相關型別帶進 signature。**建議讓 function/parameter 核心路徑保持靜態，provider 邊界的動態成本另行評估與記錄。**

不要預先承諾零 overhead，也不要以「LLM 很慢」作為不量測框架成本的理由。後續 benchmark 至少比較 direct call、one-shot run、prepared run、0/1/3 個參數與 boxed boundary；分別報告框架成本及真正 I/O 成本。

## 15. Rust async、lifetime、arity 的技術風險

### 15.1 Async function 與 future 型別

普通 `async fn(P) -> O` 可從 `Fn(P) -> Fut`、`Fut: Future<Output = O>` 的角度包裝。MVP 應先用 owned params 驗證這條路。

若參數借用 world，回傳 future 可能隨 lifetime 改變；單一 `Fut` 型別的 bound 不一定足以表達整個關係。這時可能涉及 higher-ranked bounds、GAT、`AsyncFn` family 或局部 boxing，不能直接將同步 Bevy trait 改成 `async` 就視為完成。

Rust 的 async closure 有自己的借用與 call-trait 規則；某些借用捕捉狀態的 async closure 不能直接符合一般 `FnMut` 回傳 future 的 adapter。第一版只要求 ordinary async functions，closure 支援範圍另以編譯測試決定。[Rust closure reference](https://doc.rust-lang.org/reference/types/closure.html#async-closure-traits)

### 15.2 不跨 await 借用整個 world

長時間持有 world 的 exclusive borrow 會限制並行；跨 await 持有 lock guard 也可能造成阻塞或死鎖。第一版先傳 owned snapshot、Arc handle 或受限 service facade，不提供從 world 直接借出的任意 `&mut T`。

未來的 `Write<T>` 可考慮短 transaction、command buffer 或具體同步策略。不能把它當成一個 `Arc<T>` 標籤後就聲稱已實現互斥。

### 15.3 Arity 與 impl coherence

需為不同參數數量產生 impl；建議第一版以 `macro_rules!` 支援 0–8 個參數，並測試上限。這是本文建議的實作範圍，不是原討論已選定的限制。

Marker 需能約束參數與 output/future 型別，避免 unconstrained type parameter 或 overlapping impl。也應注意 `Param::Item` 的非一對一關係可能讓推導變難；MVP 先以 `Self` 作為解析結果，減少這類複雜度。

### 15.4 Send、Sync 與 'static

先說清楚 runtime 是否會 spawn 到多執行緒 executor。只在當前 task `.await`，和交給 `spawn` 的需求不同；不必無條件把所有 output/context 都限制成 `'static`。

若第一版為簡化 provider storage 而要求 owned、`Send + Sync + 'static` 資源，要把它記為 API 限制，而不是宣稱這是所有 async 函式的必要條件。

### 15.5 Dyn compatibility

帶有 async methods、回傳 opaque future 或某些 generic associated types 的 trait，不能不經設計就當作 `dyn Trait` 使用。若異質容器需要 erased interface，應另設 adapter，而不是把核心 trait 直接改成處處 boxed。[Rust trait dyn compatibility](https://doc.rust-lang.org/reference/items/traits.html#dyn-compatibility)

### 15.6 其他必須決定的邊界

- `FnMut` 是否意味同一 `FunctionAgent` 一次只允許一個 in-flight run？MVP 建議是。
- 重複 `Read<T>` 可否共享？重複 `Write<T>` 如何拒絕？寫入語意實作前不做空泛保證。
- 解析到一半失敗，已產生的值或 context 如何釋放？MVP 依 RAII，resolver 後續需考慮取消安全。
- Future 被 drop 不等於外部副作用回滾；retry 需由 domain/provider 定義 idempotency。
- 動態註冊的 dependency 缺失一般在 init/runtime 發現，不能宣稱全部是 compile-time DI 驗證。

## 16. 未來擴充：同一個核心，不同 execution context

Agent 是起點，`worldfn` 這個名字保留了其他領域的可能性。

| 領域 | 可能的參數 | 可提供的價值 | 額外工作 |
|---|---|---|---|
| Web/backend | `Service<Users>`、`Authenticated<User>`、`Input<CreateUser>` | Handler dependencies、測試、policy metadata | HTTP extraction、response/error mapping、request scope |
| Background jobs | `Job<ImageUploaded>`、`Resource<Gpu>`、`Retry<3>` | Worker placement、資源需求、重試描述 | Queue、lease、idempotency、durability |
| Data pipeline | `Input<RawData>`、`Input<Config>` → `NormalizedData` | Typed edges、版本與快取基礎 | Artifact identity、explicit DAG、增量失效 |
| Test framework | `Fresh<Db>`、`FixedClock`、`Fixture<User>` | 隔離環境、fixture、可重現測試 | Setup/teardown 與失敗清理 |
| CLI | `GitRepo`、`Config<Deploy>`、`Service<Docker>` | 命令依賴宣告、工具整合 | CLI args、secret scope、terminal output |
| ML/GPU | `Model<T>`、`Tensor<T>`、`Exclusive<Gpu>` | Device/記憶體需求與排程 | Device lifetime、placement、resource accounting |
| ETL/DB | `ReadTable<T>`、`WriteTable<T>`、`Transaction` | 存取需求與 audit | 真正的 transaction boundary 與資料庫隔離 |
| Plugins/tools | `Capability<ReadFiles>`、`Arg<Query>` | 限定可用介面、schema 與測試替換 | Sandbox、版本、serialization、授權執行 |

Web 領域可借鏡 Axum extractor：參數會從 request/state 產生。但 HTTP body ownership、extractor 順序等是 web adapter 的專屬問題，不能直接等同一般的 World dependency resolution。[Axum extract 文件](https://docs.rs/axum/latest/axum/extract/index.html)

Tool schema 也不會只靠任意 Rust function type 完整生成：參數名稱、說明、JSON schema、serializable input/output，仍需額外 trait 或明確 metadata；`Arg<T>` 與注入的服務必須分開。

未來可能的套件布局：

```text
worldfn-core    World / Param / Function / metadata
worldfn-agent   Llm / Tool / Context / agent adapters
worldfn-test    fake providers / fixtures
worldfn-web     web adapters（有具體需求再做）
```

第一版建議單一 crate，用 module 分隔即可。等至少兩個實際領域驗證共通部分，再決定抽出 `Param<C>` 等泛化介面。

## 17. MVP milestones 與驗收條件

### M0：編譯可行性 spike

目的：最早揭露 trait inference、async future 與 arity 的問題。

- 以一個 ordinary async function 和 0/1/3 個 owned parameters 跑通。
- 驗證 `FnMut(...) -> Fut`、marker 與 tuple expansion。
- 記錄 stable toolchain/MSRV 候選及限制。
- 不接 SDK、不建 graph、不使用 proc macro 掩蓋推導問題。

驗收：使用者端能接近 `world.run(researcher).await`，且不必手寫 agent struct 或 function-specific adapter。

### M1：最小 typed agent runtime

- `AgentWorld`、`AgentParam`、`AgentFunction`、`IntoAgent`、`FunctionAgent`。
- `Llm`、`Tool<T>`、`Context<T>` 與明確 fake binding。
- 安全的 typed dependency resolution。
- 最小 `AgentMeta` 與 missing-dependency diagnostics。
- 普通 async function、typed output、domain/runtime error 區分。
- FakeLlm、FakeTool、已 materialize context。
- 測試隔離、不同 arity、可重用 param state；prepared API 可採最小形式。
- 一個 researcher example 與說明技術取捨的 README。

驗收：不需 API key，測試即可展示同一 researcher 以不同 fake world 執行；參數清單只在 signature 宣告一次。

原討論提到約 300–500 行核心 prototype，應理解為控制規模的意圖，不是包含測試、文件與錯誤處理的硬性行數限制。

### M2：Context 與受限 capability

- Invocation scope / `Input<Task>`。
- 非同步 materialization、token budget、來源與快取規則。
- Read/Write/Emit 等 facade 的實際語意。
- Policy hook 與 tracing；必要時接一個真實 provider。

驗收：兩種不同任務得到不同的 relevant context；未授權操作在 provider/runtime 層被拒絕，而不僅存在 metadata 標籤。

### M3：Typed graph

- 顯式 typed edges、node identity、error paths。
- 明確的 sequential/fan-out/fan-in 行為。
- 在 access 與 dataflow 規則都確定後才做並行。

驗收：Planner → Coder → Reviewer 可測試；不相容的連線被檢查，多 producer 不會默默選錯來源。

### M4：Probabilistic transitions

- `Distribution<NextStep>`、policy 與選擇策略分離。
- 權重驗證、終止與 budget、可追蹤路由決策。
- 固定測試輸入與 RNG 的可重現測試。

驗收：非法 transition 不被抽中；過濾後零可用選項有明確行為；迴圈有上限。

### M5：泛化與效能

- 根據量測改善 lookup、caching、allocation。
- 由第二個實際使用領域決定是否抽 core crate。
- 再評估 web/job/data pipeline adapters，避免預先打造通用平台。

## 18. 非目標

第一版不包含：

- 完整 agent framework、prompt editor 或低程式碼 workflow UI。
- 真實 LLM SDK、provider marketplace、streaming protocol。
- 完整 ECS、entity/component/query 或替代 Bevy scheduler。
- 任意 Rust 的靜態 effect 分析與完備 sandbox。
- 從型別自動推導所有 workflow 意圖。
- Probabilistic planning、distributed execution、durable recovery。
- 自動 transaction、跨服務原子性、exactly-once effects。
- 對所有 lifetime、borrowed async closure 或無限 arity 的支援。
- 每個 function 必須使用 attribute/proc macro。
- 未經量測的 zero-overhead 承諾。

這些限制讓第一個 prototype 能回答最重要的問題：**signature-driven parameter model 是否能在 Rust async 情境中保有自然的 API、清楚的語意與可測試性？**

## 19. 待決策清單

| 問題 | 本文建議的起點 | 改變決策的依據 |
|---|---|---|
| World storage | TypeId map + safe downcast | 效能量測或 binding 需求 |
| Param ownership | Owned / Arc handles | 實際出現不可接受的複製成本 |
| Context resolution | M1 已生成 snapshot；M2 async resolver | M1 trait 路徑穩定 |
| Llm backend | Fake facade，明確標示 erasure | 真實 provider adapter 的需求 |
| Function dispatch | Generic F + concrete Fut | 異質 registry 的實際需求 |
| 重複執行 | Prepared object 保存 state | API 人體工學與量測 |
| Binding replacement | 舊 prepared state 明確失效 | 是否需要 hot swap |
| Error flattening | 保留 runtime/domain 兩層 | 有不衝突且更自然的 API |
| Arity | 建議 0–8 | 實際使用與 compile time |
| Scheduler | M1 不做 | Graph、access semantics 就緒 |
| Crate 分拆 | 單 crate、多 modules | 第二個領域的實際使用 |

## 20. 給 Codex 的 implementation brief

以下內容可獨立複製成實作任務；本文件本身只整理設計，不代表已執行 repository 修改。

```text
Repository: bistin/worldfn

Objective
Build a minimal Bevy-inspired typed function runtime for AI agents.
The central design constraint is:

    Function signature declares the world it needs.

Users should write ordinary Rust async functions whose parameter types
declare dependencies, tools, and context. The runtime resolves these
parameters and invokes the function without a per-function adapter
or a required proc-macro attribute.

Target experience

    async fn researcher(
        llm: Llm,
        web: Tool<WebSearch>,
        memory: Context<RelevantMemory>,
    ) -> Answer

    world.run(researcher).await

Start by inspecting the existing repository and its instructions.
Do not assume the repository is empty or overwrite existing work.

Study the responsibilities of Bevy's SystemParam,
SystemParamFunction, IntoSystem, FunctionSystem, and SystemState.
Reuse the architectural ideas, not ECS-specific unsafe internals.

Milestone 1 scope
- AgentWorld
- AgentParam with an initialization/state boundary
- AgentFunction for ordinary async functions
- IntoAgent and FunctionAgent
- Typed dependency resolution with clear missing-dependency errors
- Minimal parameter-derived dependency metadata
- Llm, Tool<T>, and Context<T>
- FakeLlm and FakeTool with recorded calls and deterministic responses
- Tests and a small researcher example
- A minimal reusable prepared execution path, or an equally explicit
  way to demonstrate that parameter initialization state is retained

Suggested implementation constraints
- Prefer safe Rust and a small crate.
- Start with owned parameters or Arc-backed handles.
- A TypeId map with safe downcasts is acceptable for the prototype.
- Start with FnMut(P1, ...) -> Fut and concrete Future types.
- Use marker types and macro_rules! for arity implementations.
- Suggested initial arity range: 0 through 8.
- First verify 0/1/3-parameter ordinary async functions compile.
- Keep function dispatch generic; document any dynamic dispatch or
  future boxing introduced at provider boundaries.
- Bind fake implementations to logical requirements explicitly:
  FakeLlm is not automatically the same TypeId as Llm, and
  FakeTool<WebSearch> must satisfy Tool<WebSearch> through a defined API.
- M1 Context<T> may hold an already materialized snapshot.
  Do not claim task-aware retrieval is implemented if it is not.
- Separate runtime resolution errors from function/domain errors.
- Define duplicate binding and prepared-state invalidation behavior.
- Do not add a full AgentOutput abstraction unless M1 requires it.

Out of scope
- Real LLM SDK/provider integration
- Proc macros unless a concrete limitation is demonstrated
- Graph scheduling or automatic graph inference
- Probabilistic transitions
- Full Read/Write effect scheduling or arbitrary borrowed world access
- Distributed execution, durable workflows, or sandbox guarantees
- Premature multi-crate decomposition

Required validation
1. The target researcher runs using fakes with no network/API key.
2. Dependencies are declared in the function signature only.
3. Missing dependencies fail before the function body executes.
4. Fake substitution does not change the agent signature or body.
5. Typed outputs and domain errors are preserved.
6. Zero, one, multiple, and maximum-supported parameter counts work.
7. Metadata matches the declared parameters.
8. Repeated prepared runs retain initialization state correctly.
9. Different worlds are isolated; stale/cross-world prepared state
   cannot silently use the wrong bindings.
10. Tests, formatting, and applicable repository checks pass.

Deliverables
- Minimal working implementation
- Focused tests and runnable fake-only example
- README with usage and explicit limitations
- A short design note covering async/lifetime/arity constraints,
  ownership, error layering, caching, dispatch, and boxing boundaries
- A concise change summary suitable for review

Keep the prototype small enough to review in one sitting.
The original 300–500-line idea was a size aspiration for the core,
not a reason to omit correctness, diagnostics, tests, or documentation.

Primary review question
Does this preserve the Bevy-inspired parameter model—initialization,
parameter resolution, metadata, and ordinary function adaptation—or
does it merely expose a container that agents must query manually?
```

## 21. 設計 review 的判準

Review 第一個實作時，優先檢查下列問題：

1. 使用者是否仍在寫普通 async function？
2. Param 是否同時定義值的取得與 runtime 可觀察的需求？
3. World、function adapter、parameter state 的責任是否分開？
4. Fake 是否驗證同一套 public contract，而不是另一條測試專用路徑？
5. 是否清楚區分已完成的 dependency/context 注入與尚未完成的 capability enforcement？
6. 哪些工作只做一次，哪些每次 invocation 必須重做？
7. Async/boxing/lifetime 的妥協是否被說清楚，且沒有污染一般使用方式？
8. 是否留下 context、graph 與 probabilistic policy 的擴充空間，同時維持 M1 規模？

## 22. 參考來源與整理註記

- 設計脈絡以原始對話「說服我用 Lisp」為主；repo 權限排錯不屬於 runtime 設計，因此未納入正文。
- 本次核對的 Bevy 文件顯示版本為 0.19.1，本文使用固定版本連結；實作時以實際選用版本的 source 為準。
- [Bevy SystemParam](https://docs.rs/bevy_ecs/0.19.1/bevy_ecs/system/trait.SystemParam.html)
- [Bevy SystemParamFunction](https://docs.rs/bevy_ecs/0.19.1/bevy_ecs/system/trait.SystemParamFunction.html)
- [Bevy IntoSystem](https://docs.rs/bevy_ecs/0.19.1/bevy_ecs/system/trait.IntoSystem.html)
- [Bevy FunctionSystem](https://docs.rs/bevy_ecs/0.19.1/bevy_ecs/system/struct.FunctionSystem.html)
- [Bevy SystemState](https://docs.rs/bevy_ecs/0.19.1/bevy_ecs/system/struct.SystemState.html)
- [Rust Reference：Closure types](https://doc.rust-lang.org/reference/types/closure.html)
- [Rust Reference：Dyn compatibility](https://doc.rust-lang.org/reference/items/traits.html#dyn-compatibility)
- [Axum extractors](https://docs.rs/axum/latest/axum/extract/index.html)

文中的 worldfn API、效能策略與 roadmap 是設計草案；沒有以這些參考文件當作 worldfn 已實作或已完成 benchmark 的證據。
