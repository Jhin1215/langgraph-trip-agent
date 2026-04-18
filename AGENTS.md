# 🤖 Xiec Assistant 智能体说明文档（中文）

> 本文档用于说明项目中各个 Agent 的职责边界、输入输出、可调用工具与协作关系。  
> 当前系统基于 **LangChain v1 + LangGraph** 构建，整体采用 **Supervisor + Specialist Agents** 多智能体编排模式。  
> 所有 Agent 均在 `src/deep_agent/graph.py` 中接入父图，并由统一总图负责调度。

---

## 📌 1. 总体说明

当前系统包含以下 6 个核心 Agent：
- `supervisor_agent`
- `flight_booking_agent`
- `hotel_booking_agent`
- `car_rental_booking_agent`
- `trip_booking_agent`
- `research_agent`

系统主流程如下：

```text
START
  -> fetch_user_info_node
  -> supervisor_agent
  -> specialist agent
  -> END
```

其中：

- `fetch_user_info_node`：负责预取用户航班背景信息
- `supervisor_agent`：负责识别任务类型并执行 handoff
- `specialist agents`：负责本领域任务理解、工具调用与结果生成

---

## 🧭 2. 通用约定（Conventions）

所有 Agent 在设计与扩展时统一遵循以下约定：

### 2.1 编排原则

- `supervisor_agent` **只做路由，不直接执行业务工具**
- specialist agent **只处理本领域任务**
- 一次只激活一个 specialist agent
- 复杂业务意图必须通过 handoff 明确传递

### 2.2 状态管理原则

项目统一使用 `TravelState` 管理动态状态，典型字段包括：

- `messages`
- `active_agent`
- `handoff_task_type`
- `handoff_user_request`
- `handoff_slots`
- `user_flight_info`

静态上下文信息（如 `passenger_id`）通过 `runtime.context` 注入，不混入动态任务槽位。

###  2.3 工程实现原则

- 尽量优先采用 **async-friendly / async-native** 风格
- 新增工具应保持**低依赖、远程可运行、安全**
- 项目部署于 Web Server 环境，新增逻辑应尽量避免直接依赖本地文件系统
- 敏感操作应优先考虑接入 `HumanInTheLoopMiddleware`

---

## 👑 3. `supervisor_agent`

###  3.1 角色定位

`supervisor_agent` 是整个多智能体系统的中心调度器。  
它的职责不是处理具体业务，而是识别用户当前任务属于哪个业务域，并将任务结构化后转交给对应 specialist agent。

### 3.2 主要职责

- 识别用户当前问题属于：
  - 航班
  - 酒店
  - 租车
  - 旅行产品
  - 联网搜索
- 调用 handoff tool 完成任务转交
- 将以下任务态写入父图 state：
  - `active_agent`
  - `handoff_task_type`
  - `handoff_user_request`
  - `handoff_slots`

###  3.3 不负责什么

- 不直接调用航班、酒店、租车等业务工具
- 不直接返回编造的业务结果
- 不在路由阶段执行数据库写操作

###  3.4 典型可调用工具

- `transfer_to_flight_booking_agent`
- `transfer_to_hotel_booking_agent`
- `transfer_to_car_rental_booking_agent`
- `transfer_to_trip_booking_agent`
- `transfer_to_research_agent`

### 3.5 核心价值

`supervisor_agent` 解决的是**任务分发与职责解耦**问题，是整个系统由单智能体迁移到多智能体架构的关键入口。

---

## ✈️ 4. `flight_booking_agent`

###  4.1 角色定位

`flight_booking_agent` 是航班域 specialist，负责处理与机票、航班、退票、改签、航班政策相关的全部任务。

###  4.2 主要职责

- 查询当前用户机票信息
- 查询符合条件的航班列表
- 改签现有机票
- 取消 / 退票
- 查询航班相关政策

###  4.3 典型输入

handoff 后常见输入状态包括：

- `handoff_task_type = "cancel_ticket"`
- `handoff_slots = {"ticket_no": "..."}`
- `handoff_task_type = "update_ticket"`
- `handoff_slots = {"ticket_no": "...", ...}`

同时还可以从：

- `runtime.context.passenger_id`
- `state["user_flight_info"]`

中读取用户身份与预取航班背景。

### 4 .4 典型可调用工具

#### 安全类工具

- `fetch_user_flight_information`
- `search_flights`
- `lookup_policy`

#### 敏感类工具

- `update_ticket_to_new_flight`
- `cancel_ticket`

### 4.5 HITL 策略

当前以下工具已接入 `HumanInTheLoopMiddleware`：

- `update_ticket_to_new_flight`
- `cancel_ticket`

执行流程为：

1. 模型先产出 sensitive tool call
2. graph 触发 interrupt
3. 用户输入 `approve / reject`
4. graph 使用同一 `thread_id` 恢复执行

###  4.6 当前完成度

`flight_booking_agent` 是当前项目中完成度最高的 specialist，已经打通：

- 查询当前机票信息
- 多轮记忆追问
- 航班搜索
- 正常改签
- 无效改签
- 取消机票触发 HITL
- `approve / reject` 恢复路径

---

## 🏨 5. `hotel_booking_agent`

###  5.1 角色定位

`hotel_booking_agent` 是酒店域 specialist，负责处理酒店查询、预订、修改订单等任务。

###  5.2 主要职责

- 查询酒店信息
- 查询可预订酒店
- 酒店预订
- 酒店订单修改
- 酒店订单取消

### 5.3 典型可调用工具

- `search_hotels`
- `book_hotel`
- `update_hotel`
- `cancel_hotel`

###  5.4 当前状态

当前酒店域主链路已具备基础查询能力，能够完成：

- 路由到酒店 agent
- 调用酒店查询工具
- 在有结果和无结果场景下返回稳定输出

###  5.5 后续建议

- 将 hotel handoff 统一迁移到与 flight 相同的状态驱动模式
- 明确哪些工具属于 sensitive tools，并接入 HITL
- 完善酒店预订与修改订单的端到端测试

---

## 🚗 6. `car_rental_booking_agent`

###  6.1 角色定位

`car_rental_booking_agent` 是租车域 specialist，负责处理租车相关任务。

### 6.2 主要职责

- 查询租车信息
- 查询可租车辆
- 创建租车订单
- 修改租车订单
- 取消租车订单

###  6.3 典型可调用工具

- `search_car_rentals`
- `book_car_rental`
- `update_car_rental`
- `cancel_car_rental`

###  6.4 当前状态

当前租车智能体已支持：

- 被 `supervisor_agent` 正确路由
- 处理基础查询
- 在无结果场景下返回稳定空结果提示

###  6.5 后续建议

- 对租车订单类写操作补充状态驱动 handoff
- 明确是否需要对订单修改/取消引入 HITL
- 完善回归测试用例

---

## 🎫 7. `trip_booking_agent`

###  7.1 角色定位

`trip_booking_agent` 是旅行产品域 specialist，负责处理旅行产品查询、预订、修改等任务。

###  7.2 主要职责

- 查询旅行推荐
- 查询旅游产品
- 创建旅行订单
- 修改旅行订单
- 取消旅行订单

###  7.3 典型可调用工具

- `search_trip_recommendations`
- `book_excursion`
- `update_excursion`
- `cancel_excursion`

###  7.4 当前状态

当前已完成基础查询链路验证，能够正确响应：

- 现实地点推荐查询
- 不现实地点的空结果返回

###  7.5 后续建议

- 与 `flight_agent` 一样引入更明确的 handoff task schema
- 为订单类操作补充测试与安全控制
- 完善 specialist 内部 prompt 约束

---

## 🌐 8. `research_agent`

###  8.1 角色定位

`research_agent` 负责处理联网搜索、外部知识检索、政策查询等不直接依赖内部业务数据库的任务。

###  8.2 主要职责

- 联网搜索外部信息
- 查询航空政策
- 查询旅行相关公开资料
- 对搜索结果进行整理与总结

###  8.3 典型可调用工具

- `TavilySearch`
- `lookup_policy`
- 其他外部检索类工具

###  8.4 当前状态

当前已具备：

- 正常接收 `supervisor_agent` 路由
- 执行外部检索
- 返回整理后的搜索结果
- 在虚构查询 / 无结果查询下返回空结果提示

###  8.5 后续建议

- 继续提升搜索结果摘要质量
- 对搜索失败、超时、无结果场景做更清晰的兜底设计
- 考虑对外部信息增加引用来源或结果压缩策略

---

## 🧠 9. `fetch_user_info_node`

> 严格来说这不是一个 Agent，而是父图中的普通 Graph Node。  
> 但它在整个系统中起到非常关键的“上下文预注入”作用，因此一并说明。

###  9.1 作用

在正式进入 `supervisor_agent` 之前，先根据：

- `runtime.context.passenger_id`

查询用户当前航班背景信息，并写入：

- `state["user_flight_info"]`

###  9.2 设计意义

它解决的问题是：

- 将用户背景信息提前注入
- 避免 specialist 每次都重复查库
- 不再通过 AIMessage 污染消息流
- 让预取结果以“状态”而不是“对话消息”形式参与后续推理

---

