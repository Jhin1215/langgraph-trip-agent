# LangGraph Trip Agent

> 1. 基于 **LangChain v1 + LangGraph** 构建的多智能体出行助手。  
> 2. 项目核心目标并非实现一个简单的对话机器人，而是完成一次面向工程实践的架构迁移：**从旧版“单智能体挂全量工具”重构为新版“Supervisor + Specialist Agents + 状态驱动 Handoff + HITL”多智能体系统。**


## 1. 项目简介

`LangGraph Trip Agent` 是一个面向出行场景的多智能体系统，围绕以下五类任务进行协作处理：

- 航班查询、改签、退票、政策咨询
- 酒店查询、预订、订单修改
- 租车查询、预订、订单修改
- 旅行产品查询、预订、订单修改
- 外部信息检索与联网搜索

项目采用 **Supervisor 模式** 进行多智能体编排：由 `supervisor` 统一负责路由分发，再将任务交给各自的 specialist agent 处理。

本项目真正的重点不在于“工具数量”或“功能堆叠”，而在于解决多智能体系统中的几个核心工程问题：
- 如何从旧版高耦合单智能体架构迁移到低耦合多智能体架构
- 如何实现 **Supervisor 只路由，不执行业务**
- 如何通过 **状态驱动 Handoff** 避免任务意图在消息流中漂移
- 如何将 **HITL（Human-in-the-Loop）** 正式接入敏感操作
- 如何对多轮状态、工具调用、中断恢复进行可验证闭环
---
## 2. 版本演进
本项目并非一次性完成，而是沿着“**先完成多智能体拆分基线，再补上下文工程，最后打通敏感操作 HITL 闭环**”的路径逐步演进。  
结合实际开发过程，可以将当前迭代分为以下三个关键阶段。

### V1：Supervisor + Specialist Agents 静态提示词基线重构

#### 阶段目标
将旧版“单智能体挂全量工具”的高耦合结构，重构为 `supervisor + specialist agents` 的多智能体架构，先建立一个可以稳定运行的结构化基线。

#### 关键改动
- 将系统拆分为：
  - `supervisor`
  - `flight_booking_agent`
  - `hotel_booking_agent`
  - `car_rental_booking_agent`
  - `trip_booking_agent`
  - `research_agent`

- 为 `supervisor` 与各 specialist 分别设计静态提示词
- 明确 `supervisor` 的职责仅为**任务识别与路由分发**
- 各 specialist 只绑定本领域工具，不再承担跨域任务处理
- 完成基础链路测试，验证多智能体拆分后整体流程可运行

#### 解决的问题
这一阶段重点解决的是**架构层面的高耦合问题**：

- 将旧版“路由 + 执行”混杂在同一个 agent 中的问题拆开
- 初步建立起 **Supervisor 只路由、Specialist 只执行业务** 的职责边界
- 为后续 handoff、状态管理、HITL 接入提供了结构基础

#### 阶段局限
虽然多智能体结构已经成型，但这一阶段主要依赖**静态提示词**完成任务理解与分发，尚未系统解决以下问题：

- 不同任务场景下上下文注入能力不足
- specialist 对当前任务意图的把握仍依赖 `messages`
- handoff 只是“跳转控制流”，还没有形成稳定的“任务态传递”

---

### V2：静态 + 动态提示词上下文工程

#### 阶段目标
在多智能体结构已经拆分完成的基础上，进一步增强各 agent 对“当前用户身份、当前时间、当前业务约束、相对时间表达”等上下文信息的理解能力，提升任务识别与工具调用的稳定性。

#### 关键改动
- 在原有静态提示词基础上，引入 `dynamic_prompt`
- 为 `supervisor` 与各 specialist 增加运行时上下文补充能力
- 将 `passenger_id`、当前系统时间、相对时间解释规则等动态信息注入模型上下文
- 优化 specialist 对以下情况的处理：
  - 用户当前身份信息读取
  - 相对时间转显式时间范围
  - 空结果处理
  - 工具返回字段的严格使用
- 完成“静态提示词 + 动态提示词”联合上下文工程重构

#### 解决的问题
这一阶段重点解决的是**模型上下文利用不足的问题**：

- 让 agent 不再只依赖用户最后一句自然语言，而是结合运行时上下文决策
- 提升了多轮交互下的任务理解稳定性
- 使 supervisor 与 specialist 的提示词逻辑从“固定模板”升级为“静态规则 + 动态运行时约束”结合的形式

#### 阶段局限
这一阶段虽然已经强化了上下文工程，但 handoff 仍主要依赖消息流传递任务语义。  
在复杂链路下，specialist 仍可能出现以下问题：

- 看见“已转交”消息，却没有准确继承用户真实意图
- 对 sensitive tools（如退票、改签）没有形成可中断、可审批的执行闭环
- graph 级 interrupt / resume 机制尚未正式打通

### V3：状态驱动 Handoff + HITL 闭环打通

#### 阶段目标
解决多智能体 handoff 过程中“任务意图漂移”的问题，并正式将敏感操作接入 HITL（Human-in-the-Loop），实现 graph 级中断与恢复闭环。

#### 关键改动
- 引入统一状态 `TravelState`
- 不再仅依赖 `messages` 传递 handoff 语义，而是在 handoff 时显式写入：
  - `active_agent`
  - `handoff_task_type`
  - `handoff_user_request`
  - `handoff_slots`
  - `user_flight_info`
- 将 `fetch_user_info_node` 的预取结果从 AIMessage 改为写入 state，避免污染对话上下文，为 `flight agent`显式声明 `state_schema=TravelState`
- 让 specialist 在运行时优先读取 handoff state，而不是依赖转接消息去“猜任务”
- 为敏感工具接入 `HumanInTheLoopMiddleware`：
  - `cancel_ticket`
  - `update_ticket_to_new_flight`
- 打通完整链路：
  - specialist 产出 sensitive tool call
  - graph 触发 interrupt
  - 人工输入 `approve / reject`
  - graph 使用同一 `thread_id` 恢复执行
  - 工具继续执行或被拒绝执行

#### 解决的问题
这一阶段是项目中最关键的一次升级，正式解决了以下几个核心工程问题：

1. **解决 handoff 只跳转、不传意图的问题**
   - specialist 不再只看到“handoff 成功 ToolMessage”和 “最后一条 AIMessage”
   - 而是能直接读取 `task_type + slots` 等结构化任务态

2. **解决 sensitive tools 无法受控执行的问题**
   - 改签、退票等高风险工具执行前可以被人工审批拦截
   - `approve / reject` 两条恢复路径均已验证可用

3. **解决 graph 级状态恢复问题**
   - 基于统一 `thread_id + checkpointer` 打通 interrupt / resume 闭环
   - 验证了多轮状态、工具调用与人工审批在同一图中的连续性

#### 阶段成果
截至当前版本，`flight` 主链路已经完成端到端验证：

- supervisor 正确识别取消机票任务
- handoff 正确传递 `cancel_ticket + ticket_no`
- `flight_booking_agent` 正确消费 handoff state
- `cancel_ticket` 成功触发 HITL 中断
- `approve / reject` 路径均可正确恢复并完成后续处理

三个阶段的演进逻辑

将上述三个版本连起来看，本项目的演进主线非常明确：

- **Version 1** 解决“结构先拆开”的问题，完成多智能体架构基线重构

- **Version 2** 解决“上下文怎么补进去”的问题，完成静态 + 动态提示词上下文工程

- **Version 3** 解决“任务怎么稳定传下去、敏感操作怎么可控执行”的问题，完成状态驱动 handoff 与 HITL 闭环

因此，本项目的核心价值并不只是“做了一个多智能体助手”，而是完整验证了一条面向工程实践的演进路径：
> **单智能体高耦合原型 → 多智能体职责拆分 → 静态 + 动态上下文工程 → 状态驱动 handoff → graph 级 HITL 中断与恢复闭环**
---

## 3. 项目结构
```
xiec_assistant/
├── assets/faq/org_faq.json # 航空公司政策
├── src/
│   └── deep_agent/
│       ├── agents/
│       │   ├── supervisor.py
│       │   ├── flight_agent.py
│       │   ├── hotel_agent.py
│       │   ├── car_rent_agent.py
│       │   ├── trip_agent.py
│       │   ├── research_agent.py
│       │   └── common.py
│       ├── tools/
│       │   ├── flights_tools.py
│       │   ├── hotels_tools.py
│       │   ├── cars_tools.py
│       │   ├── trip_tools.py
│       │   └── retrieve_tools.py
│       ├── context.py  # 自定义上下文
│       ├── state.py    # 自定义状态
│       ├── llms.py
│       └── embeddings.py   # 统一管理向量嵌入模型（Embedding Model）的初始化与封装。 
│       └── env_uitl.py   # 统一加载和管理环境变量。  
│       └── init_db.py   # 负责数据库初始化、测试数据库准备，对齐目前时间。
│       └── retrivers.py   # 实现项目航空政策中的检索 
│       └── config.py   # 项目中用到的常量  
├── tests/
│   ├── integration_tests/
│   │   ├── __init__.py
│   │   └── regression_test.py         # 集成 / 回归测试
│   ├── manual/
│   │   ├── __init__.py
│   │   ├── chat_test.py               # 手动多轮对话测试脚本
│   │   └── ctrip多智能体测试说明.md    # 手测说明文档
│   ├── unit_tests/                    # 单元测试目录
│   └── conftest.py                    # pytest 全局测试配置
├── .env                               # 环境变量配置
```
---

## 4. 项目使用说明
### 4.1 环境准备
项目依赖的三方库已存放于`requirements.txt`中，请根据自身环境进行安装。
```
# 1.conda
conda create -n xiec_assistant python=3.11 -y
conda activate xiec_assistant
pip install -r requirements.txt

# 2.venv / virtualenv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3.uv
uv venv
uv pip install -r requirements.txt
```
### 4.2 环境变量配置
在正式运行项目前，需要先完成环境变量配置。
建议在项目根目录创建 .env 文件，并补充模型调用相关`API_KEY`和`BASE_URL`配置，例如：
```
ZHIPU_API_KEY=x

DEEPSEEK_API_KEY=x
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

QWEN_API_KEY=x
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

TAVILY_API_KEY=x
```
其他项目运行所需的 Provider Key

这一步的作用是确保各 agents、tools 和检索模块能够正常调用外部服务。

### 4.3 初始化测试数据库
本项目的手动测试与回归验证依赖本地测试数据库。
在开始测试前，需要先运行：
> ```python src/deep_agent/init_db.py```

该脚本主要负责：

初始化测试数据库
准备出行场景相关业务数据
将数据库中的时间节点对齐到当前测试时间附近

这一步非常重要。
如果不先更新时间相关数据，航班查询、改签、退票等流程可能因为测试数据时间过期而无法正常触发。

### 4.4 手动对话测试
完成环境准备和数据库初始化后，可以直接运行手动测试脚本：

python tests/manual/chat_test.py

该脚本会启动一个命令行交互式测试循环，用于模拟真实用户与多智能体图的交互过程。

手动测试时，建议结合 tests/manual/ctrip多智能体测试说明.md 中的测试说明进行验证，重点关注以下几类链路：
- 当前机票信息查询 
- 航班搜索 
- 改签成功 / 改签失败 
- 取消机票 
- sensitive tools 的 HITL 中断与恢复
### 4.5 HITL（人工审批）测试方式

对于退票、改签等敏感操作，系统已经接入 HumanInTheLoopMiddleware。
当测试过程中触发敏感工具时，命令行会进入中断状态，并提示输入审批结果：
```
approve
reject
```

例如，在执行取消机票操作时，graph 会先触发 interrupt，等待人工输入审批结果，再继续执行后续流程。

这一步主要用于验证：
- specialist 是否正确触发 sensitive tool call 
- graph 是否正确产生 interrupt 
- approve / reject 是否能基于同一 thread_id 正确恢复执行
### 4.6 回归测试与后续扩展

当前项目已经围绕 flight 主链路完成了较完整的手动验证。
后续可以进一步结合 tests/integration_tests/regression_test.py 补充自动化回归测试，用于稳定验证以下能力：

多轮短期记忆 
- supervisor 路由正确性 
- specialist 工具调用正确性 
- HITL interrupt / resume 闭环