---
name: commit
description: 提交暂存区代码并推送到当前分支
---

# Git Commit and Push Skill

当用户使用 `/commit` 命令时，执行以下流程将暂存区的代码提交并推送到 GitHub。

## 工作流程

### 1. 检查暂存区状态

使用 Bash 工具执行以下命令：
```bash
git status
git diff --cached --stat
```

检查：
- 是否有暂存的文件（staged files）
- 如果暂存区为空，提示用户先使用 `git add` 添加文件
- 显示即将提交的文件列表

### 2. 分析改动内容

根据暂存区的文件类型，分析改动：

- **模型文件** (`src/model.py`, `src/box_model.py`, `src/layers.py`, `src/box_layers.py`)
  - 读取主要改动，理解修改内容

- **训练器文件** (`src/trainer.py`, `src/box_trainer.py`)
  - 分析训练逻辑变化

- **配置文件** (`config/*.json`)
  - 检查参数调整

- **数据处理** (`src/data.py`)
  - 理解数据处理变化

- **主程序** (`src/main.py`, `src/box_main.py`)
  - 分析流程变化

- **文档/其他** (`.md`, `.txt`, `README`)
  - 文档更新

### 3. 生成 Commit Message

根据改动内容，生成**中文** commit message，遵循项目风格：

**格式规则**：
- 使用中文冒号 `：`（不是英文 `:`）
- 类型标签：`feat`, `fix`, `docs`, `chore`, `test`, `refactor`
- 简洁明了，一句话说清楚改动

**示例**：
```
feat：实现Box嵌入体积正则化功能
fix：修复梯度消失问题
feat：添加动态规则置信度计算
chore：更新kinship数据集配置参数
docs：更新README安装说明
refactor：重构模型初始化逻辑
```

**生成逻辑**：
- 如果修改了模型核心功能 → `feat：` 或 `fix：`
- 如果只是修复 bug → `fix：`
- 如果修改配置 → `chore：`
- 如果修改文档 → `docs：`
- 如果是代码重构 → `refactor：`

### 4. 显示预览并确认

在执行提交前，显示：
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 准备提交的文件：
  - src/box_model.py
  - src/box_layers.py
  - config/box_kinship_config.json

💬 Commit Message：
  feat：实现Box嵌入体积正则化功能

🌿 目标分支：
  rule_box

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

使用 AskUserQuestion 工具询问用户是否继续。

### 5. 执行提交

用户确认后，执行：

```bash
# 提交代码，添加 Co-authored 信息
git commit -m "$(cat <<'EOF'
[生成的 commit message]

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

**注意**：
- 使用 HEREDOC 格式确保格式正确
- 自动添加 Claude 协作标记

### 6. 获取当前分支并推送

```bash
# 获取当前分支名
current_branch=$(git rev-parse --abbrev-ref HEAD)

# 推送到当前分支
git push origin $current_branch
```

### 7. 报告结果

提交成功后，显示：
```
✅ 提交成功！

📊 Commit Hash: abc1234
🌿 分支: rule_box
🚀 已推送到远程仓库

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

如果推送失败，提示可能的原因：
- 远程分支有新提交，需要先 pull
- 网络问题
- 权限问题

## 错误处理

### 暂存区为空
```
⚠️ 暂存区没有文件！

请先使用以下命令添加文件：
  git add <文件名>        # 添加指定文件
  git add .              # 添加所有改动
  git add -u             # 添加已追踪的文件
```

### Commit 失败
- 检查是否有 git hooks 阻止提交
- 检查 commit message 格式

### Push 失败
- 提示用户先执行 `git pull --rebase`
- 或检查网络和权限

## 特殊情况

### 如果用户提供了自定义 message

如果用户使用 `/commit "自定义消息"`，则：
- 直接使用用户提供的消息
- 跳过分析和生成步骤
- 仍然添加 Co-authored 信息
- 继续执行提交和推送

## 注意事项

1. **只提交暂存区文件**：不会自动 `git add`，只提交已经在暂存区的文件
2. **保持项目风格**：生成的 commit message 遵循项目已有的中文风格
3. **安全确认**：推送前让用户确认，避免误操作
4. **Co-authored**：自动添加 Claude 协作标记，符合项目规范
5. **当前分支**：自动推送到当前所在分支，不会切换分支

## 使用示例

```bash
# 场景1：自动生成 commit message
git add src/box_model.py
/commit

# 场景2：使用自定义 message
git add config/
/commit "chore：更新所有配置文件参数"

# 场景3：提交所有改动
git add .
/commit
```
