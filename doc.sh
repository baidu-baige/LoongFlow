# 1. 备份 (安全起见)
cp -r docs docs_backup_$(date +%s)

# 定义一个函数来处理每种语言的目录 (en 和 zh)
restructure_lang() {
    local lang=$1
    local base="docs/$lang"

    echo "正在处理语言目录: $lang ..."

    # 1. 创建新目录结构
    mkdir -p "$base/overview"
    mkdir -p "$base/agents"
    mkdir -p "$base/build"
    mkdir -p "$base/components"
    mkdir -p "$base/reference"

    # 2. 移动现有文件 (如果存在)
    # Quick Start -> Overview
    [ -f "$base/quickstart.md" ] && mv "$base/quickstart.md" "$base/overview/quickstart.md"
    
    # Agents (原 Guide)
    [ -f "$base/guide/general_evolve.md" ] && mv "$base/guide/general_evolve.md" "$base/agents/general_evolve.md"
    [ -f "$base/guide/ml_evolve.md" ] && mv "$base/guide/ml_evolve.md" "$base/agents/ml_evolve.md"

    # Reference (原 SDK)
    [ -f "$base/sdk/sdk.md" ] && mv "$base/sdk/sdk.md" "$base/reference/reference.md"

    # 3. 创建缺失的新文件 (占位符)
    
    # Overview 组
    [ ! -f "$base/overview/features.md" ] && echo "# Features & Guarantees\n\nComing soon..." > "$base/overview/features.md"
    [ ! -f "$base/overview/faq.md" ] && echo "# FAQ\n\nFrequently Asked Questions..." > "$base/overview/faq.md"
    [ ! -f "$base/overview/performance.md" ] && echo "# Performance\n\nPerformance benchmarks..." > "$base/overview/performance.md"
    [ ! -f "$base/overview/design.md" ] && echo "# Design\n\nArchitecture design docs..." > "$base/overview/design.md"

    # Build Agents 组
    [ ! -f "$base/build/get_started.md" ] && echo "# Get Started with Building Agents\n\nHow to start..." > "$base/build/get_started.md"
    [ ! -f "$base/build/build_agent.md" ] && echo "# Build Your Agent\n\nStep by step guide..." > "$base/build/build_agent.md"

    # Components 组
    [ ! -f "$base/components/components.md" ] && echo "# Components\n\nList of components..." > "$base/components/components.md"

    # 4. 清理旧空目录
    rmdir "$base/guide" 2>/dev/null
    rmdir "$base/sdk" 2>/dev/null
    rmdir "$base/core/evolve_agent" 2>/dev/null
    rmdir "$base/core/react_agent" 2>/dev/null
    rmdir "$base/core" 2>/dev/null
    rmdir "$base/community" 2>/dev/null # 假设 community 暂时没用到新结构中，或者你可以保留
}

# 执行重构
restructure_lang "en"
restructure_lang "zh"

echo "目录重构完成！"
