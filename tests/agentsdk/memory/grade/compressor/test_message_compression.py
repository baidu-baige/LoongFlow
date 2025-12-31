#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file generate test message to compress
"""

import uuid

from src.agentsdk.message import (
    ContentElement,
    Message,
    MimeType,
    Role,
    ThinkElement,
    ToolCallElement,
    ToolOutputElement,
    ToolStatus,
)


def generate_test_messages():
    """Generate test messages simulating React workflows with curl and bash commands"""

    messages = []

    # React Workflow 1: Curl website analysis
    # =========================================

    # 1. User request to fetch website data
    user_request_1 = Message.from_text(
        data="Please fetch the latest data from https://api.github.com/repos/octocat/Hello-World and analyze the repository statistics.",
        sender="User",
        role=Role.USER,
        conversation_id="react_workflow_1",
    )
    messages.append(user_request_1)

    # 2. Agent thinking about the approach
    think_1 = Message(
        role=Role.ASSISTANT,
        sender="ReactAgent",
        conversation_id="react_workflow_1",
        content=[
            ThinkElement(
                content={
                    "reasoning": "The user wants GitHub repository data. I need to use curl to fetch the API endpoint and then analyze the JSON response.",
                    "plan": [
                        "Use curl to fetch https://api.github.com/repos/octocat/Hello-World",
                        "Parse the JSON response to extract key statistics",
                        "Analyze repository metrics like stars, forks, issues",
                        "Provide insights based on the data",
                    ],
                    "considerations": [
                        "Handle potential API rate limits",
                        "Ensure proper error handling for network issues",
                        "Validate JSON response format",
                    ],
                }
            )
        ],
    )
    messages.append(think_1)

    # 3. Tool call to execute curl command
    call_id = uuid.uuid4()
    tool_call_1 = Message(
        role=Role.ASSISTANT,
        sender="ReactAgent",
        conversation_id="react_workflow_1",
        content=[
            ToolCallElement(
                call_id=call_id,
                target="execute_shell_command",
                arguments={
                    "command": "curl -s -H 'Accept: application/json' https://api.github.com/repos/octocat/Hello-World",
                    "description": "Fetch GitHub repository data using curl",
                    "timeout": 30,
                },
            )
        ],
    )
    messages.append(tool_call_1)

    # 4. Tool output with curl results
    tool_output_1 = Message(
        role=Role.TOOL,
        sender="shell_executor",
        conversation_id="react_workflow_1",
        content=[
            ToolOutputElement(
                call_id=call_id,
                tool_name="execute_shell_command",
                status=ToolStatus.SUCCESS,
                result=[
                    ContentElement(
                        mime_type=MimeType.APPLICATION_JSON,
                        data={
                            "command": "curl -s -H 'Accept: application/json' https://api.github.com/repos/octocat/Hello-World",
                            "exit_code": 0,
                            "stdout": {
                                "id": 1296269,
                                "name": "Hello-World",
                                "full_name": "octocat/Hello-World",
                                "private": False,
                                "owner": {
                                    "login": "octocat",
                                    "id": 583231,
                                    "type": "User",
                                },
                                "description": "My first repository on GitHub!",
                                "fork": False,
                                "created_at": "2011-01-26T19:01:12Z",
                                "updated_at": "2024-10-15T14:20:15Z",
                                "pushed_at": "2024-09-20T08:45:25Z",
                                "size": 19,
                                "stargazers_count": 1850,
                                "watchers_count": 1850,
                                "language": "Assembly",
                                "forks_count": 1560,
                                "open_issues_count": 9,
                                "license": {"key": "mit", "name": "MIT License"},
                                "topics": ["example", "hello-world", "octocat"],
                                "default_branch": "master",
                            },
                            "stderr": "",
                            "execution_time": "1.2s",
                        },
                    )
                ],
            )
        ],
    )
    messages.append(tool_output_1)

    # 5. Agent analysis of curl results
    analysis_1 = Message.from_text(
        data="""Based on the GitHub API response for the octocat/Hello-World repository:

Repository Analysis:
- **Basic Info**: Repository created on January 26, 2011, last updated October 15, 2024
- **Popularity**: 1,850 stars and 1,560 forks indicate high community interest
- **Activity**: 9 open issues, last push was September 20, 2024
- **Technology**: Written in Assembly language
- **License**: MIT License, allowing open usage

Key Insights:
1. This is a well-established repository with over 13 years of history
2. The high star count suggests it's used as an example or learning resource
3. Recent activity shows the repository is still maintained
4. Assembly language usage is unusual for a Hello-World example, making it unique

The repository serves as Octocat's first GitHub repository and appears to be maintained as a historical example.""",
        sender="ReactAgent",
        role=Role.ASSISTANT,
        conversation_id="react_workflow_1",
    )
    messages.append(analysis_1)

    # React Workflow 2: Bash command execution and analysis
    # ======================================================

    # 6. User request to analyze system performance
    user_request_2 = Message.from_text(
        data="Please analyze the current system performance by checking disk usage, memory consumption, and running processes. Provide a comprehensive system health report.",
        sender="User",
        role=Role.USER,
        conversation_id="react_workflow_2",
    )
    messages.append(user_request_2)

    # 7. Agent thinking about system analysis approach
    think_2 = Message(
        role=Role.ASSISTANT,
        sender="ReactAgent",
        conversation_id="react_workflow_2",
        content=[
            ThinkElement(
                content={
                    "reasoning": "The user wants a system performance analysis. I need to execute multiple bash commands to gather system metrics and then synthesize the information.",
                    "plan": [
                        "Use df -h to check disk usage",
                        "Use free -h to check memory consumption",
                        "Use top -bn1 to get process snapshot",
                        "Combine results into a comprehensive report",
                        "Identify potential issues and recommendations",
                    ],
                    "considerations": [
                        "Handle different Linux distributions",
                        "Parse command output correctly",
                        "Provide actionable recommendations",
                    ],
                }
            )
        ],
    )
    messages.append(think_2)

    # 8. Tool call to execute system analysis commands
    call_id = uuid.uuid4()
    tool_call_2 = Message(
        role=Role.ASSISTANT,
        sender="ReactAgent",
        conversation_id="react_workflow_2",
        content=[
            ToolCallElement(
                call_id=call_id,
                target="execute_shell_command",
                arguments={
                    "command": "df -h && echo '---' && free -h && echo '---' && top -bn1 | head -20",
                    "description": "Execute system performance analysis commands",
                    "timeout": 10,
                },
            )
        ],
    )
    messages.append(tool_call_2)

    # 9. Tool output with system analysis results
    tool_output_2 = Message(
        role=Role.TOOL,
        sender="shell_executor",
        conversation_id="react_workflow_2",
        content=[
            ToolOutputElement(
                call_id=call_id,
                tool_name="execute_shell_command",
                status=ToolStatus.SUCCESS,
                result=[
                    ContentElement(
                        mime_type=MimeType.TEXT_PLAIN,
                        data="""Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        20G   12G  7.2G  62% /
/dev/sdb1       100G   45G   50G  47% /data
---
              total        used        free      shared  buff/cache   available
Mem:           16Gi       4.2Gi       8.1Gi       245Mi       3.7Gi        11Gi
Swap:         2.0Gi       128Mi       1.9Gi
---
top - 14:30:25 up 15 days,  3:45,  1 user,  load average: 0.15, 0.12, 0.09
Tasks: 245 total,   1 running, 244 sleeping,   0 stopped,   0 zombie
%Cpu(s):  2.3 us,  0.7 sy,  0.0 ni, 97.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem :  16384.0 total,   4326.4 free,   4291.2 used,   7766.4 buff/cache
MiB Swap:   2048.0 total,   1958.2 free,     89.8 used.  11264.0 avail Mem 

    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
    1 root      20   0  169212  10832   8160 S   0.0   0.1   0:12.45 systemd
    2 root      20   0       0      0      0 S   0.0   0.0   0:00.03 kthreadd
    3 root       0 -20       0      0      0 I   0.0   0.0   0:00.00 rcu_gp""",
                    )
                ],
            )
        ],
    )
    messages.append(tool_output_2)

    # 10. Agent analysis of system performance
    analysis_2 = Message.from_text(
        data="""System Performance Analysis Report:

Disk Usage Analysis:
- Root partition (/): 12GB used of 20GB (62%) - Healthy, good available space
- Data partition (/data): 45GB used of 100GB (47%) - Optimal usage level

Memory Consumption:
- Total RAM: 16GB
- Used: 4.2GB (26%) - Low memory usage, system has ample resources
- Available: 11GB - Excellent memory availability
- Swap: 128MB used of 2GB (6%) - Minimal swap usage, indicating good memory management

Process Analysis:
- System uptime: 15 days - Stable system operation
- Load average: 0.15, 0.12, 0.09 - Very low system load
- CPU usage: 2.3% user, 0.7% system, 97.0% idle - Excellent CPU performance
- Total processes: 245, with only 1 running - System is mostly idle

Overall System Health: EXCELLENT
- Disk usage is within optimal ranges
- Memory consumption is low with high availability  
- CPU utilization is minimal with low load averages
- System stability confirmed by 15-day uptime

Recommendations:
- No immediate action required
- Consider monitoring /data partition as it approaches 50% usage
- Current configuration supports additional workload""",
        sender="ReactAgent",
        role=Role.ASSISTANT,
        conversation_id="react_workflow_2",
    )
    messages.append(analysis_2)

    return messages
