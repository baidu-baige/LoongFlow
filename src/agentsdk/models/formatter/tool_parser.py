#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file provides a tool parser for Model.
"""
#
# import json
# import re
#
# from fuzzywuzzy import process
#
# from agentsdk.logger import get_logger
#
# logger = get_logger(__name__)
#
#
# def qianfan_tool_parser(text):
#     """match every content between each <tool_call_begin> and <tool_call_end> block."""
#     tool_calls_start_token = "<tool▁calls▁begin>"
#     if tool_calls_start_token not in text:
#         return [], text
#
#     tool_call_regrex = re.compile(
#         r"<tool▁call▁begin>function<tool▁sep>(?P<function_name>.*)\n"
#         + r"```json\n(?P<function_arguments>.*)\n```<tool▁call▁end>"
#     )
#
#     try:
#         function_call_tuples = tool_call_regrex.findall(text)
#         logger.info(
#             f"Found {len(function_call_tuples)} tool calls using QianFan Tool Parser."
#         )
#         tool_calls = []
#         for match in function_call_tuples:
#             function_name, function_arguments = match
#             tool_calls.append(
#                 {
#                     "tool_type": "function",
#                     "tool_name": function_name,
#                     "tool_parameter": json.loads(function_arguments),
#                 }
#             )
#
#         # Extract content before first tool call
#         first_tool_idx = text.find(tool_calls_start_token)
#         content = text[:first_tool_idx] if first_tool_idx > 0 else ""
#         return tool_calls, content
#     except Exception as e:
#         logger.error(f"Error parsing QianFan tool calls: {e}")
#         return [], text
#
#
# def deepseekv3_tool_parser(text):
#     """match every content between each <｜tool▁call▁begin｜> and <｜tool▁call▁end｜> block"""
#     tool_calls_start_token = "<｜tool▁calls▁begin｜>"
#     if tool_calls_start_token not in text:
#         return [], text
#
#     tool_call_regrex = re.compile(
#         r"<｜tool▁call▁begin｜>function<｜tool▁sep｜>(?P<function_name>.*)\n"
#         + r"```json\n(?P<function_arguments>.*)\n```<｜tool▁call▁end｜>"
#     )
#
#     try:
#         function_call_tuples = tool_call_regrex.findall(text)
#         logger.info(
#             f"Found {len(function_call_tuples)} tool calls using DeepSeekV3 Tool Parser."
#         )
#
#         tool_calls = []
#         for match in function_call_tuples:
#             function_name, function_arguments = match
#             tool_calls.append(
#                 {
#                     "tool_type": "function",
#                     "tool_name": function_name,
#                     "tool_parameter": json.loads(function_arguments),
#                 }
#             )
#
#         # Extract content before first tool call
#         first_tool_idx = text.find(tool_calls_start_token)
#         content = text[:first_tool_idx] if first_tool_idx > 0 else ""
#         return tool_calls, content
#     except Exception as e:
#         logger.error(f"Error parsing DeepSeekV3 tool calls: {e}")
#         return [], text
#
#
# def deepseekv31_tool_parser(text):
#     """match every content between each <｜tool▁call▁begin｜> and <｜tool▁call▁end｜> block"""
#     tool_calls_start_token = "<｜tool▁calls▁begin｜>"
#     if tool_calls_start_token not in text:
#         return [], text
#
#     tool_call_regrex = re.compile(
#         r"<｜tool▁call▁begin｜>(?P<function_name>.*?)<｜tool▁sep｜>(?P<function_arguments>.*?)<｜tool▁call▁end｜>"
#     )
#
#     try:
#         function_call_tuples = tool_call_regrex.findall(text)
#         logger.info(
#             f"Found {len(function_call_tuples)} tool calls using DeepSeekV3.1 Tool Parser."
#         )
#         tool_calls = []
#         for match in function_call_tuples:
#             function_name, function_arguments = match
#             tool_calls.append(
#                 {
#                     "tool_type": "function",
#                     "tool_name": function_name,
#                     "tool_parameter": json.loads(function_arguments),
#                 }
#             )
#
#         # Extract content before first tool call
#         first_tool_idx = text.find(tool_calls_start_token)
#         content = text[:first_tool_idx] if first_tool_idx > 0 else ""
#         return tool_calls, content
#     except Exception as e:
#         logger.error(f"Error parsing DeepSeekV3.1 tool calls: {e}")
#         return [], text
#
#
# def deepseekv32_tool_parser(text):
#     """match every content between each <｜DSML｜function_calls> and </｜DSML｜function_calls> block"""
#     tool_calls_start_token = "<｜DSML｜function_calls>"
#     if tool_calls_start_token not in text:
#         return [], text
#
#     tool_call_complete_regex = re.compile(
#         r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>", re.DOTALL
#     )
#     invoke_complete_regex = re.compile(
#         r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>', re.DOTALL
#     )
#     parameter_complete_regex = re.compile(
#         r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(?:true|false)"\s*>(.*?)</｜DSML｜parameter>',
#         re.DOTALL,
#     )
#
#     def _parse_invoke_params(invoke_str: str) -> dict | None:
#         param_dict = dict()
#         for param_name, param_val in parameter_complete_regex.findall(invoke_str):
#             param_dict[param_name] = param_val
#         return param_dict
#
#     try:
#         tool_calls = []
#         # Find all complete tool_call blocks
#         for tool_call_match in tool_call_complete_regex.findall(text):
#             # Find all invokes within this tool_call
#             for invoke_name, invoke_content in invoke_complete_regex.findall(
#                 tool_call_match
#             ):
#                 param_dict = _parse_invoke_params(invoke_content)
#                 tool_calls.append(
#                     {
#                         "tool_type": "function",
#                         "tool_name": invoke_name,
#                         "tool_parameter": param_dict,
#                     }
#                 )
#
#         if not tool_calls:
#             return [], text
#
#         logger.info(
#             f"Found {len(tool_calls)} tool calls using DeepSeekV3.2 Tool Parser."
#         )
#
#         # Extract content before first tool call
#         first_tool_idx = text.find(tool_calls_start_token)
#         content = text[:first_tool_idx] if first_tool_idx > 0 else ""
#         return tool_calls, content
#
#     except Exception as e:
#         logger.error(f"Error parsing DeepSeekV3.2 tool calls: {e}")
#         return [], text
#
#
# def deepseekr1_tool_parser(text):
#     """match every content between each <｜tool▁call▁begin｜> and <｜tool▁call▁end｜> block"""
#     tool_calls_start_token = "<｜tool▁calls▁begin｜>"
#     if tool_calls_start_token not in text:
#         return [], text
#
#     tool_call_regrex = re.compile(
#         r"<｜tool▁call▁begin｜>function<｜tool▁sep｜>"
#         + r"(?P<function_name>.*)\n```json\n(?P<function_arguments>.*)\n```<｜tool▁call▁end｜>"
#     )
#
#     try:
#         function_call_tuples = tool_call_regrex.findall(text)
#         logger.info(
#             f"Found {len(function_call_tuples)} tool calls using DeepSeekR1 Tool Parser."
#         )
#         tool_calls = []
#         for match in function_call_tuples:
#             function_name, function_arguments = match
#             tool_calls.append(
#                 {
#                     "tool_type": "function",
#                     "tool_name": function_name,
#                     "tool_parameter": json.loads(function_arguments),
#                 }
#             )
#
#         # Extract content before first tool call
#         first_tool_idx = text.find(tool_calls_start_token)
#         content = text[:first_tool_idx] if first_tool_idx > 0 else ""
#         return tool_calls, content
#     except Exception as e:
#         logger.error(f"Error parsing DeepSeekR1 tool calls: {e}")
#         return [], text
#
#
# def tool_parser(text: str, model_url: str, model_name: str):
#     """Choose the right tool parser based on the given model name"""
#     template_names = ["deepseekv3", "deepseekv31", "deepseekv32", "deepseekr1"]
#
#     def find_closest_template(model_name, template_names):
#         """Find the most similar string in `template_names` to `model_name`"""
#         closest_match = process.extractOne(model_name, template_names)
#         return closest_match[0]
#
#     closet_template = find_closest_template(model_name, template_names)
#
#     if closet_template == "deepseekv3":
#         return deepseekv3_tool_parser(text)
#     elif closet_template == "deepseekv31":
#         return deepseekv31_tool_parser(text)
#     elif closet_template == "deepseekv32":
#         return deepseekv32_tool_parser(text)
#     elif closet_template == "deepseekr1":
#         return deepseekr1_tool_parser(text)
#
#     if "qianfan.baidubce.com" in model_url:
#         return qianfan_tool_parser(text)
#     else:
#         raise ValueError("Unsupported tool parser.")
