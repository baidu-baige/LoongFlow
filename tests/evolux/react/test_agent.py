# -*- coding: utf-8 -*-
"""
This file define a common react work paradigm.
"""
from typing import Dict, List, Optional

import pytest
from pydantic import BaseModel, Field

from agentsdk.message import Message
from agentsdk.models import LiteLLMModel
from agentsdk.tools import LsTool, ReadTool, ShellTool, TodoReadTool, TodoWriteTool, Toolkit, WriteTool
from evolux.react import ReActAgent


class LocationInfo(BaseModel):
    city: str = Field(..., description="The city where the main campus is located.")
    state_province: str = Field(..., description="The state or province of the university.")
    country: str = Field(..., description="The country where the university is located.")


# The main, comprehensive model for a university profile
class UniversityProfile(BaseModel):
    """A comprehensive model to hold detailed information about a university."""
    # Basic Information
    full_name: str = Field(..., description="The full official name of the university.")
    founding_year: int = Field(..., description="The year the university was officially founded.")
    location: LocationInfo = Field(..., description="Geographical location of the university.")
    university_type: str = Field("Unknown", description="Type of university (e.g., 'Private', 'Public', 'Ivy League').")

    # Academic Profile
    ranking: int = Field(..., description="The university's national or global ranking number.")
    major_achievements: List[str] = Field(default_factory=list,
                                          description="A list of significant achievements or awards.")
    colleges_and_schools: List[str] = Field(default_factory=list,
                                            description="List of main colleges, schools, or faculties.")
    student_faculty_ratio: Optional[str] = Field(None,
                                                 description="The ratio of students to faculty members, e.g., '15:1'.")
    total_students: Optional[int] = Field(None,
                                          description="Total number of enrolled students (undergraduate and graduate).")

    # Campus & Culture
    official_website: str = Field(..., description="The complete URL of the university's official website.")


# Final output structure
class Output(BaseModel):
    """The final output model, containing a dictionary of university profiles."""
    universities: Dict[str, UniversityProfile] = Field(
            default_factory=dict,
            description="A dictionary mapping a university's common name to its detailed profile."
    )


@pytest.mark.asyncio
async def test_run():
    """Create ReActAgent instance"""
    model = LiteLLMModel(
            model_name="deepseek-v3",
            base_url="https://qianfan.baidubce.com/v2",
            api_key="xxx",
    )

    toolkit = Toolkit()
    toolkit.register_tool(LsTool())
    toolkit.register_tool(ShellTool())
    toolkit.register_tool(TodoReadTool())
    toolkit.register_tool(TodoWriteTool())
    toolkit.register_tool(ReadTool())
    toolkit.register_tool(WriteTool())

    agent = ReActAgent.create_default(model=model,
                                      sys_prompt="You are a helpful assistant. When task complete, you must call "
                                                 "generate_final_answer function",
                                      output_format=Output,
                                      toolkit=toolkit,
                                      parallel_tool_run=True,
                                      max_steps=32)

    agent.register_hook("pre_run", lambda *args, **kwargs: print("hello before run"))
    agent.register_hook("post_run", lambda *args, **kwargs: print("byebye after run"))
    agent.register_hook("pre_reason", lambda *args, **kwargs: print("hello before reason"))
    agent.register_hook("pre_act", lambda *args, **kwargs: print("hello before act"))
    agent.register_hook("post_observe", lambda *args, **kwargs: print("byebye after observe"))
    agent.register_hook("post_run", lambda *args, **kwargs: print("byebye after run"))
    message = await agent.run(Message.from_text(
            "On the website https://180416.xyz, there is a self-introduction. "
            "Please read it and provide me with information about the university mentioned in it."
    ))

    print(message)
